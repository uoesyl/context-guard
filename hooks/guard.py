"""context-guard hook (PreInvocation + PreToolUse).

Detects Antigravity auto-compaction (CHECKPOINT steps) in transcript_full.jsonl and
injects a recovery pack (verbatim user requests, task_state.md, edited files, recall
usage). Also warns before the compaction threshold based on real prompt tokens.

Never blocks the agent: any failure prints {} and logs to guard.log.
"""
from datetime import datetime
import json
import os
import re
import sys
import time
import traceback

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

PLUGIN_DIR = os.path.dirname(HERE)
RECALL = os.path.join(PLUGIN_DIR, "scripts", "recall.py")
STATS = os.path.join(PLUGIN_DIR, "scripts", "context_stats.py")
RULE_FILE = os.path.join(PLUGIN_DIR, "rules", "AGENTS.md")
# Registered in the global ~/.gemini/config/hooks.json (plugin hooks do not run in
# swarm workers), so PLUGIN_DATA is usually unset: use a fixed data dir.
DATA_DIR = os.environ.get("CONTEXT_GUARD_DATA") or os.path.join(
    os.path.expanduser("~"), ".gemini", "antigravity", "plugin_data", "context-guard")
SWARM_WORKER_MARK = "Your assignment (recorded on the board"

# Thresholds are model-aware and self-calibrating (see calib.py). The resolved levels are
# written into every state file (`levels`); the context_meter sidecar reads them from there.
import calib as K  # noqa: E402
import usage as U  # noqa: E402

# Defaults for an unknown model (Gemini's 512k window): 350k / 420k warnings.
LEVELS_INFO = K.levels_for({}, None)
TOKEN_LEVELS = [tuple(x) for x in LEVELS_INFO["token"]]
CHAR_LEVELS = [tuple(x) for x in LEVELS_INFO["char"]]
COMPACTION_TARGET_TOKENS = LEVELS_INFO["target"]
COMPACTION_RANGE = tuple(LEVELS_INFO["range"])

USER_BUDGET = 10_000
LAST_FULL = 3
LAST_FULL_CAP = 3_000
STATE_FILE_CAP = 8_000
MAX_FILES = 15
PACK_MAX_CHARS = 42_000
TS_STALE_STEPS = 20
TS_STALE_RESPONSES = 5
TS_MTIME_GRACE_SEC = 60


LOG_MAX_BYTES = 2 * 1024 * 1024
CONV_ID_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def is_valid_conv_id(conv):
    return isinstance(conv, str) and conv not in (".", "..") and bool(CONV_ID_SAFE_RE.fullmatch(conv))


def log(msg):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        p = os.path.join(DATA_DIR, "guard.log")
        try:
            if os.path.exists(p) and os.path.getsize(p) > LOG_MAX_BYTES:
                os.replace(p, p + ".1")
        except OSError:
            pass
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def state_path(conv):
    d = os.path.join(DATA_DIR, "state")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{conv}.json")


def load_state(conv):
    try:
        with open(state_path(conv), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(conv, st):
    p = state_path(conv)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, p)


def read_new(path, offset):
    """Read complete lines after offset. Returns (objects, new_offset)."""
    size = os.path.getsize(path)
    if size < offset:  # file rewritten/truncated: rescan
        offset = 0
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    objs = []
    for raw in data[: end + 1].splitlines():
        try:
            objs.append(json.loads(raw.decode("utf-8", errors="replace")))
        except Exception:
            continue
    return objs, offset + end + 1


def render_users(users, max_head_chars=None):
    if not users:
        return "（无）"
    lines = []
    tail = users[-LAST_FULL:]
    head = users[:-LAST_FULL] if len(users) > LAST_FULL else []
    tail_txt = [(s, C.head_tail(t, LAST_FULL_CAP)) for s, t in tail]
    used = sum(len(t) for _, t in tail_txt)
    per = max((USER_BUDGET - used) // max(len(head), 1), 160)
    if max_head_chars is None:
        for i, (s, t) in enumerate(head, 1):
            lines.append(f"{i}. [step {s}]\n{C.quote(C.head_tail(t, per))}")
    elif head:
        if max_head_chars <= 0:
            lines.append(f"（更早的 {len(head)} 条用户原话因总长度上限已省略，用 recall.py users 查看）")
        else:
            per_cap = min(per, max(max_head_chars // max(len(head), 1), 80))
            kept, total = [], 0
            for idx in range(len(head) - 1, -1, -1):
                s, t = head[idx]
                entry = f"{idx + 1}. [step {s}]\n{C.quote(C.head_tail(t, per_cap))}"
                if total + len(entry) + (1 if kept else 0) > max_head_chars:
                    break
                kept.append(entry)
                total += len(entry) + 1
            kept.reverse()
            omitted = len(head) - len(kept)
            if omitted > 0:
                lines.append(f"（更早的 {omitted} 条用户原话因总长度上限已省略，用 recall.py users 查看）")
            lines.extend(kept)
    for j, (s, t) in enumerate(tail_txt, len(head) + 1):
        lines.append(f"{j}. [step {s}]\n{C.quote(t)}")
    return "\n".join(lines)




MUTATING_TOOLS = {
    "write_file", "edit_file", "replace_file_content", "multi_replace_file_content", "write_to_file",
    "run_command", "invoke_subagent", "send_message", "post_message", "swarm", "run_workflow",
    "call_mcp_tool", "define_subagent", "manage_subagents",
}
PLAN_CAP = 3_000
# More model calls than this after a pending checkpoint = we are catching up after the
# hook was not running; do not deny tools then (inject via PreInvocation instead).
STALE_RESPONSES = 5
NEUTRAL = {"decision": "ask"}  # "ask" respects the user's permission preset; {} would DENY
DENY_MARKER = "【Context Guard 拦截】刚刚发生"


def latest_plan(full_path, conv, rows=None):
    """Return (plan_step, plan_text, report_text) from exit/enter_plan_mode calls."""
    plan = (None, None)
    report = None
    for o in (C.iter_active_steps(full_path, light=True) if rows is None else rows):
        for tc in o.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            a = tc.get("args") or {}
            if tc.get("name") == "exit_plan_mode" and a.get("plan"):
                plan, report = (o.get("step_index"), a["plan"]), None
            elif tc.get("name") == "enter_plan_mode" and a.get("reason"):
                report = a["reason"]
    if plan[1]:
        try:
            d = os.path.join(DATA_DIR, "plans")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{conv}.md"), "w", encoding="utf-8") as f:
                f.write(plan[1])
        except Exception:
            pass
    return plan[0], plan[1], report


def _parse_iso_ts(s):
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _build_agent_parts(col, has_users, budget_in=4000, budget_board=3000, budget_dispatch=2000,
                       omit_incoming=False, omit_dispatched=False, omit_board=False, drop_outgoing=False):
    raw = C.render_agent_sections(col, has_users=has_users, budget_in=budget_in,
                                  budget_board=budget_board, budget_dispatch=budget_dispatch)
    out = []
    for i in range(0, len(raw), 2):
        h, b = raw[i], raw[i + 1] if i + 1 < len(raw) else ""
        if h.startswith("### 你最近发给") and drop_outgoing:
            continue
        if h.startswith("### 你派发过") and omit_dispatched:
            b = "（因总长度上限已省略，用 recall.py messages 查看）"
        elif h.startswith("### swarm 消息板") and omit_board:
            b = "（因总长度上限已省略，用 read_board{since:0} 或 recall.py messages 查看）"
        elif h.startswith("### 其他代理") and omit_incoming:
            b = "（因总长度上限已省略，用 recall.py messages 查看）"
        out += [h, b]
    return out


def build_recovery(full_path, artifact_dir, ckpt_step, conv="", manual=False):
    users, edited = [], []
    col = C.AgentCollector(conv_dir=C.conv_dir_for_transcript(full_path))
    rows = C.iter_active_steps(full_path, light=True)
    ts_path = os.path.join(artifact_dir, "task_state.md") if artifact_dir else ""
    ts_norm = os.path.normcase(os.path.abspath(ts_path)) if ts_path else ""
    ts_step, ts_step_at, ts_resp_after = None, None, 0
    for o in rows:
        typ = o.get("type")
        if typ == "USER_INPUT":
            users.append((o.get("step_index"), C.user_text(o)))
        ep_list = C.edited_paths(o)
        edited += ep_list
        if typ == "PLANNER_RESPONSE":
            if ts_step is not None:
                ts_resp_after += 1
            for ep in ep_list:
                ep_str = str(ep)
                hit = ((os.path.normcase(os.path.abspath(ep_str)) == ts_norm)
                       if (ts_norm and os.path.isabs(ep_str))
                       else (os.path.basename(ep_str.replace("\\", "/")).lower() == "task_state.md"))
                if hit:
                    ts_step = C._sidx(o)
                    ts_step_at = _parse_iso_ts(o.get("created_at"))
                    ts_resp_after = 0
        col.feed(o)
    col.load_board_files()

    ts_heading = "### task_state.md"
    state_body = f"（不存在。请现在在 {artifact_dir} 下创建 task_state.md）"
    if ts_path and os.path.isfile(ts_path):
        try:
            mtime = os.path.getmtime(ts_path)
        except OSError:
            mtime = None
        mtime_str = time.strftime("%m-%d %H:%M", time.localtime(mtime)) if mtime else "未知"
        with open(ts_path, encoding="utf-8", errors="replace") as f:
            state_md = C.quote(C.head_tail(f.read(), STATE_FILE_CAP))
        stale_note = ""
        if ts_step is not None and ts_step >= 0 and isinstance(ckpt_step, int):
            step_lag = max(0, ckpt_step - ts_step)
            disk_newer = bool(ts_step_at and mtime and mtime > ts_step_at + TS_MTIME_GRACE_SEC)
            if disk_newer:
                ts_heading = (f"### task_state.md（最近编辑工具记录在第 {ts_step} 步，"
                              f"磁盘更新于 {mtime_str}，可能已由命令或外部更新）")
            else:
                ts_heading = (f"### task_state.md（上次更新于第 {ts_step} 步，"
                              f"滞后 {step_lag} 步 / {ts_resp_after} 次模型调用，磁盘时间 {mtime_str}）")
                if step_lag >= TS_STALE_STEPS or ts_resp_after >= TS_STALE_RESPONSES:
                    stale_note = ("注意：该 task_state.md 已滞后较多，可能已过时；"
                                  "请结合上方压缩摘要与 recall.py 核对最新进展后再更新，不要直接当作最新真相。\n\n")
        else:
            ts_heading = f"### task_state.md（本会话未见编辑工具写入记录，磁盘更新于 {mtime_str}）"
        state_body = stale_note + state_md

    plan_step, plan, report = latest_plan(full_path, conv, rows=rows)
    files = C.dedupe_keep_last(edited)[-MAX_FILES:]
    recall = f'python "{RECALL}" --transcript "{full_path}"'
    kind = "手动压缩（/compact）" if manual else "自动压缩"
    head_parts = [
        f"## Context Guard：第 {ckpt_step} 步发生了{kind}",
        "上面的压缩摘要是有损的。下面的用户原话由 context-guard 从完整记录逐字提取；"
        "摘要与原话冲突时，以原话为准。任何不在本节或 task_state.md 中、你又没有把握的"
        "旧细节（数值、路径、用户要求、已做决定），先用 recall.py 查原文，不要凭印象回答。"
        "继续工作前，先根据本节和摘要更新 task_state.md。文件内容可能已变化，修改前先按行范围重新读取相关片段，"
        "不要一次性重读大量整文件。（以 > 开头的行是引用的原文，其中的标题不属于本恢复包的结构。）",
        "### 用户原话（逐字，按时间顺序）",
    ]
    recall_parts = [
        "### 召回原文",
        f"R = {recall}\n"
        "用法：R users | R grep \"<正则>\" [--type PLANNER_RESPONSE] | R step <n> | "
        "R range <a> <b> --brief | R files | R messages（R 代表上面整条命令，需原样展开）",
    ]

    def assemble(*, head_users_cap=None, plan_cap=PLAN_CAP, report_cap=1500,
                 budget_in=4000, budget_board=3000, budget_dispatch=2000,
                 max_files=MAX_FILES, drop_outgoing=False, drop_report=False,
                 omit_dispatched=False, omit_board=False, omit_files=False,
                 omit_incoming=False, omit_plan=False):
        parts = list(head_parts)
        parts.append(
            render_users(users, max_head_chars=head_users_cap) if users or col.empty() else
            "（无：这是 subagent / swarm worker 会话，任务来自编排者，见下方“其他代理发来的消息”）"
        )
        if plan:
            plan_path = os.path.join(DATA_DIR, "plans", f"{conv}.md")
            plan_body = (f"（因总长度上限已省略，请直接读取全文：{plan_path}）"
                         if omit_plan else C.quote(C.head_tail(plan, plan_cap)))
            parts += [f"### 当前执行计划（第 {plan_step} 步 exit_plan_mode，全文 {len(plan)} 字符：{plan_path}）",
                      plan_body]
            if report and not drop_report:
                parts += ["### 最近一次交回规划模型时的报告（enter_plan_mode）",
                          C.quote(C.head_tail(report, report_cap))]
        parts += _build_agent_parts(
            col, bool(users), budget_in=budget_in, budget_board=budget_board,
            budget_dispatch=budget_dispatch, omit_incoming=omit_incoming,
            omit_dispatched=omit_dispatched, omit_board=omit_board,
            drop_outgoing=drop_outgoing,
        )
        files_body = ("（因总长度上限已省略，用 recall.py files 查看）"
                      if omit_files else ("\n".join(files[-max_files:]) or "（无）"))
        parts += [
            ts_heading,
            state_body,
            f"### 本会话修改过的文件（最近 {MAX_FILES} 个）",
            files_body,
            *recall_parts,
        ]
        return "\n\n".join(parts)

    opts = {}
    pack = assemble(**opts)
    if len(pack) > PACK_MAX_CHARS:
        trim_steps = [
            {"drop_outgoing": True},
            {"omit_dispatched": True},
            {"budget_board": 1200},
            {"omit_board": True},
            {"max_files": 5},
            {"omit_files": True},
            {"report_cap": 400},
            {"drop_report": True},
            {"budget_in": 1500 if users else 2500},
            {"omit_incoming": True} if users else {"budget_in": 1500},
            {"plan_cap": 1000},
            {"omit_plan": True},
            {"head_users_cap": 2000},
            {"head_users_cap": 600},
            {"head_users_cap": 0},
        ]
        for patch in trim_steps:
            opts.update(patch)
            pack = assemble(**opts)
            if len(pack) <= PACK_MAX_CHARS:
                break
    return pack


def compute_level(st):
    L = st.get("levels") or LEVELS_INFO
    if st.get("tokens"):
        return max([lv for th, lv in L["token"] if st["tokens"] >= th] or [0])
    return max([lv for th, lv in L["char"] if st.get("chars", 0) >= th] or [0])


def warning_text(level, tokens, chars, artifact_dir, levels=None, usage=None, conv="", full=""):
    L = levels or LEVELS_INFO
    lo, hi = L["range"]
    who = f"；按 {L.get('model')} 的{SOURCE_LABEL.get(L.get('source'), L.get('source'))}阈值估算" if L.get("model") else ""
    if tokens:
        size = f"当前上下文约 {tokens // 1000}k token（自动压缩约在 {lo // 1000}k–{hi // 1000}k 触发{who}）"
    else:
        size = f"距上次压缩已累积约 {chars // 1000}k 字符的记录"
    p = os.path.join(artifact_dir or "<artifact目录>", "task_state.md")
    top = U.top_line(usage)
    stat_arg = (f'"{conv}"' if " " in conv else conv) if conv else (f'--transcript "{full}"' if full else "<会话ID>")
    src = (f"\n{top}（完整分类：python \"{STATS}\" {stat_arg}）" if top else "")
    sections = "目标 / 用户原话约束 / 已做决定及原因 / 已完成 / 待办 / 关键路径、命令与数值；保持精简，≤ 6KB"
    if level == 1:
        return (f"## Context Guard 预警：{size}。\n"
                f"请在当前步骤完成后更新 {p}（{sections}），"
                "如果用户在线，在下次回复时提示：可以在自然断点手动执行 /compact。" + src)
    return (f"## Context Guard 紧急：{size}，即将自动压缩。\n"
            f"立即更新 {p}（{sections}），确保所有未完成事项、用户约束和关键数值都已写入，再继续其他工作。" + src)


class FileLock:
    """Cross-process lock: PreToolUse may run for many parallel tool calls."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        deadline = time.time() + 5
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > 15:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError("state lock busy")
                time.sleep(0.03)

    def __exit__(self, *a):
        os.close(self.fd)
        try:
            os.remove(self.path)
        except OSError:
            pass


class StateLock(FileLock):
    def __init__(self, conv):
        super().__init__(state_path(conv) + ".lock")


SOURCE_LABEL = {"observed": "实测", "builtin": "内置", "db_limit": "数据库上限", "default": "默认",
                "override": "手动指定"}
for _k in list(SOURCE_LABEL):
    SOURCE_LABEL[_k + "+seen"] = SOURCE_LABEL[_k] + "（已按实际更大上下文上调）"


def sync_calibration(st, events, conv, full):
    """Feed calibration with this batch's prompts / auto compactions; return levels for st['model']."""
    model = st.get("model")
    cal = K.load(DATA_DIR)
    dirty = bool(events) or K.needs_db(cal, model)
    if not dirty:
        return K.levels_for(cal, model)
    with FileLock(K.path(DATA_DIR) + ".lock"):
        cal = K.load(DATA_DIR)  # re-read under the lock
        changed = False
        for ev in events:
            if ev[0] == "auto":
                if K.record_auto(cal, ev[1], ev[2], conv, ev[3]):
                    changed = True
                    log(f"{conv} calibration: auto compaction on {ev[1]} at {ev[2]} tokens (step {ev[3]})")
            elif K.note_prompt(cal, ev[1], ev[2]):
                changed = True
        if K.needs_db(cal, model):
            limits = K.read_db_limits(K.db_path_for(full, conv))
            K.apply_db(cal, model, limits)
            changed = True
            log(f"{conv} calibration: db contextLimit for {model} = {limits.get(model)}")
        if changed:
            K.save(DATA_DIR, cal)
    return K.levels_for(cal, model)


def _empty_usage_stats():
    return {
        "steps": 0,
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "cache_hit_rate": 0.0,
        "cache_misses": 0,
    }


def _accum_usage(bucket, u):
    bucket["steps"] = int(bucket.get("steps") or 0) + 1
    bucket["input_tokens"] = int(bucket.get("input_tokens") or 0) + u["input"]
    bucket["cache_read_tokens"] = int(bucket.get("cache_read_tokens") or 0) + u["cache_read"]
    bucket["prompt_tokens"] = int(bucket.get("prompt_tokens") or 0) + u["prompt"]
    bucket["output_tokens"] = int(bucket.get("output_tokens") or 0) + u["output"]
    if u["prompt"] > 0 and u["cache_read"] == 0:
        bucket["cache_misses"] = int(bucket.get("cache_misses") or 0) + 1
    p = bucket["prompt_tokens"]
    bucket["cache_hit_rate"] = round(bucket["cache_read_tokens"] / p, 4) if p > 0 else 0.0


def update_state(st, objs, first_run):
    """Advance counters; mark a new checkpoint as pending (needs a recovery pack).

    Returns calibration events in step order: ("prompt", model, tokens, None) for prompts
    that did not immediately trigger an automatic compaction, and ("auto", model, tokens, step)
    for automatic compactions.
    """
    st.setdefault("cycle_stats", _empty_usage_stats())
    st.setdefault("total_stats", _empty_usage_stats())
    events, pending_prompt = [], None
    for o in objs:
        typ = o.get("type")
        if typ == "USER_INPUT":
            st["last_user_compact"] = C.user_text(o).strip().startswith("/compact")
        if typ == "CHECKPOINT":
            ck = o.get("step_index")
            manual = bool(st.get("last_user_compact"))
            if not manual:
                if st.get("model") and st.get("tokens"):
                    events.append(("auto", st["model"], st["tokens"], ck))
                pending_prompt = None
            elif pending_prompt:
                events.append(("prompt", pending_prompt[0], pending_prompt[1], None))
                pending_prompt = None
            compactions = int(st.get("compaction_count") or 0) + 1
            st.pop("verify_retry", None)
            st.pop("rewind_recheck", None)
            st.update(
                last_ckpt=ck,
                chars=len(o.get("content") or ""),
                tokens=None,
                input_tokens=None,
                cache_read_tokens=None,
                output_tokens=None,
                cache_hit_rate=None,
                cycle_stats=_empty_usage_stats(),
                warned=0,
                model_warned=0,
                warned_target=None,
                compaction_count=compactions,
                last_event="compacted",
                ckpt_manual=manual,
                # one /compact accounts for one checkpoint; a later one without a new
                # /compact is automatic
                last_user_compact=False,
                resp_since_ckpt=0,
            )
            if not first_run:
                st.update(pending=ck, pretool_sent=None, deny_seen=None)
            continue
        if typ == "GENERIC" and st.get("deny_seen") is False and DENY_MARKER in (o.get("content") or ""):
            st["deny_seen"] = True
        if typ == "PLANNER_RESPONSE":
            st["resp_since_ckpt"] = int(st.get("resp_since_ckpt") or 0) + 1
        st["chars"] += C.step_chars(o)
        u = C.step_usage(o)
        if u:
            if u["prompt"] > 0:
                if pending_prompt:
                    events.append(("prompt", pending_prompt[0], pending_prompt[1], None))
                st["tokens"] = u["prompt"]
                st["input_tokens"] = u["input"]
                st["cache_read_tokens"] = u["cache_read"]
                st["cache_hit_rate"] = u["hit_rate"]
                mdl = o.get("model") if isinstance(o.get("model"), str) and o.get("model") else None
                st["model"] = mdl
                pending_prompt = (mdl, u["prompt"]) if mdl else None
            if u["output"] > 0:
                st["output_tokens"] = u["output"]
            _accum_usage(st["cycle_stats"], u)
            _accum_usage(st["total_stats"], u)
    if pending_prompt:
        events.append(("prompt", pending_prompt[0], pending_prompt[1], None))
    return events


MAX_DELIVERY_TRIES = 2


def _has_rewind(objs, prev_max_step):
    max_step = prev_max_step if isinstance(prev_max_step, int) and prev_max_step >= 0 else None
    for o in objs:
        if not isinstance(o, dict):
            continue
        s = C._sidx(o)
        if s < 0:
            continue
        if o.get("type") == "USER_INPUT" and max_step is not None and s <= max_step:
            return True
        if max_step is None or s > max_step:
            max_step = s
    return False


def update_usage(st, objs, full, prev_max_step=None):
    """Context usage by category / tool / skill (see usage.py). Backfills once from the whole
    transcript when the state predates this feature or a rewind discards an earlier branch."""
    u = st.get("usage")
    if not isinstance(u, dict) or u.get("v") != 1 or _has_rewind(objs, prev_max_step):
        u = U.new_usage()
        limit = int(st.get("offset") or 0)
        objs = C.iter_active_steps(full, max_bytes=limit, transform=U.compact_step) if limit > 0 else []
    for o in objs:
        U.feed(u, o)
    st["usage"] = U.compact_for_state(u)


def check_delivery(st, objs, conv, full=""):
    """Confirm an injected recovery pack really reached the transcript.

    The pack shows up as a SYSTEM_MESSAGE before the next PLANNER_RESPONSE. If a model
    response appears without it, the injection was dropped: re-arm `pending` so the next
    PreToolUse deny / PreInvocation delivers it again (bounded by MAX_DELIVERY_TRIES).
    """
    v = st.get("verify")
    if not v:
        return
    for o in objs:
        typ = o.get("type")
        if typ == "CHECKPOINT":
            st.pop("verify", None)
            st.pop("verify_retry", None)
            return
        if typ == "SYSTEM_MESSAGE" and v["marker"] in (o.get("content") or ""):
            st.pop("verify", None)
            st.pop("verify_retry", None)
            st["last_delivery"] = "ok"
            log(f"{conv} delivery confirmed for checkpoint {v['ckpt']} (try {v['tries']})")
            return
        if typ == "PLANNER_RESPONSE":
            st.pop("verify", None)
            st["last_delivery"] = "failed"
            if int(v.get("tries") or 0) < MAX_DELIVERY_TRIES and not st.get("pending"):
                st.update(pending=v["ckpt"], pretool_sent=None, resp_since_ckpt=0, verify_retry=v["tries"])
                log(f"{conv} DELIVERY FAILED for checkpoint {v['ckpt']} (try {v['tries']}); re-armed")
            else:
                st.pop("verify_retry", None)
                log(f"{conv} DELIVERY FAILED for checkpoint {v['ckpt']} (try {v['tries']}); giving up")
                target = f'--transcript "{full}"' if full else (f'--conv "{conv}"' if conv else "--transcript <本会话 transcript_full.jsonl>")
                st["notice"] = (f"## Context Guard：第 {v['ckpt']} 步压缩后的恢复包多次注入都没有送达。"
                                f"请立即运行 python \"{RECALL}\" {target} users "
                                "查看用户原话，核对压缩后的工作是否偏离，并更新 task_state.md。")
            return


def is_stale(st):
    """Pending checkpoint seen long after the fact (hook was down / catching up)."""
    return int(st.get("resp_since_ckpt") or 0) > STALE_RESPONSES


# Audit 2026-11-26: after a recovery pack the agent often kept working for 30-100 model
# calls before touching task_state.md, so a second compaction would recover from a stale
# file. One reminder if it is still older than the pack after this many model calls.
NUDGE_AFTER_RESPONSES = 8


def mark_pack(st, ckpt):
    st["pack_at"] = round(time.time(), 3)
    st["pack_ckpt"] = ckpt
    st["pack_nudged"] = False


def task_state_nudge(conv, artifact_dir, st):
    if not st.get("pack_at") or st.get("pack_nudged") or st.get("pending"):
        return None
    if int(st.get("resp_since_ckpt") or 0) < NUDGE_AFTER_RESPONSES:
        return None
    st["pack_nudged"] = True
    p = os.path.join(artifact_dir, "task_state.md") if artifact_dir else ""
    try:
        if p and os.path.getmtime(p) >= float(st["pack_at"]):
            return None
    except OSError:
        pass
    log(f"{conv} task_state nudge after checkpoint {st.get('pack_ckpt')} "
        f"({st.get('resp_since_ckpt')} model calls since compaction)")
    return (f"## Context Guard 提醒：第 {st.get('pack_ckpt')} 步压缩后已经过 {st.get('resp_since_ckpt')} 次模型调用，"
            f"{p or 'task_state.md'} 还没有更新（或不存在）。\n"
            "请在当前步骤完成后，把恢复包里的用户原话约束、压缩后已完成的进度、新决定和下一步精确命令写进去；"
            "否则下次压缩只能从过期的工作记忆恢复。保持精简（≤ 6KB）。")


def on_pre_tool(payload, conv, full, artifact_dir, st, first_run):
    tool = (payload.get("toolCall") or {}).get("name") or ""
    if not st.get("pending") or tool not in MUTATING_TOOLS or is_stale(st):
        return NEUTRAL
    manual = bool(st.get("ckpt_manual"))
    if st.get("pretool_sent") is None:
        pack = build_recovery(full, artifact_dir, st["pending"], conv, manual)
        step_idx = payload.get("stepIdx")
        if step_idx is None:
            tc_idx = (payload.get("toolCall") or {}).get("stepIdx")
            step_idx = tc_idx if tc_idx is not None else True
        st["pretool_sent"] = step_idx
        st["deny_seen"] = False  # flipped when the deny result shows up in the transcript
        mark_pack(st, st["pending"])
        st["last_event"] = "pretool_deny"
        log(f"{conv} recovery delivered via PreToolUse deny of {tool} (checkpoint {st['pending']}, manual={manual})")
        kind = "手动压缩（/compact）" if manual else "自动压缩"
        reason = (f"{DENY_MARKER}了{kind}，你这一轮是在只有有损摘要的情况下做的决定。"
                  "本次修改类操作已被拦截、没有执行。请先阅读下面的恢复包，核对后再重新发起需要的操作。\n\n" + pack)
    else:
        short_kind = "手动压缩 (/compact)" if manual else "自动压缩"
        reason = (f"【Context Guard 拦截】{short_kind}后的恢复包已在本轮前面被拦截的工具结果中给出，"
                  "本次修改类操作同样未执行。阅读恢复包后再重新发起。")
    return {"decision": "deny", "reason": reason}


def on_pre_invocation(conv, full, artifact_dir, st, first_run, objs):
    inject = None
    if st.get("pending") and not first_run:
        ckpt_step = st["pending"]
        if st.get("pretool_sent") is not None and st.get("deny_seen") is not False:
            inject = (f"## Context Guard：第 {ckpt_step} 步发生了压缩，恢复包已在上一轮被拦截的工具结果中给出。"
                      "被拦截的修改都没有执行；请按恢复包核对后再重新发起，并先更新 task_state.md。")
            st.pop("verify_retry", None)
            st["last_delivery"] = "ok"
            log(f"{conv} delivery confirmed via PreToolUse deny for checkpoint {ckpt_step}")
        else:
            manual = bool(st.get("ckpt_manual"))
            inject = build_recovery(full, artifact_dir, ckpt_step, conv, manual)
            if st.get("pretool_sent") is not None:
                inject = ("（Context Guard 重发：上一轮拦截工具时附带的恢复包没有出现在工具结果里，这里完整重发一次。"
                          "被拦截的修改都没有执行。）\n\n" + inject)
                log(f"{conv} deny result not seen in transcript for checkpoint {ckpt_step}; full pack re-sent")
            elif is_stale(st):
                inject = (f"（Context Guard 补发：第 {ckpt_step} 步的压缩发生时本插件未运行，"
                          f"压缩后你已经做了 {st.get('resp_since_ckpt')} 次模型调用。"
                          "请对照下面的用户原话检查之前的工作是否偏离。）\n\n" + inject)
            log(f"{conv} recovery injected for checkpoint {ckpt_step} (manual={manual}, stale={is_stale(st)})")
            tries = int(st.pop("verify_retry", None) or (st.get("verify") or {}).get("tries") or 0) + 1
            mark_pack(st, ckpt_step)
            st["verify"] = {"ckpt": ckpt_step, "marker": f"Context Guard：第 {ckpt_step} 步发生了", "tries": tries}
        st["last_event"] = "compacted"
        st.update(pending=None, pretool_sent=None, deny_seen=None)
    else:
        level = compute_level(st)
        if level > st.get("model_warned", 0):
            L = st.get("levels") or LEVELS_INFO
            st["model_warned"] = level
            st["warned_target"] = int(L.get("target") or COMPACTION_TARGET_TOKENS)
            if first_run:
                st["warned"] = max(st.get("warned", 0), level)
            else:
                inject = warning_text(level, st["tokens"], st["chars"], artifact_dir, st.get("levels"), st.get("usage"),
                                      conv=conv, full=full)
                st["last_event"] = f"warn_{level}"
                st["warned"] = max(st.get("warned", 0), level)
                log(f"{conv} warning level {level} tokens={st['tokens']} chars={st['chars']} "
                    f"model={L.get('model')} target={L.get('target')} source={L.get('source')}")

    is_worker = not st.get("rule_done") and any(
        o.get("type") == "SYSTEM_MESSAGE" and SWARM_WORKER_MARK in (o.get("content") or "")
        and int(o.get("step_index") or 0) <= 5 for o in objs)
    if is_worker and not inject:
        # Swarm workers do not load plugin rules/skills: deliver the rule here once.
        with open(RULE_FILE, encoding="utf-8") as f:
            rule = f.read()
        inject = (rule + f"\n\n（你的 task_state.md 独占放在 {artifact_dir}，严禁改写 Coordinator 或其他会话的 task_state.md，"
                  f"结果请通过消息板回报；recall.py 路径：{RECALL}；"
                  "swarm 消息板的完整历史可用 read_board{since:0} 查看。）")
        st["rule_done"] = True
        log(f"{conv} rule injected for swarm worker")
    if not inject:
        inject = task_state_nudge(conv, artifact_dir, st)
    if not inject and st.get("notice"):
        # One-shot message queued by check_delivery (or by hand) for this conversation.
        inject = st.pop("notice")
        log(f"{conv} notice injected")
    if not inject:
        return {}
    # Verified 2026-11-15 (Antigravity 2.32.1): if ANY injected step carries
    # metadata.user_markdown, the whole injectSteps batch is silently dropped (the model
    # sees nothing, no SYSTEM_MESSAGE is recorded). Never attach metadata to model text.
    return {"injectSteps": [{"systemMessage": {"systemMessage": inject}}]}


def main(payload):
    event = payload.get("hookEventName") or "PreInvocation"
    is_tool = event == "PreToolUse"
    if event == "PostInvocation":  # no longer registered; stay inert if it is
        return {}
    conv = payload.get("conversationId") or ""
    if not is_valid_conv_id(conv):
        return NEUTRAL if is_tool else {}
    artifact_dir = payload.get("artifactDirectoryPath") or ""
    full = C.resolve_full_transcript(payload.get("transcriptPath"), conv)
    if not full:
        return NEUTRAL if is_tool else {}

    with StateLock(conv):
        st = load_state(conv)
        first_run = st is None
        if first_run:
            st = {
                "offset": 0, "last_ckpt": None, "chars": 0, "tokens": None,
                "input_tokens": None, "cache_read_tokens": None, "output_tokens": None, "cache_hit_rate": None,
                "cycle_stats": _empty_usage_stats(), "total_stats": _empty_usage_stats(),
                "warned": 0, "model_warned": 0, "warned_target": None, "compaction_count": 0, "last_event": "normal",
            }
        else:
            st.setdefault("model_warned", st.get("warned", 0))
            st.setdefault("compaction_count", 0)
            st.setdefault("last_event", "normal")
            st.setdefault("cycle_stats", _empty_usage_stats())
            st.setdefault("total_stats", _empty_usage_stats())
        objs, st["offset"] = read_new(full, st["offset"])
        # Workers' rule check needs the first steps even if the first run was a PreToolUse.
        st.setdefault("seen_head", [])
        if not st.get("rule_done") and not st.get("worker_check_done") and objs:
            st["seen_head"] = [o for o in (st["seen_head"] + objs)
                               if int(o.get("step_index") or 0) <= 5 and o.get("type") == "SYSTEM_MESSAGE"][:6]
        prev_max_step = st.get("max_step")
        prev_target = int((st.get("levels") or {}).get("target") or 0)
        rewound = _has_rewind(objs, prev_max_step)
        if rewound:
            st["rewind_recheck"] = True
        has_new_prompt = any((C.step_usage(o) or {}).get("prompt", 0) > 0 for o in objs)
        for o in objs:
            s = C._sidx(o)
            if s >= 0:
                cur = st.get("max_step")
                st["max_step"] = s if not isinstance(cur, int) or s > cur else cur
        events = update_state(st, objs, first_run)
        try:
            update_usage(st, objs, full, prev_max_step=prev_max_step)
        except Exception:
            log(f"{conv} usage ERROR " + traceback.format_exc().replace("\n", " | "))
        check_delivery(st, objs, conv, full)
        try:
            st["levels"] = sync_calibration(st, events, conv, full)
        except Exception:
            log(f"{conv} calibration ERROR " + traceback.format_exc().replace("\n", " | "))
            st["levels"] = st.get("levels") or LEVELS_INFO
        # Conditional monotonic warning lock: only reset/downgrade when compaction resets
        # (handled in update_state), the branch rewound, or the target window grew >10%.
        lv_now = compute_level(st)
        new_target = int((st.get("levels") or LEVELS_INFO).get("target") or 0)
        cur_warned = int(st.get("model_warned") or 0)
        base_target = int(st.get("warned_target") or prev_target or 0)
        target_rose = bool(base_target > 0 and new_target > base_target * 1.1)
        rewind_ready = bool(st.get("rewind_recheck") and has_new_prompt)
        if rewind_ready:
            st.pop("rewind_recheck", None)
        if lv_now < cur_warned:
            if target_rose or rewind_ready:
                st["model_warned"] = lv_now
                st["warned"] = min(int(st.get("warned") or 0), lv_now)
                st["warned_target"] = new_target if lv_now > 0 else None
            elif st.get("warned_target") is None and base_target > 0:
                st["warned_target"] = base_target
        if is_tool:
            out = on_pre_tool(payload, conv, full, artifact_dir, st, first_run)
        else:
            out = on_pre_invocation(conv, full, artifact_dir, st, first_run, st.get("seen_head") or objs)
            head = st.get("seen_head") or []
            if st.get("rule_done"):
                st.pop("seen_head", None)
            elif int(st.get("max_step") or 0) > 5 and not any(
                    SWARM_WORKER_MARK in (o.get("content") or "") for o in head):
                # Past the first steps without a swarm assignment: not a worker, the
                # cached head steps are no longer needed.
                st["worker_check_done"] = True
                st.pop("seen_head", None)
        st["updated_at"] = round(time.time(), 3)
        if artifact_dir:
            st["artifact_dir"] = artifact_dir
        st["task_state_exists"] = bool(artifact_dir and os.path.isfile(os.path.join(artifact_dir, "task_state.md")))
        save_state(conv, st)
    return out


def rebuild_calibration(brain=None):
    """CLI: rebuild calibration.json samples/max_prompt from all transcripts (keeps overrides/db info)."""
    import glob
    brain = brain or os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "brain")
    events = []
    for p in glob.glob(os.path.join(brain, "*", ".system_generated", "logs", "transcript_full.jsonl")):
        parts = os.path.normpath(os.path.abspath(p)).split(os.sep)
        conv = parts[-4] if len(parts) >= 4 else ""
        pending_prompt = None
        compact = False
        for o in C.iter_jsonl(p):
            typ, at = o.get("type"), o.get("created_at") or ""
            if typ == "USER_INPUT":
                compact = C.user_text(o).strip().startswith("/compact")
            elif typ == "CHECKPOINT":
                if pending_prompt:
                    p_at, p_model, p_tokens = pending_prompt
                    if not compact and p_model and p_tokens:
                        events.append((at, "auto", p_model, p_tokens, conv, o.get("step_index")))
                    elif compact and p_model and p_tokens:
                        events.append((p_at, "prompt", p_model, p_tokens, conv, None))
                    pending_prompt = None
                compact = False
            elif typ == "PLANNER_RESPONSE":
                u = C.step_usage(o)
                if u and u["prompt"] > 0:
                    if pending_prompt and pending_prompt[1]:
                        events.append((pending_prompt[0], "prompt", pending_prompt[1], pending_prompt[2], conv, None))
                    mdl = o.get("model") if isinstance(o.get("model"), str) and o.get("model") else None
                    pending_prompt = (at, mdl, u["prompt"])
        if pending_prompt and pending_prompt[1]:
            events.append((pending_prompt[0], "prompt", pending_prompt[1], pending_prompt[2], conv, None))
    events.sort(key=lambda e: e[0])
    with FileLock(K.path(DATA_DIR) + ".lock"):
        cal = K.load(DATA_DIR)
        for e in (cal.get("models") or {}).values():
            for k in ("samples", "stale_samples", "max_prompt"):
                e.pop(k, None)
        for _, kind, model, tok, conv, step in events:
            if kind == "auto":
                K.record_auto(cal, model, tok, conv, step)
            else:
                K.note_prompt(cal, model, tok)
        K.save(DATA_DIR, cal)
    for model in sorted(cal.get("models") or {}):
        L = K.levels_for(cal, model)
        e = cal["models"][model]
        print(f"{model}: target={L['target']} source={L['source']} warn={[t for t, _ in L['token']]} "
              f"samples={[s['tokens'] for s in e.get('samples') or []]} max_prompt={e.get('max_prompt')} "
              f"db_limit={e.get('db_limit')}")
    return 0


if __name__ == "__main__":
    if "--calibrate" in sys.argv[1:]:
        sys.exit(rebuild_calibration())
    raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    is_tool = '"PreToolUse"' in raw
    try:
        out = main(json.loads(raw or "{}"))
    except Exception:
        log("ERROR " + traceback.format_exc().replace("\n", " | "))
        out = NEUTRAL if is_tool else {}
    sys.stdout.write(json.dumps(out, ensure_ascii=True))
    sys.stdout.flush()
