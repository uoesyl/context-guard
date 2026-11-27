"""Shared helpers for context-guard (hook + recall CLI)."""
import json
import os
import re
import urllib.parse

USER_REQ_RE = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.S)
MSG_RE = re.compile(
    r"\[Message\] timestamp=(\S+) sender=(\S+) priority=\S+ content=(.*?)"
    r"(?=\n\[Message\] timestamp=|\n</SYSTEM_MESSAGE>|\Z)", re.S)
CONV_ID_RE = re.compile(r'"conversation_?[iI]d"\s*:\s*"([0-9a-f-]{36})"')
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
NOISE_PREFIXES = ("[swarm] new board message", "[Notice]")
EDIT_TOOLS = {"write_file", "edit_file", "replace_file_content", "write_to_file",
              "multi_replace_file_content"}
BOARD_FILE_MAX_BYTES = 4 * 1024 * 1024
STEP_OUTPUT_MAX_BYTES = 2 * 1024 * 1024
TRUNC_OUTPUT_URI_RE = re.compile(r"file://(/[^\s\)\"'\r\n]+?output\.txt)", re.I)


def conv_dir_for_transcript(path):
    """Return <conv> directory if `path` is under <conv>/.system_generated/logs/, else None."""
    if not path:
        return None
    norm = os.path.abspath(os.path.normpath(path))
    logs_dir = os.path.dirname(norm)
    sysgen_dir = os.path.dirname(logs_dir)
    conv_dir = os.path.dirname(sysgen_dir)
    if (os.path.basename(logs_dir).lower() == "logs"
            and os.path.basename(sysgen_dir).lower() == ".system_generated"
            and conv_dir and conv_dir != os.path.dirname(conv_dir)):
        return conv_dir
    return None


def _file_uri_to_path(raw_path):
    p = urllib.parse.unquote(raw_path)
    if re.match(r"^/[A-Za-z]:/", p):
        p = p[1:]
    return os.path.normpath(p)


def _is_within_dir(candidate, root_dir):
    try:
        root_real = os.path.normcase(os.path.realpath(root_dir))
        cand_real = os.path.normcase(os.path.realpath(candidate))
        return os.path.commonpath([root_real, cand_real]) == root_real
    except (OSError, ValueError):
        return False


def resolve_full_transcript(transcript_path, conversation_id=None):
    """Return path of transcript_full.jsonl (falls back to the given transcript)."""
    cands = []
    if transcript_path:
        d = os.path.dirname(transcript_path)
        cands += [os.path.join(d, "transcript_full.jsonl"), transcript_path]
    if conversation_id:
        base = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "brain",
                            conversation_id, ".system_generated", "logs")
        cands += [os.path.join(base, "transcript_full.jsonl"),
                  os.path.join(base, "transcript.jsonl")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def iter_jsonl(path, max_bytes=None):
    with open(path, "rb") as f:
        read = 0
        for raw in f:
            if max_bytes is not None:
                if read + len(raw) > max_bytes:
                    break
                read += len(raw)
            try:
                yield json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue


def _agent_slots(o):
    slots = []
    for tc in o.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name")
        a = tc.get("args") if isinstance(tc.get("args"), dict) else {}
        slots.append(name in ("read_board", "swarm_wait", "invoke_subagent")
                     or (name == "swarm" and a.get("action") == "add"))
    return slots


def _slim_recovery_step(o, keep_generic_content=False):
    typ = o.get("type")
    step = o.get("step_index")
    created_at = o.get("created_at")
    if typ == "USER_INPUT":
        out = {"type": typ, "step_index": step, "content": o.get("content") or ""}
        if o.get("media"):
            out["media"] = o["media"]
    elif typ == "SYSTEM_MESSAGE":
        out = {"type": typ, "step_index": step}
        c = o.get("content") or ""
        if "[Message] timestamp=" in c:
            out["content"] = c
    elif typ == "PLANNER_RESPONSE":
        tcs = []
        for tc in o.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            a = tc.get("args") if isinstance(tc.get("args"), dict) else {}
            if name in ("invoke_subagent", "send_message", "post_message") or (name == "swarm" and a.get("action") == "add"):
                tcs.append(tc)
            elif name == "exit_plan_mode":
                tcs.append({"name": name, "args": {"plan": a["plan"]}} if a.get("plan") else {"name": name})
            elif name == "enter_plan_mode":
                tcs.append({"name": name, "args": {"reason": a["reason"]}} if a.get("reason") else {"name": name})
            elif name in EDIT_TOOLS:
                tcs.append({"name": name, "args": {k: a[k] for k in ("Path", "TargetFile", "path") if a.get(k)}})
            else:
                tcs.append({"name": name})
        out = {"type": typ, "step_index": step}
        if tcs:
            out["tool_calls"] = tcs
    elif typ == "GENERIC":
        if keep_generic_content:
            out = {"type": typ, "step_index": step, "content": o.get("content") or ""}
        else:
            out = {"type": typ, "step_index": step}
    else:
        out = {"type": typ, "step_index": step}
    if created_at:
        out["created_at"] = created_at
    return out


def iter_active_steps(path_or_rows, light=False, max_bytes=None, transform=None):
    """Return active branch steps, dropping superseded branches after user edit/retry."""
    rows = iter_jsonl(path_or_rows, max_bytes=max_bytes) if isinstance(path_or_rows, (str, os.PathLike)) else path_or_rows
    active = []
    max_step = None
    agent_queue = []
    for o in rows:
        if not isinstance(o, dict):
            continue
        s = _sidx(o)
        has_step = s >= 0
        if o.get("type") == "USER_INPUT" and has_step and max_step is not None and s <= max_step:
            kept = []
            in_discarded = False
            for prev in active:
                ps = _sidx(prev)
                if ps >= 0:
                    in_discarded = ps >= s
                if not in_discarded:
                    kept.append(prev)
            active = kept
            max_step = max((_sidx(x) for x in active if _sidx(x) >= 0), default=None)
            if light:
                agent_queue = []
                generics_after = 0
                for prev in reversed(active):
                    pt = prev.get("type")
                    if pt == "GENERIC":
                        generics_after += 1
                    elif pt == "PLANNER_RESPONSE":
                        agent_queue = _agent_slots(prev)[generics_after:]
                        break
        if light:
            typ = o.get("type")
            if typ == "PLANNER_RESPONSE":
                agent_queue = _agent_slots(o)
                o = _slim_recovery_step(o)
            elif typ == "GENERIC":
                keep = agent_queue.pop(0) if agent_queue else False
                o = _slim_recovery_step(o, keep_generic_content=keep)
            else:
                o = _slim_recovery_step(o)
        elif transform is not None:
            o = transform(o)
        active.append(o)
        if has_step and (max_step is None or s > max_step):
            max_step = s
    return active


def step_chars(o):
    n = len(o.get("content") or "") + len(o.get("thinking") or "")
    tc = o.get("tool_calls")
    if tc:
        n += len(json.dumps(tc, ensure_ascii=False, separators=(",", ":")))
    return n


def step_usage(o):
    """Return token & cache usage dict for a PLANNER_RESPONSE step, or None."""
    if o.get("type") != "PLANNER_RESPONSE":
        return None
    try:
        inp = max(0, int(o.get("input_tokens") or 0))
        cr = max(0, int(o.get("cache_read_tokens") or 0))
        out = max(0, int(o.get("output_tokens") or 0))
    except (TypeError, ValueError):
        return None
    prompt = inp + cr
    if prompt <= 0 and out <= 0:
        return None
    hit_rate = round(cr / prompt, 4) if prompt > 0 else 0.0
    return {
        "prompt": prompt,
        "input": inp,
        "cache_read": cr,
        "output": out,
        "hit_rate": hit_rate,
    }


def ctx_tokens(o):
    """Real prompt size of a PLANNER_RESPONSE step, or None."""
    u = step_usage(o)
    return (u["prompt"] or None) if u else None


def user_text(o):
    c = o.get("content") or ""
    m = USER_REQ_RE.search(c)
    text = m.group(1) if m else c
    media = [x.get("uri") for x in (o.get("media") or []) if isinstance(x, dict) and x.get("uri")]
    if media:
        text += "\n  [附件] " + " ; ".join(media)
    return text.strip()


def edited_paths(o):
    out = []
    for tc in o.get("tool_calls") or []:
        if not isinstance(tc, dict) or tc.get("name") not in EDIT_TOOLS:
            continue
        args = tc.get("args") or {}
        p = args.get("Path") or args.get("TargetFile") or args.get("path")
        if p:
            out.append(p)
    return out


def dedupe_keep_last(items):
    seen, out = set(), []
    for x in reversed(items):
        k = x.replace("\\", "/").lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return list(reversed(out))


def quote(text):
    """Render embedded content as a blockquote so its markdown headings cannot
    break the recovery pack's own structure."""
    return "\n".join("> " + ln if ln.strip() else ">" for ln in (text or "").splitlines())


def head_tail(text, limit):
    if len(text) <= limit:
        return text
    h = max(limit * 2 // 3, 50)
    t = max(limit - h - 40, 30)
    return f"{text[:h]}\n  …[中间省略 {len(text) - h - t} 字符，用 recall.py step 查看全文]…\n{text[-t:]}"


def _sidx(o):
    try:
        return int(o.get("step_index"))
    except (TypeError, ValueError):
        return -1


def _short(cid):
    return cid[:8] if UUID_RE.fullmatch(cid or "") else cid


class AgentCollector:
    """Collects multi-agent traffic from transcript steps.

    incoming  : messages from other agents (subagent task/briefs from the parent,
                subagent reports to an orchestrator, swarm worker assignment)
    outgoing  : send_message / post_message issued by this conversation
    dispatched: invoke_subagent / swarm add, with resulting conversation id
    board     : swarm board messages seen via read_board (dedup by seq)
    """

    def __init__(self, conv_dir=None):
        self.incoming, self.outgoing, self.dispatched = [], [], []
        self.board = {}
        self._queue = []
        self._conv_dir = os.path.realpath(conv_dir) if conv_dir and os.path.isdir(conv_dir) else None
        self._trunc_board_files = []

    def _parse_board_text(self, text, step):
        k = text.find('{"messages"')
        if k < 0 or text[:k].count("\n") > 8:  # tool output header, not embedded text
            return False
        try:
            data, _ = json.JSONDecoder().raw_decode(text[k:])
        except Exception:
            return False
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return False
        for msg in msgs:
            if isinstance(msg, dict) and "seq" in msg:
                seq = msg["seq"]
                prev = self.board.get(seq)
                prev_step = (prev["step"] if isinstance(prev, dict)
                             and isinstance(prev.get("step"), int) and prev["step"] >= 0 else -1)
                use_step = (step if isinstance(step, int) and step >= 0 and prev_step < 0
                            else (prev_step if prev_step >= 0 else step))
                self.board[seq] = {"step": use_step, "seq": seq,
                                   "agent": msg.get("agent") or "?",
                                   "text": msg.get("text") or ""}
        return True

    def load_board_files(self):
        """Read full board messages from swarm/board.jsonl (coordinator) or truncated step output files (worker)."""
        if not self._conv_dir:
            return
        bp = os.path.join(self._conv_dir, ".system_generated", "swarm", "board.jsonl")
        if os.path.isfile(bp) and _is_within_dir(bp, self._conv_dir):
            try:
                sz = os.path.getsize(bp)
                with open(bp, "rb") as f:
                    if sz > BOARD_FILE_MAX_BYTES:
                        f.seek(sz - BOARD_FILE_MAX_BYTES)
                        f.readline()  # discard partial first line
                    for raw in f:
                        try:
                            msg = json.loads(raw.decode("utf-8", errors="replace"))
                        except Exception:
                            continue
                        if isinstance(msg, dict) and "seq" in msg:
                            seq = msg["seq"]
                            prev = self.board.get(seq)
                            step = (prev["step"] if isinstance(prev, dict)
                                    and isinstance(prev.get("step"), int) and prev["step"] >= 0 else -1)
                            self.board[seq] = {"step": step, "seq": seq,
                                               "agent": msg.get("agent") or "?",
                                               "text": msg.get("text") or ""}
                return
            except OSError:
                pass
        steps_root = os.path.join(self._conv_dir, ".system_generated", "steps")
        if not os.path.isdir(steps_root):
            return
        seen = set()
        budget = BOARD_FILE_MAX_BYTES
        for step, fp in reversed(self._trunc_board_files):
            norm = os.path.normcase(os.path.normpath(fp))
            if norm in seen:
                continue
            seen.add(norm)
            if not os.path.isfile(fp) or not _is_within_dir(fp, steps_root):
                continue
            try:
                sz = os.path.getsize(fp)
                if sz <= 0 or sz > STEP_OUTPUT_MAX_BYTES or sz > budget:
                    continue
                with open(fp, encoding="utf-8", errors="replace") as f:
                    txt = f.read(STEP_OUTPUT_MAX_BYTES + 1)
                budget -= sz
                self._parse_board_text(txt, step)
            except OSError:
                continue

    def feed(self, o):
        t, s = o.get("type"), _sidx(o)
        if t in ("SYSTEM_MESSAGE", "USER_INPUT"):
            for m in MSG_RE.finditer(o.get("content") or ""):
                ts, sender, text = m.group(1), m.group(2), m.group(3).strip()
                if sender == "system" or "/task-" in sender or not UUID_RE.fullmatch(sender):
                    continue
                if text.startswith(NOISE_PREFIXES):
                    continue
                self.incoming.append({"step": s, "time": ts, "sender": sender, "text": text})
        elif t == "PLANNER_RESPONSE":
            self._queue = []
            for tc in o.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name")
                a = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                slot = None
                if name in ("read_board", "swarm_wait"):
                    slot = ("board", None)
                elif name == "invoke_subagent":
                    typ = a.get("TypeName") or a.get("agent_config_name") or a.get("type_name") or ""
                    label = a.get("Role") or a.get("role") or typ or "?"
                    text = a.get("Prompt") or a.get("prompt") or a.get("description") or ""
                    d = {"step": s, "kind": "subagent", "label": label,
                         "type": typ, "text": text, "conv": None}
                    self.dispatched.append(d)
                    slot = ("dispatch", d)
                elif name == "swarm" and a.get("action") == "add":
                    d = {"step": s, "kind": "swarm", "label": a.get("name") or "?", "type": "swarm worker",
                         "text": a.get("brief") or "", "conv": None}
                    self.dispatched.append(d)
                    slot = ("dispatch", d)
                elif name == "send_message":
                    to = a.get("Recipient") or a.get("recipient") or a.get("conversation_id") or a.get("conversationId") or "?"
                    text = a.get("Message") or a.get("message") or ""
                    self.outgoing.append({"step": s, "to": to, "text": text})
                elif name == "post_message":
                    self.outgoing.append({"step": s, "to": "board", "text": a.get("text") or ""})
                self._queue.append(slot)
        elif t == "GENERIC":
            slot = self._queue.pop(0) if self._queue else None
            if not slot:
                return
            kind, target = slot
            c = o.get("content") or ""
            if kind == "dispatch":
                m = CONV_ID_RE.search(c)
                if m:
                    target["conv"] = m.group(1)
            elif kind == "board":
                ok = self._parse_board_text(c, s)
                if not ok or "output.txt" in c[:1000]:
                    m = TRUNC_OUTPUT_URI_RE.search(c[:1000])
                    if m:
                        self._trunc_board_files.append((s, _file_uri_to_path(m.group(1))))

    def empty(self):
        return not (self.incoming or self.outgoing or self.dispatched or self.board)


def render_agent_sections(col, has_users=False, budget_in=4000, budget_board=3000, budget_dispatch=2000):
    """Render recovery sections for multi-agent traffic. Returns list of strings.

    has_users=False means this is a subagent / swarm worker: its earliest incoming
    message is the task given by its orchestrator and is kept (nearly) whole.
    """
    parts = []
    if col.incoming:
        lines, rest = [], col.incoming
        if not has_users:
            first = col.incoming[0]
            lines.append(f"[step {first['step']}] 来自 {_short(first['sender'])}（编排者下发的任务说明）：\n"
                         f"{quote(head_tail(first['text'], min(5000, budget_in)))}")
            rest = col.incoming[1:]
        remain = max(budget_in - sum(len(x) for x in lines), 1500)
        tail = rest[-3:]
        head = rest[:-3]
        if len(head) > 8:  # older messages: count only, recall.py messages has them all
            lines.append(f"（更早的 {len(head) - 8} 条消息已省略，用 recall.py messages 查看）")
            head = head[-8:]
        tail_txt = [head_tail(m["text"], 800) for m in tail]
        per = max((remain - sum(len(x) for x in tail_txt)) // max(len(head), 1), 120)
        for m in head:
            lines.append(f"[step {m['step']}] 来自 {_short(m['sender'])}：\n{quote(head_tail(m['text'], per))}")
        for m, txt in zip(tail, tail_txt):
            lines.append(f"[step {m['step']}] 来自 {_short(m['sender'])}：\n{quote(txt)}")
        parts += ["### 其他代理发来的消息（逐字；subagent 的任务说明 / subagent 回报 / swarm 任务）",
                  "\n\n".join(lines)]
    if col.dispatched:
        shown = col.dispatched[-12:]
        per = min(max(budget_dispatch // len(shown) - 120, 80), 200)

        def gist(t):  # first non-empty line only; full brief via recall.py
            line = next((x.strip() for x in t.splitlines() if x.strip()), "")
            return line if len(line) <= per else line[:per] + "…"
        lines = [f"- [step {d['step']}] {d['kind']} `{d['label']}` conv={d['conv'] or '未知'}："
                 f"\n{quote(gist(d['text']))}" for d in shown]
        if len(col.dispatched) > len(shown):
            lines.insert(0, f"（更早派发的 {len(col.dispatched) - len(shown)} 个已省略，用 recall.py messages 查看）")
        parts += ["### 你派发过的 subagent / swarm worker（可用 recall.py --conv <conv> 查其完整记录）",
                  "\n".join(lines)]
    if col.board or any(m["to"] == "board" for m in col.outgoing):
        msgs = sorted(col.board.values(), key=lambda m: m["seq"])[-15:]
        per = max(budget_board // max(len(msgs), 1), 150)
        lines = [f"#{m['seq']} {m['agent']}:\n{quote(head_tail(m['text'], per))}" for m in msgs]
        parts += ["### swarm 消息板（最近 15 条；完整历史用 read_board{since:0}）",
                  "\n".join(lines) or "（未读到消息板内容）"]
    sent = [m for m in col.outgoing if m["to"] != "board"][-6:]
    if sent:
        parts += ["### 你最近发给其他代理的消息",
                  "\n".join(f"- [step {m['step']}] → {_short(m['to'])}：\n{quote(head_tail(m['text'], 250))}"
                            for m in sent)]
    return parts
