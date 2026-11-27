"""Context usage accounting for context-guard (pure, incremental, JSON-serialisable).

Answers "what made the context grow" and "which tools / skills / MCP servers were used".

Token attribution: every PLANNER_RESPONSE reports its real prompt size (input + cache_read).
The growth between two consecutive model calls is split over the items that entered the
context in between (the previous call's own output, tool results, user/system messages),
proportionally to their character counts. The first prompt of a window (conversation start
or right after a compaction) minus its estimated items is booked as ``baseline`` (system
prompt, tool schemas, rules, skills list). Negative deltas (the app trimmed something without
a checkpoint) are booked as ``trimmed``. Character counts are exact; token figures are
estimates whose sum per window matches the real prompt size.
"""
import json
import re

CHARS_PER_TOKEN = 3.0  # only used to size the first-prompt items against the baseline
TOP_TOOLS_IN_STATE = 60
SKILL_RE = re.compile(r"[\\/]skills[\\/]([^\\/]+)[\\/](.+)$", re.I)
MSG_SENDER_RE = re.compile(r"sender=\S+?/(task|[0-9a-f]{8})", re.I)

CAT_LABELS = {
    "baseline": "基线（系统提示/工具定义/规则）",
    "user_input": "用户输入",
    "model:thinking": "模型思考",
    "model:text": "模型回复正文",
    "system:context_guard": "Hook 注入（Context Guard 等）",
    "system:task_notice": "后台任务/子代理通知",
    "system:other": "其他系统消息",
    "checkpoint": "压缩摘要",
    "error": "错误消息",
    "trimmed": "被裁剪（负增长）",
    "unattributed": "未归因",
}


def label(cat):
    if cat.startswith("tool:"):
        return f"工具结果 · {cat[5:]}"
    if cat.startswith("args:"):
        return f"工具调用参数 · {cat[5:]}"
    return CAT_LABELS.get(cat, cat)


def new_usage():
    return {
        "v": 1,
        "window": {"cats": {}, "first_prompt": None, "prompt": None, "since_step": None},
        "total": {"cats": {}},
        "tools": {},
        "skills": {},
        "mcp": {},
        "pending": [],
        "queue": [],
        "last_prompt": None,
        "compactions": 0,
    }


def _bump(cats, cat, chars=0, tokens=0, n=1):
    e = cats.setdefault(cat, {"chars": 0, "tokens": 0, "n": 0})
    e["chars"] += int(chars)
    e["tokens"] += int(round(tokens))
    e["n"] += n


def _book(u, cat, chars=0, tokens=0, n=1):
    _bump(u["window"]["cats"], cat, chars, tokens, n)
    _bump(u["total"]["cats"], cat, chars, tokens, n)


def _tool(u, name):
    return u["tools"].setdefault(name, {"calls": 0, "arg_chars": 0, "result_chars": 0,
                                        "result_tokens": 0, "arg_tokens": 0, "errors": 0})


def _system_cat(o):
    if o.get("source") == "SYSTEM_SDK" or "injected by a PreInvocation hook" in (o.get("content") or ""):
        return "system:context_guard"
    if MSG_SENDER_RE.search(o.get("content") or ""):
        return "system:task_notice"
    return "system:other"


def _summarize_tool_call(tc):
    if tc.get("_summarized"):
        return tc
    name = str(tc.get("name") or "unknown")
    args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
    mcp_key = None
    if name == "call_mcp_tool" or name.startswith("mcp_"):
        mcp_key = (f"{args.get('ServerName') or 'unknown'}/{args.get('ToolName') or 'unknown'}"
                   if name == "call_mcp_tool" else name[4:].replace("_", "/", 1))
    skill = None
    for k in ("Path", "AbsolutePath", "path", "CommandLine", "ScriptPath"):
        v = args.get(k)
        if not isinstance(v, str):
            continue
        m = SKILL_RE.search(v)
        if m:
            skill = (m.group(1), m.group(2).strip().lower().startswith("skill.md"))
            break
    return {
        "_summarized": True,
        "name": name,
        "_arg_chars": len(json.dumps(args, ensure_ascii=False, separators=(",", ":"))),
        "_tc_chars": len(json.dumps(tc, ensure_ascii=False, separators=(",", ":"))),
        "_mcp_key": mcp_key,
        "_skill": skill,
    }


def _note_tool_call(u, tc, step):
    sum_tc = _summarize_tool_call(tc)
    name = sum_tc["name"]
    t = _tool(u, name)
    t["calls"] += 1
    t["arg_chars"] += sum_tc["_arg_chars"]
    if sum_tc.get("_mcp_key"):
        key = sum_tc["_mcp_key"]
        u["mcp"][key] = int(u["mcp"].get(key) or 0) + 1
    # skills: reading SKILL.md = load; other files under skills/<name>/ = resource use
    if sum_tc.get("_skill"):
        sk_name, is_load = sum_tc["_skill"]
        s = u["skills"].setdefault(sk_name, {"loads": 0, "resource_uses": 0,
                                             "first_step": step, "last_step": step})
        if is_load:
            s["loads"] += 1
        else:
            s["resource_uses"] += 1
        s["last_step"] = step
    return sum_tc


def _credit_tool(u, cat, share):
    if cat.startswith("tool:"):
        _tool(u, cat[5:])["result_tokens"] += int(round(share))
    elif cat.startswith("args:"):
        t = _tool(u, cat[5:])
        t["arg_tokens"] = int(t.get("arg_tokens") or 0) + int(round(share))


def _settle(u, prompt):
    """A model call with real prompt size `prompt` happened: attribute growth to pending items."""
    w, pending = u["window"], u["pending"]
    chars = sum(p[1] for p in pending)
    if u["last_prompt"] is None:
        est = min(prompt, chars / CHARS_PER_TOKEN)
        for cat, c, _ in pending:
            share = est * c / chars if chars else 0
            _book(u, cat, c, share)
            _credit_tool(u, cat, share)
        _book(u, "baseline", 0, prompt - est)
        w["first_prompt"] = prompt
    else:
        delta = prompt - u["last_prompt"]
        if delta < 0:
            _book(u, "trimmed", 0, delta)
            delta = 0
        if not pending:
            if delta:
                _book(u, "unattributed", 0, delta)
        for cat, c, _ in pending:
            share = delta * c / chars if chars else delta / len(pending)
            _book(u, cat, c, share)
            _credit_tool(u, cat, share)
    u["pending"] = []
    u["last_prompt"] = prompt
    w["prompt"] = prompt


def compact_step(o):
    """Shrink a transcript step for low-memory usage replay while keeping exact char counts."""
    typ = o.get("type")
    step = o.get("step_index")
    content = o.get("content") or ""
    if typ == "PLANNER_RESPONSE":
        tcs = [tc for tc in (o.get("tool_calls") or []) if isinstance(tc, dict)]
        return {
            "type": typ,
            "step_index": step,
            "input_tokens": o.get("input_tokens"),
            "cache_read_tokens": o.get("cache_read_tokens"),
            "_content_len": len(content),
            "_thinking_len": len(o.get("thinking") or ""),
            "tool_calls": [_summarize_tool_call(tc) for tc in tcs] if tcs else None,
        }
    if typ == "SYSTEM_MESSAGE":
        return {"type": typ, "step_index": step, "_content_len": len(content), "_sys_cat": _system_cat(o)}
    if typ == "ERROR_MESSAGE":
        return {"type": typ, "step_index": step, "_error_len": len(o.get("error") or content)}
    if typ == "GENERIC":
        return {"type": typ, "step_index": step, "_content_len": len(content),
                "status": o.get("status"), "error": bool(o.get("error"))}
    if typ in ("CHECKPOINT", "USER_INPUT"):
        return {"type": typ, "step_index": step, "_content_len": len(content)}
    return {"type": typ, "step_index": step}


def feed(u, o):
    """Consume one transcript object (raw or compact_step)."""
    typ = o.get("type")
    step = o.get("step_index")
    content_len = o["_content_len"] if "_content_len" in o else len(o.get("content") or "")
    if typ == "CHECKPOINT":
        u["compactions"] += 1
        u["window"] = {"cats": {}, "first_prompt": None, "prompt": None, "since_step": step}
        u["last_prompt"] = None
        u["queue"] = []
        u["pending"] = [["checkpoint", content_len, None]]
        return
    if typ == "PLANNER_RESPONSE":
        try:
            prompt = int(o.get("input_tokens") or 0) + int(o.get("cache_read_tokens") or 0)
        except (TypeError, ValueError):
            prompt = 0
        if prompt > 0:
            _settle(u, prompt)
        # this call's output becomes part of the next prompt
        thinking_len = o["_thinking_len"] if "_thinking_len" in o else len(o.get("thinking") or "")
        if thinking_len > 0:
            u["pending"].append(["model:thinking", thinking_len, None])
        if content_len > 0:
            u["pending"].append(["model:text", content_len, None])
        tcs = [tc for tc in (o.get("tool_calls") or []) if isinstance(tc, dict)]
        if tcs:
            sums = [_note_tool_call(u, tc, step) for tc in tcs]
            u["queue"] = [s["name"] for s in sums]
            for s in sums:
                u["pending"].append([f"args:{s['name']}", s["_tc_chars"], None])
        return
    if typ == "GENERIC":
        name = u["queue"].pop(0) if u["queue"] else "unknown"
        t = _tool(u, name)
        t["result_chars"] += content_len
        if o.get("status") == "ERROR" or o.get("error"):
            t["errors"] += 1
        u["pending"].append([f"tool:{name}", content_len, name])
    elif typ == "USER_INPUT":
        u["pending"].append(["user_input", content_len, None])
    elif typ == "SYSTEM_MESSAGE":
        u["pending"].append([o.get("_sys_cat") or _system_cat(o), content_len, None])
    elif typ == "ERROR_MESSAGE":
        err_len = o["_error_len"] if "_error_len" in o else len(o.get("error") or (o.get("content") or ""))
        u["pending"].append(["error", err_len, None])


def compact_for_state(u):
    """Keep the persisted state small: cap the tools table."""
    if len(u["tools"]) > TOP_TOOLS_IN_STATE:
        keep = sorted(u["tools"].items(), key=lambda kv: -kv[1]["result_chars"])[:TOP_TOOLS_IN_STATE]
        u["tools"] = dict(keep)
    return u


def ranked(cats, key="tokens"):
    return sorted(((c, e) for c, e in cats.items() if c != "trimmed"), key=lambda kv: -kv[1][key])


def top_line(u, n=4):
    """One-line summary of the biggest growth sources in the current window (for warnings)."""
    cats = (u or {}).get("window", {}).get("cats") or {}
    items = [(c, e) for c, e in ranked(cats) if c != "baseline" and e["tokens"] > 0][:n]
    if not items:
        return ""
    return "本窗口上下文主要来源：" + "、".join(f"{label(c)} ≈{e['tokens'] // 1000 or '<1'}k" for c, e in items)


def render_markdown(u, title="Context 使用分类统计"):
    out = [f"## {title}"]
    w = u.get("window") or {}
    prompt = w.get("prompt")
    out.append(f"当前窗口：prompt ≈ {prompt or 0:,} token；自第 {w.get('since_step') or 0} 步起；"
               f"共压缩 {u.get('compactions', 0)} 次。token 为按字符比例归因的估算，字符数为精确值。")
    for name, cats in (("当前窗口（上次压缩以来）", w.get("cats") or {}),
                       ("整个对话累计", (u.get("total") or {}).get("cats") or {})):
        tot = sum(max(0, e["tokens"]) for c, e in cats.items() if c != "trimmed") or 1
        out += [f"### {name}", "| 来源 | token≈ | 占比 | 字符 | 条数 |", "|---|---:|---:|---:|---:|"]
        for c, e in ranked(cats):
            out.append(f"| {label(c)} | {e['tokens']:,} | {100 * max(0, e['tokens']) / tot:.1f}% | "
                       f"{e['chars']:,} | {e['n']} |")
        if "trimmed" in cats:
            out.append(f"| {label('trimmed')} | {cats['trimmed']['tokens']:,} | | | {cats['trimmed']['n']} |")
    tools = sorted((u.get("tools") or {}).items(), key=lambda kv: -(kv[1]["result_tokens"] + kv[1].get("arg_tokens", 0)))
    out += ["### 工具使用", "| 工具 | 调用 | 结果 token≈ | 参数 token≈ | 结果字符 | 参数字符 | 错误 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    out += [f"| {k} | {v['calls']} | {v['result_tokens']:,} | {v.get('arg_tokens', 0):,} | {v['result_chars']:,} | {v['arg_chars']:,} | "
            f"{v['errors']} |" for k, v in tools] or ["| （无） | | | | | | |"]
    out += ["### Skill 使用", "| Skill | 加载 SKILL.md | 读取资源/脚本 | 首次步 | 最近步 |", "|---|---:|---:|---:|---:|"]
    out += [f"| {k} | {v['loads']} | {v['resource_uses']} | {v['first_step']} | {v['last_step']} |"
            for k, v in sorted((u.get("skills") or {}).items())] or ["| （无） | | | | |"]
    if u.get("mcp"):
        out += ["### MCP 调用", "| 服务/工具 | 次数 |", "|---|---:|"]
        out += [f"| {k} | {v} |" for k, v in sorted(u["mcp"].items(), key=lambda kv: -kv[1])]
    return "\n".join(out)
