"""recall.py - query the lossless transcript_full.jsonl of an Antigravity conversation.

Usage:
  python recall.py [--transcript PATH | --conv ID] [--all-branches] users
  python recall.py ... grep REGEX [--type TYPE] [--max N] [--context C]
  python recall.py ... step N
  python recall.py ... range A B [--brief]
  python recall.py ... files
  python recall.py ... messages [--sender PREFIX]

Long output (> --limit chars) is written to a file next to the transcript's
conversation artifacts (scratch/) and only the path + preview is printed.
"""
import argparse
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))
import common as C  # noqa: E402


def fmt_step(o, full=True, cap=None):
    head = f"--- step {o.get('step_index')} | {o.get('type')} | {o.get('created_at', '')}"
    body = []
    if o.get("type") == "USER_INPUT":
        body.append(C.user_text(o))
    else:
        if o.get("thinking") and full:
            body.append("[thinking]\n" + o["thinking"])
        if o.get("content"):
            body.append(o["content"])
        if o.get("tool_calls"):
            body.append("[tool_calls] " + json.dumps(o["tool_calls"], ensure_ascii=False))
        if o.get("error"):
            body.append("[error] " + str(o["error"]))
    text = "\n".join(body)
    if cap:
        text = C.head_tail(text, cap)
    return head + "\n" + text


def brief(o):
    names = ",".join(tc.get("name", "?") for tc in (o.get("tool_calls") or []) if isinstance(tc, dict))
    snippet = (o.get("content") or o.get("thinking") or "").replace("\n", " ")[:100]
    return f"{o.get('step_index'):>5} {o.get('type', ''):<17} {names:<30} {snippet}"


def sidx(o):
    try:
        return int(o.get("step_index"))
    except (TypeError, ValueError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript")
    ap.add_argument("--conv")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--all-branches", action="store_true", default=False,
                    help="包含用户编辑/重试后被丢弃的旧分支步骤")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_users = sub.add_parser("users")
    g = sub.add_parser("grep")
    g.add_argument("pattern")
    g.add_argument("--type")
    g.add_argument("--max", type=int, default=20)
    g.add_argument("--context", type=int, default=300)
    s = sub.add_parser("step")
    s.add_argument("n", type=int)
    r = sub.add_parser("range")
    r.add_argument("a", type=int)
    r.add_argument("b", type=int)
    r.add_argument("--brief", action="store_true")
    p_files = sub.add_parser("files")
    mg = sub.add_parser("messages")
    mg.add_argument("--sender", help="只看某个发送方（会话 ID 前缀或 swarm 成员名）")
    for sp in (p_users, g, s, r, p_files, mg):
        sp.add_argument("--all-branches", action="store_true", default=argparse.SUPPRESS,
                        help="包含用户编辑/重试后被丢弃的旧分支步骤")
    a = ap.parse_args()

    path = C.resolve_full_transcript(a.transcript, a.conv)
    if not path:
        print("找不到 transcript，请提供 --transcript 或 --conv")
        return 2
    rows = list(C.iter_jsonl(path)) if a.all_branches else C.iter_active_steps(path)
    out = []

    if a.cmd == "users":
        for i, o in enumerate([x for x in rows if x.get("type") == "USER_INPUT"], 1):
            out.append(f"{i}. [step {o.get('step_index')} {o.get('created_at', '')}]\n{C.user_text(o)}\n")
    elif a.cmd == "grep":
        try:
            rx = re.compile(a.pattern, re.I)
        except re.error as e:
            print(f"正则表达式无效: {e}", file=sys.stderr)
            return 2
        hits = 0
        for o in rows:
            if a.type and o.get("type") != a.type:
                continue
            fields = [("content", o.get("content") or ""), ("thinking", o.get("thinking") or "")]
            if o.get("tool_calls"):
                fields.append(("tool_calls", json.dumps(o["tool_calls"], ensure_ascii=False)))
            for name, text in fields:
                if not text:
                    continue
                m = rx.search(text)
                if not m:
                    continue
                lo, hi = max(0, m.start() - a.context), min(len(text), m.end() + a.context)
                out.append(f"--- step {o.get('step_index')} | {o.get('type')} | {name}\n…{text[lo:hi]}…\n")
                hits += 1
                break
            if hits >= a.max:
                out.append(f"(已达 --max {a.max}，可能还有更多匹配)")
                break
        if not hits:
            out.append("无匹配")
    elif a.cmd == "step":
        found = [o for o in rows if sidx(o) == a.n]
        out += [fmt_step(o) for o in found] or [f"step {a.n} 不存在"]
    elif a.cmd == "range":
        sel = [o for o in rows if a.a <= sidx(o) <= a.b]
        out += [brief(o) if a.brief else fmt_step(o, cap=3000) for o in sel]
    elif a.cmd == "files":
        edited = []
        for o in rows:
            edited += [(sidx(o), p) for p in C.edited_paths(o)]
        last = {}
        for s_, p in edited:
            last[p.replace("\\", "/").lower()] = (s_, p)
        for s_, p in sorted(last.values()):
            out.append(f"{s_:>5} {p}")
        if not out:
            out.append("（无）")
    elif a.cmd == "messages":
        col = C.AgentCollector()
        for o in rows:
            col.feed(o)
        f = (a.sender or "").lower()
        ok = lambda who: not f or str(who).lower().startswith(f)  # noqa: E731
        out.append("## 收到的代理消息")
        out += [f"--- step {m['step']} | {m['time']} | from {m['sender']}\n{m['text']}\n"
                for m in col.incoming if ok(m["sender"])] or ["（无）"]
        out.append("## 派发的 subagent / swarm worker")
        out += [f"--- step {d['step']} | {d['kind']} {d['label']} ({d['type']}) | conv={d['conv']}\n{d['text']}\n"
                for d in col.dispatched if ok(d["label"]) or ok(d["conv"] or "")] or ["（无）"]
        out.append("## 发出的消息")
        out += [f"--- step {m['step']} | to {m['to']}\n{m['text']}\n"
                for m in col.outgoing if ok(m["to"])] or ["（无）"]
        out.append("## swarm 消息板（本会话读到过的）")
        out += [f"#{m['seq']} [step {m['step']}] {m['agent']}: {m['text']}"
                for m in sorted(col.board.values(), key=lambda x: x["seq"]) if ok(m["agent"])] or ["（无）"]

    text = "\n".join(out)
    if len(text) > a.limit:
        norm = os.path.abspath(os.path.normpath(path))
        logs_dir = os.path.dirname(norm)
        sysgen_dir = os.path.dirname(logs_dir)
        conv_dir = os.path.dirname(sysgen_dir)
        if (os.path.basename(logs_dir).lower() == "logs"
                and os.path.basename(sysgen_dir).lower() == ".system_generated"
                and conv_dir and conv_dir != os.path.dirname(conv_dir)):
            d = os.path.join(conv_dir, "scratch")
        else:
            d = logs_dir
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, f"recall_{a.cmd}_{time.strftime('%H%M%S')}.txt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"输出 {len(text)} 字符，已写入：{f}\n预览：\n{text[:3000]}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
