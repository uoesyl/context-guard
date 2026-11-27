"""Context usage breakdown for one conversation (context-guard).

Recomputes from transcript_full.jsonl, so it works for any conversation, with or without
hook state. Usage:

  python context_stats.py                       # latest conversation under ~/.gemini/antigravity/brain
  python context_stats.py <conversation-id>
  python context_stats.py --conv <conversation-id>
  python context_stats.py --transcript <path/to/transcript_full.jsonl>
  python context_stats.py <conv> --json         # machine-readable
"""
import argparse
import glob
import json
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hooks"))
import common as C  # noqa: E402
import usage as U  # noqa: E402

BRAIN = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "brain")


def find_transcript(conv=None):
    if conv:
        p = os.path.join(BRAIN, conv, ".system_generated", "logs", "transcript_full.jsonl")
        return p if os.path.isfile(p) else None
    cands = glob.glob(os.path.join(BRAIN, "*", ".system_generated", "logs", "transcript_full.jsonl"))
    return max(cands, key=os.path.getmtime) if cands else None


def compute(path):
    u = U.new_usage()
    for o in C.iter_active_steps(path, transform=U.compact_step):
        U.feed(u, o)
    return u


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conv", nargs="?")
    ap.add_argument("--conv", dest="conv_opt")
    ap.add_argument("--transcript")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    conv_arg = a.conv_opt or a.conv
    path = a.transcript or find_transcript(conv_arg)
    if not path or not os.path.isfile(path):
        print("找不到 transcript_full.jsonl", file=sys.stderr)
        return 1
    u = compute(path)
    u.pop("pending", None)
    u.pop("queue", None)
    if a.json:
        sys.stdout.write(json.dumps(u, ensure_ascii=False, indent=1))
    else:
        parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
        conv = (parts[-4] if len(parts) >= 4 and parts[-2].lower() == "logs"
                and parts[-3].lower() == ".system_generated" else (conv_arg or ""))
        title = f"Context 使用分类统计 {conv[:8]}".rstrip()
        out = U.render_markdown(u, title)
        sys.stdout.buffer.write((out + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
