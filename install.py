"""Register / remove the Context Guard hooks in ~/.gemini/config/hooks.json.

Why a global hook instead of the plugin's own hooks.json: plugin hooks do not run in swarm
workers, and the recovery pack must reach every conversation. Antigravity silently ignores
the WHOLE hooks.json when it does not parse (e.g. a UTF-8 BOM), so this script always writes
strict UTF-8 without BOM and keeps a backup of the previous file.

Usage:
  python install.py              # add / update the "context-guard" entry
  python install.py --dry-run    # print the resulting hooks.json, change nothing
  python install.py --uninstall  # remove the entry (other hooks are kept)
  python install.py --python "C:/Python313/python.exe"   # interpreter used by the hook
"""
import argparse
import json
import os
import shutil
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = os.path.join(HERE, "hooks", "guard.py")
KEY = "context-guard"


def hooks_path():
    return os.path.join(os.path.expanduser("~"), ".gemini", "config", "hooks.json")


def q(p):
    p = p.replace("\\", "/")
    return f'"{p}"' if " " in p else p


def entry(python):
    cmd = f"{q(python)} {q(GUARD)}"
    return {
        "PreInvocation": [{"type": "command", "command": cmd, "timeout": 10}],
        "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": cmd, "timeout": 10}]}],
    }


def load(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        print("note: existing hooks.json had a UTF-8 BOM (Antigravity ignores such files); it will be removed")
    text = raw.decode("utf-8-sig").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"error: {path} is not valid JSON ({e}); fix or move it away first")
    if not isinstance(data, dict):
        sys.exit(f"error: {path} must contain a JSON object")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable, help="interpreter for the hook (default: this one)")
    ap.add_argument("--hooks-file", default=hooks_path())
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if sys.version_info < (3, 9):
        sys.exit("error: Python 3.9+ is required")
    data = load(a.hooks_file)
    if a.uninstall:
        if data.pop(KEY, None) is None:
            print(f"{KEY} is not registered in {a.hooks_file}; nothing to do")
            return 0
    else:
        data[KEY] = entry(a.python)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if a.dry_run:
        print(text)
        return 0
    os.makedirs(os.path.dirname(a.hooks_file), exist_ok=True)
    if os.path.isfile(a.hooks_file):
        shutil.copy2(a.hooks_file, a.hooks_file + ".bak")
    tmp = a.hooks_file + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:  # no BOM
        f.write(text)
    os.replace(tmp, a.hooks_file)
    action = "removed from" if a.uninstall else "registered in"
    print(f"{KEY} hooks {action} {a.hooks_file}")
    if not a.uninstall:
        print(f"  command: {data[KEY]['PreInvocation'][0]['command']}")
        print("New conversations pick it up immediately; running ones on their next model call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
