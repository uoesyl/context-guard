"""Replay real local transcripts through guard.py.

Run: python tests/test_guard.py
Uses a temp dir for state and transcript copies; never touches originals.

Replay timing mirrors real Antigravity auto-compactions: the CHECKPOINT
step is written AFTER the PreInvocation hook of the next model call (compaction
happens between the hook and the model call), and PreToolUse runs for every tool
call of that model response.
"""
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD = os.path.join(PLUGIN, "hooks", "guard.py")
RECALL = os.path.join(PLUGIN, "scripts", "recall.py")
BRAIN = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "brain")
FIX = os.path.join(PLUGIN, "tests", "fixtures")
TMP = tempfile.mkdtemp(prefix="ctxguard_")
os.environ["CONTEXT_GUARD_DATA"] = os.path.join(TMP, "data")
sys.path.insert(0, os.path.join(PLUGIN, "hooks"))
import guard  # noqa: E402

guard.DATA_DIR = os.environ["CONTEXT_GUARD_DATA"]
FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def call(payload):
    return guard.main(payload)


def text_of(res):
    if res and res.get("injectSteps"):
        return res["injectSteps"][0]["systemMessage"]["systemMessage"]
    return ""


def replay(conv_id):
    src = os.path.join(BRAIN, conv_id, ".system_generated", "logs", "transcript_full.jsonl")
    lines = open(src, "rb").read().splitlines(keepends=True)
    d = os.path.join(TMP, conv_id, ".system_generated", "logs")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "transcript_full.jsonl")
    open(dst, "wb").close()
    base = {"conversationId": "replay-" + conv_id, "transcriptPath": os.path.join(d, "transcript.jsonl"),
            "artifactDirectoryPath": os.path.join(TMP, conv_id)}
    ev = {"pack_inject": [], "pack_deny": [], "note": [], "w1": [], "w2": [], "leak": [], "sizes": []}
    times, deferred, since_ckpt = [], None, None
    with open(dst, "ab") as out:
        for raw in lines:
            o = json.loads(raw)
            if o.get("type") == "CHECKPOINT":
                deferred, since_ckpt = raw, int(o["step_index"])
                continue
            if o.get("type") == "PLANNER_RESPONSE":
                out.flush()
                t0 = time.perf_counter()
                m = text_of(call(dict(base, hookEventName="PreInvocation")))
                times.append(time.perf_counter() - t0)
                s = int(o["step_index"])
                if "恢复包已在上一轮" in m:
                    ev["note"].append(s)
                    since_ckpt = None
                elif "## Context Guard：第 " in m and ("发生了自动压缩" in m or "发生了手动压缩" in m):
                    ev["pack_inject"].append(s)
                    ev["sizes"].append(len(m))
                    since_ckpt = None
                elif m.startswith("## Context Guard 预警"):
                    ev["w1"].append(s)
                elif m.startswith("## Context Guard 紧急"):
                    ev["w2"].append(s)
                if m:  # simulate a successful injection being recorded
                    out.write((json.dumps({"step_index": -1, "type": "SYSTEM_MESSAGE", "content": m},
                                          ensure_ascii=False) + "\n").encode("utf-8"))
                if deferred:  # compaction lands between the hook and the model call
                    out.write(deferred)
                    deferred = None
                out.write(raw)
                out.flush()
                for i, tc in enumerate(o.get("tool_calls") or []):
                    r = call(dict(base, hookEventName="PreToolUse", stepIdx=s,
                                  toolCall={"id": str(i), "name": tc.get("name"), "args": tc.get("args")}))
                    if r.get("decision") == "deny":
                        # simulate the tool result (deny reason) being recorded in the transcript
                        out.write((json.dumps({"step_index": -1, "type": "GENERIC", "content": r["reason"]},
                                              ensure_ascii=False) + "\n").encode("utf-8"))
                        out.flush()
                        if "【Context Guard 拦截】刚刚发生了自动压缩" in r["reason"] or "【Context Guard 拦截】刚刚发生了手动压缩" in r["reason"]:
                            ev["pack_deny"].append(s)
                            ev["sizes"].append(len(r["reason"]))
                    elif since_ckpt is not None and tc.get("name") in guard.MUTATING_TOOLS:
                        ev["leak"].append((s, tc.get("name")))
                    check_neutral = r.get("decision") in ("ask", "deny")
                    if not check_neutral:
                        FAIL.append(f"non-neutral PreToolUse decision {r}")
                continue
            out.write(raw)
    return ev, times


def recovery_for(fixture):
    """Copy fixture, append a synthetic CHECKPOINT, return the PreInvocation pack."""
    name = os.path.splitext(fixture)[0]
    d = os.path.join(TMP, "fx_" + name, ".system_generated", "logs")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "transcript_full.jsonl")
    shutil.copy(os.path.join(FIX, fixture), dst)
    pl = {"conversationId": "fx-" + name, "transcriptPath": os.path.join(d, "transcript.jsonl"),
          "artifactDirectoryPath": os.path.join(TMP, "fx_" + name), "hookEventName": "PreInvocation"}
    call(pl)
    with open(dst, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": "999", "type": "CHECKPOINT", "content": "# Resuming from a compaction"}) + "\n")
    return text_of(call(pl))


def first_run_injection(fixture):
    name = "first_" + os.path.splitext(fixture)[0]
    d = os.path.join(TMP, name, ".system_generated", "logs")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(FIX, fixture), os.path.join(d, "transcript_full.jsonl"))
    return text_of(call({"conversationId": name, "transcriptPath": os.path.join(d, "transcript.jsonl"),
                         "artifactDirectoryPath": os.path.join(TMP, name), "hookEventName": "PreInvocation"}))


def multi_agent_tests():
    m = first_run_injection("swarm_worker.jsonl")
    check("Context Guard" in m and "task_state.md" in m, "swarm worker: rule injected on first run")
    check(first_run_injection("subagent.jsonl") == "", "subagent: no rule injection (it loads plugin rules)")
    check(first_run_injection("orchestrator.jsonl") == "", "orchestrator: no rule injection")

    d = os.path.join(TMP, "late_worker", ".system_generated", "logs")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "transcript_full.jsonl")
    open(dst, "w", encoding="utf-8").close()
    pl = {"conversationId": "late_worker", "transcriptPath": dst, "artifactDirectoryPath": TMP}
    r0 = call(dict(pl, hookEventName="PreToolUse", toolCall={"name": "read_file"}))
    r1 = call(dict(pl, hookEventName="PreInvocation"))
    shutil.copy(os.path.join(FIX, "swarm_worker.jsonl"), dst)
    r2 = call(dict(pl, hookEventName="PreInvocation"))
    r3 = call(dict(pl, hookEventName="PreInvocation"))
    check(r0 == {"decision": "ask"} and not r1 and text_of(r2) and not r3,
          "swarm worker: late transcript still gets rule exactly once")

    m = recovery_for("subagent.jsonl")
    check("Get-Date" in m and "编排者下发的任务说明" in m, "subagent: parent's task text recovered verbatim")
    check("（无：这是 subagent" in m, "subagent: no-user-input notice shown")

    m = recovery_for("swarm_worker.jsonl")
    check("Your assignment" in m and "PROBE-FOLLOWUP" in m, "swarm worker: brief + later board post recovered")
    check("中文消息测试 ABC-123" in m, "swarm worker: Chinese board text intact")
    check("new board message" not in m, "swarm worker: board doorbell noise filtered")

    m = recovery_for("orchestrator.jsonl")
    for cid in ("61f53ede", "100979ef", "537d32c1"):
        check(f"conv={cid}" in m, f"orchestrator: dispatched worker {cid} listed with conv id")
    check("Conversation ID: 61f53ede-4a1d-4e10-b5a2-5f6581ac06dc, DONE" in m, "orchestrator: subagent report recovered")
    check("#2 probe:\n> @coordinator PROBE-REPORT" in m, "orchestrator: board message from worker recovered")
    check("#3 coordinator" not in m.split("### swarm 消息板")[-1].split("\n### ")[0],
          "orchestrator: no false board entries from embedded tool output")
    check("编排者下发的任务说明" not in m, "orchestrator: subagent report not mislabeled as its task")

    r = subprocess.run([sys.executable, RECALL, "--transcript", os.path.join(FIX, "orchestrator.jsonl"),
                        "messages", "--sender", "61f5"], capture_output=True)
    out = r.stdout.decode("utf-8", errors="replace")
    check(r.returncode == 0 and "61f53ede" in out and "100979ef" not in out.split("## 派发")[0],
          "recall messages --sender filters")


def structure_ok(pack):
    """Every '#' heading line in the pack must belong to the pack's own skeleton."""
    own = ("## Context Guard", "### ")
    in_board = False
    for ln in pack.splitlines():
        if ln.startswith("### "):
            in_board = ln.startswith("### swarm 消息板")
        if ln.startswith("#") and not ln.startswith(own):
            if in_board and re.match(r"^#\d+ .+:$", ln):
                continue
            return False, ln
    heads = [ln for ln in pack.splitlines() if ln.startswith("### ")]
    allowed = ("### 用户原话", "### 当前执行计划", "### 最近一次交回", "### 其他代理", "### 你派发过",
               "### swarm 消息板", "### 你最近发给", "### task_state.md", "### 本会话修改过", "### 召回原文")
    bad = [h for h in heads if not h.startswith(allowed)]
    return (not bad), (bad[:1] or [""])[0]


def sidecar_linkage_tests():
    conv = "linkage-test-conv"
    d = os.path.join(TMP, conv, ".system_generated", "logs")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "transcript_full.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 1, "type": "USER_INPUT", "content": "<USER_REQUEST>hi</USER_REQUEST>"}) + "\n")
    pl = {"conversationId": conv, "transcriptPath": os.path.join(d, "transcript.jsonl"),
          "artifactDirectoryPath": os.path.join(TMP, conv)}

    # First PreInvocation initializes state cleanly
    r0 = call(dict(pl, hookEventName="PreInvocation"))
    check(r0 == {}, "linkage: initial PreInvocation returns {}")

    # Model responds and crosses level 1 -> PostInvocation is inert (user_markdown never displayed
    # and drops the batch), PreInvocation delivers a plain model-facing warning
    with open(dst, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 2, "type": "PLANNER_RESPONSE", "input_tokens": 365000, "cache_read_tokens": 0, "output_tokens": 500}) + "\n")
    r_post = call(dict(pl, hookEventName="PostInvocation"))
    check(r_post == {}, "linkage: PostInvocation is inert")

    r_pre = call(dict(pl, hookEventName="PreInvocation"))
    sm_pre = (r_pre.get("injectSteps") or [{}])[0].get("systemMessage", {})
    check("Context Guard 预警" in sm_pre.get("systemMessage", "") and "metadata" not in sm_pre,
          "linkage: PreInvocation warns model with plain systemMessage")

    st = guard.load_state(conv)
    check(st and st.get("warned") == 1 and st.get("model_warned") == 1 and st.get("last_event") == "warn_1",
          "linkage: state tracks warned, model_warned, and last_event")
    check((st.get("levels") or {}).get("token") == [[350000, 1], [420000, 2]],
          "linkage: state carries threshold levels for the meter")
    check("updated_at" in st and st.get("compaction_count") == 0 and st.get("task_state_exists") is False,
          "linkage: state tracks updated_at, compaction_count, and task_state_exists")
    check(st.get("input_tokens") == 365000 and st.get("cache_read_tokens") == 0
          and st.get("output_tokens") == 500 and st.get("cache_hit_rate") == 0.0,
          "modeling: state records input/cache_read/output tokens and cache_hit_rate")
    check((st.get("cycle_stats") or {}).get("output_tokens") == 500
          and (st.get("cycle_stats") or {}).get("cache_misses") == 1,
          "modeling: cycle_stats accumulates output_tokens and cache_misses")

    # Compaction increments compaction_count and emits recovery user_markdown
    with open(dst, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 3, "type": "CHECKPOINT", "content": "# Resuming"}) + "\n")
    r_ckpt = call(dict(pl, hookEventName="PreInvocation"))
    sm_ckpt = (r_ckpt.get("injectSteps") or [{}])[0].get("systemMessage", {})
    check("Context Guard：第 3 步发生了" in sm_ckpt.get("systemMessage", "") and "metadata" not in sm_ckpt,
          "linkage: recovery pack is plain systemMessage (metadata would drop the whole batch)")
    st2 = guard.load_state(conv)
    check(st2.get("compaction_count") == 1 and st2.get("warned") == 0 and st2.get("model_warned") == 0,
          "linkage: compaction resets warned/model_warned and increments compaction_count")
    check((st2.get("cycle_stats") or {}).get("steps") == 0
          and (st2.get("total_stats") or {}).get("prompt_tokens") == 365000
          and (st2.get("total_stats") or {}).get("output_tokens") == 500,
          "modeling: compaction resets cycle_stats while preserving total_stats")
    check((st2.get("verify") or {}).get("ckpt") == 3, "delivery: verify armed after injection")

    # Injection dropped: model responds with no SYSTEM_MESSAGE -> re-armed
    with open(dst, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 4, "type": "PLANNER_RESPONSE", "input_tokens": 3000, "cache_read_tokens": 6000, "output_tokens": 200}) + "\n")
    r_tool = call(dict(pl, hookEventName="PreToolUse", toolCall={"name": "edit_file"}, stepIdx=4))
    check(r_tool.get("decision") == "deny" and "Context Guard：第 3 步发生了" in r_tool.get("reason", ""),
          "delivery: dropped injection re-armed and redelivered via PreToolUse deny")
    st3 = guard.load_state(conv)
    check(st3.get("last_delivery") == "failed", "delivery: failure recorded")
    check(abs((st3.get("cache_hit_rate") or 0) - 0.6667) < 0.001
          and (st3.get("total_stats") or {}).get("cache_read_tokens") == 6000
          and (st3.get("total_stats") or {}).get("output_tokens") == 700,
          "modeling: cache_hit_rate and total_stats updated across post-compaction steps")

    # Delivered case: SYSTEM_MESSAGE with marker before the response -> confirmed
    with open(dst, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 5, "type": "CHECKPOINT", "content": "# Resuming"}) + "\n")
    call(dict(pl, hookEventName="PreInvocation"))
    with open(dst, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 6, "type": "SYSTEM_MESSAGE",
                            "content": "injected: ## Context Guard：第 5 步发生了自动压缩"}) + "\n")
        f.write(json.dumps({"step_index": 7, "type": "PLANNER_RESPONSE", "input_tokens": 9000}) + "\n")
    call(dict(pl, hookEventName="PreToolUse", toolCall={"name": "read_file"}, stepIdx=7))
    st4 = guard.load_state(conv)
    check(st4.get("last_delivery") == "ok" and not st4.get("verify") and not st4.get("pending"),
          "delivery: confirmed when SYSTEM_MESSAGE carries the marker")


def manual_and_stale_tests():
    def setup(conv):
        d = os.path.join(TMP, conv, ".system_generated", "logs")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "transcript_full.jsonl")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>做事</USER_REQUEST>"}) + "\n")
            f.write(json.dumps({"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000}) + "\n")
        pl = {"conversationId": conv, "transcriptPath": dst, "artifactDirectoryPath": os.path.join(TMP, conv)}
        call(dict(pl, hookEventName="PreInvocation"))
        return dst, pl

    def add(dst, *objs):
        with open(dst, "a", encoding="utf-8") as f:
            for o in objs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # Manual /compact: CHECKPOINT is written before the next user turn -> full pack at PreInvocation.
    dst, pl = setup("manual-compact")
    add(dst, {"step_index": 2, "type": "USER_INPUT", "content": "<USER_REQUEST>\n/compact \n</USER_REQUEST>"},
        {"step_index": 3, "type": "CHECKPOINT", "content": "# Resuming from a compaction"},
        {"step_index": 4, "type": "USER_INPUT", "content": "<USER_REQUEST>继续</USER_REQUEST>"})
    m = text_of(call(dict(pl, hookEventName="PreInvocation")))
    check("第 3 步发生了手动压缩（/compact）" in m and "### 用户原话" in m,
          "manual /compact: pack delivered at the very next PreInvocation, labelled manual")
    r = call(dict(pl, hookEventName="PreToolUse", toolCall={"name": "run_command"}))
    check(r == {"decision": "ask"}, "manual /compact: no deny after the pack was injected")

    # Stale: hook was down, catches up with many model calls after the checkpoint -> no deny.
    dst, pl = setup("stale-ckpt")
    add(dst, {"step_index": 2, "type": "CHECKPOINT", "content": "# Resuming from a compaction"},
        *[{"step_index": 3 + i, "type": "PLANNER_RESPONSE", "input_tokens": 5000} for i in range(10)])
    r = call(dict(pl, hookEventName="PreToolUse", toolCall={"name": "edit_file"}))
    check(r == {"decision": "ask"}, "stale checkpoint: mutating tool not denied when catching up")
    m = text_of(call(dict(pl, hookEventName="PreInvocation")))
    check("Context Guard 补发" in m and "第 2 步发生了自动压缩" in m, "stale checkpoint: pack re-sent via PreInvocation")


def _mini_conv(conv, objs):
    d = os.path.join(TMP, conv, ".system_generated", "logs")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "transcript_full.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return dst, {"conversationId": conv, "transcriptPath": dst, "artifactDirectoryPath": os.path.join(TMP, conv)}


def _append(dst, *objs):
    with open(dst, "a", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def calibration_tests():
    import calib as K
    L = K.levels_for({}, None)
    check(L["target"] == 512000 and L["token"] == [[350000, 1], [420000, 2]] and L["range"] == [480000, 524000],
          "calib: unknown model keeps the historical 512k / 350k / 420k levels")
    check(K.levels_for({}, "gemini-2.5-flash")["source"] == "builtin", "calib: gemini family uses builtin 512k")
    for m in ("claude-opus-4-medium", "claude-opus-4-high"):
        Lo = K.levels_for({"models": {m: {"db_limit": 128000, "max_prompt": 195472}}}, m)
        check(Lo["target"] == 512000 and Lo["source"] == "builtin" and Lo["token"] == [[350000, 1], [420000, 2]],
              f"calib: {m} uses the Gemini 512k levels, not the DB contextLimit")
    check(not K.needs_db({}, "claude-opus-4-medium"), "calib: builtin claude-opus skips DB lookup")
    cal = {"models": {"claude-x": {"db_limit": 128000}}}
    L = K.levels_for(cal, "claude-x")
    check(L["target"] == 128000 and L["source"] == "db_limit" and L["token"][1][0] == 105000,
          f"calib: db contextLimit drives levels for unknown families ({L['token']})")
    K.note_prompt(cal, "claude-x", 160000)
    L = K.levels_for(cal, "claude-x")
    check(L["target"] == 168000 and L["source"] == "db_limit+seen", "calib: larger prompt without compaction raises target")
    K.record_auto(cal, "claude-x", 200000, "conv-a", 9)
    K.record_auto(cal, "claude-x", 200000, "conv-a", 9)  # duplicate ignored
    L = K.levels_for(cal, "claude-x")
    check(L["target"] == 200000 and L["source"] == "observed" and len(cal["models"]["claude-x"]["samples"]) == 1,
          "calib: observed auto compaction wins (deduplicated)")
    K.note_prompt(cal, "claude-x", 230000)
    L = K.levels_for(cal, "claude-x")
    check(not cal["models"]["claude-x"]["samples"] and L["source"] == "db_limit+seen" and L["target"] == 241500,
          "calib: samples dropped once the window clearly grew")
    K.record_auto(cal, "claude-x", 100000, "conv-b", 3)
    L = K.levels_for(cal, "claude-x")
    check(L["target"] == 100000 and cal["models"]["claude-x"]["max_prompt"] == 100000,
          "calib: window shrink resets max_prompt so the new observation is used")
    cal = {"models": {"claude-x": {"db_limit": 128000, "override": 300000}}}
    check(K.levels_for(cal, "claude-x")["source"] == "override", "calib: manual override has top priority")

    # Hook integration: model-aware warnings + self-calibration + one /compact = one checkpoint.
    saved = K.load(guard.DATA_DIR)
    K.save(guard.DATA_DIR, {"models": {"claude-test": {"db_limit": 128000}}})
    dst, pl = _mini_conv("calib-hook", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>做事</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "model": "claude-test", "input_tokens": 1000}])
    call(dict(pl, hookEventName="PreInvocation"))
    _append(dst, {"step_index": 2, "type": "PLANNER_RESPONSE", "model": "claude-test", "input_tokens": 95000})
    m1 = text_of(call(dict(pl, hookEventName="PreInvocation")))
    check("Context Guard 预警" in m1 and "120k–131k" in m1 and "claude-test" in m1,
          f"hook: level-1 warning uses the model's own window ({m1[:80]!r})")
    _append(dst, {"step_index": 3, "type": "PLANNER_RESPONSE", "model": "claude-test", "input_tokens": 110000})
    m2 = text_of(call(dict(pl, hookEventName="PreInvocation")))
    check("Context Guard 紧急" in m2, "hook: level-2 warning for the small-window model")
    st = guard.load_state("calib-hook")
    check(st["levels"]["model"] == "claude-test" and st["levels"]["target"] == 128000,
          "hook: state carries model-specific levels for the meter")
    _append(dst, {"step_index": 4, "type": "CHECKPOINT", "content": "# Resuming"},
            {"step_index": 5, "type": "PLANNER_RESPONSE", "model": "claude-test", "input_tokens": 20000})
    call(dict(pl, hookEventName="PreInvocation"))
    e = K.load(guard.DATA_DIR)["models"]["claude-test"]
    check([s["tokens"] for s in e.get("samples") or []] == [110000], "hook: auto compaction recorded as calibration sample")
    check(guard.load_state("calib-hook")["levels"]["source"] == "observed", "hook: observed window used afterwards")
    _append(dst, {"step_index": 6, "type": "USER_INPUT", "content": "<USER_REQUEST>/compact</USER_REQUEST>"},
            {"step_index": 7, "type": "CHECKPOINT", "content": "# Resuming"})
    call(dict(pl, hookEventName="PreInvocation"))
    check(guard.load_state("calib-hook").get("ckpt_manual") is True, "hook: /compact checkpoint is manual")
    _append(dst, {"step_index": 8, "type": "PLANNER_RESPONSE", "model": "claude-test", "input_tokens": 105000},
            {"step_index": 9, "type": "CHECKPOINT", "content": "# Resuming"})
    call(dict(pl, hookEventName="PreInvocation"))
    e = K.load(guard.DATA_DIR)["models"]["claude-test"]
    check(guard.load_state("calib-hook").get("ckpt_manual") is False and len(e["samples"]) == 2,
          "hook: a later checkpoint without a new /compact counts as automatic")
    # Raising the target later clears stale warned/model_warned flags
    K.save(guard.DATA_DIR, {"models": {"claude-test": {"db_limit": 128000}}})
    dst2, pl2 = _mini_conv("warn-reset", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>hi</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "model": "claude-test", "input_tokens": 110000}])
    call(dict(pl2, hookEventName="PreInvocation"))
    check(guard.load_state("warn-reset").get("warned") == 2, "hook: level-2 set before threshold rise")
    K.save(guard.DATA_DIR, {"models": {"claude-test": {"override": 512000}}})
    _append(dst2, {"step_index": 2, "type": "PLANNER_RESPONSE", "model": "claude-test", "input_tokens": 112000})
    call(dict(pl2, hookEventName="PreInvocation"))
    st_after = guard.load_state("warn-reset")
    check(st_after.get("model_warned") == 0 and st_after.get("warned") == 0,
          f"hook: stale warning cleared when threshold rises ({st_after.get('warned')}/{st_after.get('model_warned')})")
    K.save(guard.DATA_DIR, saved)


def usage_tests():
    import usage as U
    skill = r"C:\Users\x\.gemini\config\skills\pdf\SKILL.md"
    objs = [
        {"step_index": 0, "type": "USER_INPUT", "content": "u" * 300},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 10000, "thinking": "t" * 60,
         "tool_calls": [{"name": "read_file", "args": {"Path": skill}},
                        {"name": "call_mcp_tool", "args": {"ServerName": "blender", "ToolName": "get_objects_summary"}}]},
        {"step_index": 2, "type": "GENERIC", "content": "S" * 3000},
        {"step_index": 3, "type": "GENERIC", "content": "M" * 1000},
        {"step_index": 4, "type": "SYSTEM_MESSAGE", "source": "SYSTEM",
         "content": "[Message] sender=abc/task-9 priority=LOW content=done"},
        {"step_index": 5, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 12000,
         "tool_calls": [{"name": "run_command", "args": {"CommandLine": r"python C:\a\skills\pdf\scripts\x.py"}}]},
        {"step_index": 6, "type": "GENERIC", "content": "R" * 500},
        {"step_index": 7, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 12500, "content": "done"},
        {"step_index": 8, "type": "CHECKPOINT", "content": "c" * 900},
        {"step_index": 9, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 3000},
    ]
    u = U.new_usage()
    for o in objs[:8]:
        U.feed(u, o)
    w = u["window"]["cats"]
    check(sum(e["tokens"] for e in w.values()) == 12500, f"usage: window tokens sum to the real prompt ({w})")
    check(w["tool:read_file"]["chars"] == 3000 and w["tool:call_mcp_tool"]["chars"] == 1000
          and w["tool:run_command"]["chars"] == 500, "usage: tool results mapped to their calls in order")
    check(w["tool:read_file"]["tokens"] > w["tool:call_mcp_tool"]["tokens"] > 0, "usage: growth split by chars")
    check(w["system:task_notice"]["n"] == 1 and "args:read_file" in w and w["model:thinking"]["chars"] == 60,
          "usage: notices / tool args / thinking classified")
    check(u["skills"]["pdf"]["loads"] == 1 and u["skills"]["pdf"]["resource_uses"] == 1,
          "usage: SKILL.md load and skill script use counted")
    check(u["mcp"] == {"blender/get_objects_summary": 1} and u["tools"]["run_command"]["calls"] == 1,
          "usage: MCP and tool call counts")
    check("工具结果 · read_file" in U.top_line(u), f"usage: top line names the biggest source ({U.top_line(u)})")
    for o in objs[8:]:
        U.feed(u, o)
    check(u["compactions"] == 1 and u["window"]["prompt"] == 3000 and "checkpoint" in u["window"]["cats"]
          and "tool:read_file" not in u["window"]["cats"] and "tool:read_file" in u["total"]["cats"],
          "usage: checkpoint resets the window, total keeps history")
    md = U.render_markdown(u)
    check("### 工具使用" in md and "### Skill 使用" in md and "blender/get_objects_summary" in md,
          "usage: markdown report has tool / skill / MCP sections")

    # hook integration: state carries usage; a pre-existing state without usage is backfilled
    dst, pl = _mini_conv("usage-hook", objs[:4])
    call(dict(pl, hookEventName="PreInvocation"))
    st = guard.load_state("usage-hook")
    st.pop("usage", None)
    guard.save_state("usage-hook", st)
    _append(dst, *objs[4:8])
    call(dict(pl, hookEventName="PreInvocation"))
    su = guard.load_state("usage-hook")["usage"]
    check(sum(e["tokens"] for e in su["window"]["cats"].values()) == 12500
          and su["tools"]["read_file"]["result_chars"] == 3000,
          "usage hook: backfilled once from the transcript, no double counting")
    r = subprocess.run([sys.executable, "-B", os.path.join(PLUGIN, "scripts", "context_stats.py"),
                        "--transcript", dst, "--json"], capture_output=True, text=True, encoding="utf-8")
    check(r.returncode == 0 and json.loads(r.stdout)["window"]["prompt"] == 12500, "usage: CLI --json works")


def deny_and_head_tests():
    base = [{"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>做事</USER_REQUEST>"},
            {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000}]
    for echoed in (True, False):
        conv = f"deny-{'echo' if echoed else 'lost'}"
        dst, pl = _mini_conv(conv, base)
        call(dict(pl, hookEventName="PreInvocation"))
        _append(dst, {"step_index": 2, "type": "CHECKPOINT", "content": "# Resuming"},
                {"step_index": 3, "type": "PLANNER_RESPONSE", "input_tokens": 5000})
        r = call(dict(pl, hookEventName="PreToolUse", stepIdx=3, toolCall={"name": "run_command"}))
        check(r.get("decision") == "deny", f"{conv}: first mutating tool after compaction denied")
        if echoed:
            _append(dst, {"step_index": 4, "type": "GENERIC", "content": r["reason"]})
        m = text_of(call(dict(pl, hookEventName="PreInvocation")))
        st = guard.load_state(conv)
        if echoed:
            check("恢复包已在上一轮" in m and st.get("last_delivery") == "ok",
                  "deny echoed: short follow-up note, delivery confirmed")
        else:
            check("Context Guard 重发" in m and "第 2 步发生了自动压缩" in m and st.get("verify"),
                  "deny not echoed: full pack re-sent via PreInvocation and verified")
    dst, pl = _mini_conv("head-cleanup", base + [{"step_index": i, "type": "GENERIC", "content": "x"}
                                                 for i in range(2, 9)])
    call(dict(pl, hookEventName="PreInvocation"))
    st = guard.load_state("head-cleanup")
    check("seen_head" not in st and st.get("worker_check_done") is True,
          "non-worker conversation drops cached head steps after step 5")


def nudge_tests():
    """After a recovery pack, remind once if task_state.md is still older than the pack."""
    for written in (False, True):
        conv = f"nudge-{'written' if written else 'stale'}"
        dst, pl = _mini_conv(conv, [{"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>做事</USER_REQUEST>"},
                                    {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000}])
        os.makedirs(pl["artifactDirectoryPath"], exist_ok=True)
        call(dict(pl, hookEventName="PreInvocation"))
        _append(dst, {"step_index": 2, "type": "CHECKPOINT", "content": "# Resuming"},
                {"step_index": 3, "type": "USER_INPUT", "content": "<USER_REQUEST>继续</USER_REQUEST>"})
        m = text_of(call(dict(pl, hookEventName="PreInvocation")))
        check("发生了自动压缩" in m, f"{conv}: pack injected")
        _append(dst, {"step_index": -1, "type": "SYSTEM_MESSAGE", "content": m})  # delivered
        if written:
            time.sleep(0.05)
            with open(os.path.join(pl["artifactDirectoryPath"], "task_state.md"), "w", encoding="utf-8") as f:
                f.write("# task_state\n")
        msgs = []
        for i in range(10):
            _append(dst, {"step_index": 4 + i, "type": "PLANNER_RESPONSE", "input_tokens": 2000})
            msgs.append(text_of(call(dict(pl, hookEventName="PreInvocation"))))
        hits = [i for i, x in enumerate(msgs) if "Context Guard 提醒" in x]
        if written:
            check(not hits, f"{conv}: no reminder when task_state.md was updated after the pack")
        else:
            check(len(hits) == 1 and hits[0] == guard.NUDGE_AFTER_RESPONSES - 1,
                  f"{conv}: exactly one reminder after {guard.NUDGE_AFTER_RESPONSES} model calls (got {hits})")


def regression_tests():
    import calib as K
    import common as C

    # 1. Delivery retry cap (MAX_DELIVERY_TRIES = 2) + real paths in notice/warning_text
    dst, pl = _mini_conv("retry-cap", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>开始</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000, "output_tokens": 100},
    ])
    call(dict(pl, hookEventName="PreInvocation"))  # first_run
    _append(dst, {"step_index": 2, "type": "PLANNER_RESPONSE", "input_tokens": 365000, "output_tokens": 100})
    w1 = text_of(call(dict(pl, hookEventName="PreInvocation")))
    check("retry-cap" in w1 and "<会话ID>" not in w1 and "context_stats.py" in w1,
          "warning_text includes real conversationId instead of <会话ID> placeholder")
    _append(dst, {"step_index": 3, "type": "CHECKPOINT", "content": "# Resuming"})
    m1 = text_of(call(dict(pl, hookEventName="PreInvocation")))
    check("## Context Guard：第 3 步发生了" in m1 and guard.load_state("retry-cap")["verify"]["tries"] == 1,
          "retry cap: try 1 injected")
    _append(dst, {"step_index": 4, "type": "PLANNER_RESPONSE", "input_tokens": 4000})
    m2 = text_of(call(dict(pl, hookEventName="PreInvocation")))
    check("## Context Guard：第 3 步发生了" in m2 and guard.load_state("retry-cap")["verify"]["tries"] == 2,
          "retry cap: try 2 re-sent")
    _append(dst, {"step_index": 5, "type": "PLANNER_RESPONSE", "input_tokens": 5000})
    m3 = text_of(call(dict(pl, hookEventName="PreInvocation")))
    st_cap = guard.load_state("retry-cap")
    check("多次注入都没有送达" in m3 and f'--transcript "{dst}"' in m3 and "<本会话" not in m3,
          "retry cap: stops retrying at MAX_DELIVERY_TRIES and delivers notice with real transcript path")
    check(not st_cap.get("pending") and not st_cap.get("verify") and not st_cap.get("verify_retry")
          and not st_cap.get("notice"),
          "retry cap: pending/verify/verify_retry/notice cleared after giving up")
    check(call(dict(pl, hookEventName="PreInvocation")) == {},
          "retry cap: subsequent PreInvocation does not loop")

    # 2. PreToolUse without top-level stepIdx on manual /compact
    dst2, pl2 = _mini_conv("pretool-nostep", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>做事</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000},
    ])
    call(dict(pl2, hookEventName="PreInvocation"))
    _append(dst2, {"step_index": 2, "type": "USER_INPUT", "content": "<USER_REQUEST>/compact</USER_REQUEST>"},
            {"step_index": 3, "type": "CHECKPOINT", "content": "# Resuming"},
            {"step_index": 4, "type": "PLANNER_RESPONSE", "input_tokens": 1000})
    r1 = call(dict(pl2, hookEventName="PreToolUse", toolCall={"name": "write_file"}))
    r2 = call(dict(pl2, hookEventName="PreToolUse", toolCall={"name": "edit_file", "stepIdx": None}))
    st2 = guard.load_state("pretool-nostep")
    check(r1.get("decision") == "deny" and "## Context Guard：第 3 步发生了手动压缩（/compact）" in r1.get("reason", ""),
          "pretool without stepIdx: 1st mutating tool sends full manual pack")
    check(st2.get("pretool_sent") is not None,
          "pretool without stepIdx: pretool_sent is non-None")
    check(r2.get("decision") == "deny" and "手动压缩 (/compact)" in r2.get("reason", "")
          and "自动压缩" not in r2.get("reason", "") and "## Context Guard：第 3 步" not in r2.get("reason", ""),
          "pretool without stepIdx: 2nd mutating tool returns short manual-compaction deny message")
    _append(dst2, {"step_index": 5, "type": "GENERIC", "content": r1["reason"]})
    m_after = text_of(call(dict(pl2, hookEventName="PreInvocation")))
    check("恢复包已在上一轮" in m_after and guard.load_state("pretool-nostep").get("last_delivery") == "ok",
          "pretool without stepIdx: echoed deny avoids duplicate full injection on next PreInvocation")

    # 3. Single-batch calibration ordering: prompt right before auto CHECKPOINT does not wipe samples
    saved_cal = K.load(guard.DATA_DIR)
    try:
        K.save(guard.DATA_DIR, {
            "models": {
                "cal-batch": {
                    "samples": [{"tokens": 100000, "conv": "old1", "step": 1},
                                {"tokens": 100000, "conv": "old2", "step": 2}],
                    "max_prompt": 250000,
                }
            }
        })
        dst3, pl3 = _mini_conv("cal-batch-conv", [
            {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>hi</USER_REQUEST>"},
            {"step_index": 1, "type": "PLANNER_RESPONSE", "model": "cal-batch", "input_tokens": 50000},
        ])
        call(dict(pl3, hookEventName="PreInvocation"))
        _append(dst3,
                {"step_index": 2, "type": "PLANNER_RESPONSE", "model": "cal-batch", "input_tokens": 120000},
                {"step_index": 3, "type": "CHECKPOINT", "content": "# Resuming"})
        call(dict(pl3, hookEventName="PreInvocation"))
        entry = K.load(guard.DATA_DIR)["models"]["cal-batch"]
        check([s["tokens"] for s in entry.get("samples") or []] == [100000, 100000, 120000]
              and entry.get("max_prompt") == 120000,
              f"calibration batch order preserves new sample and shrinks max_prompt ({entry})")
    finally:
        K.save(guard.DATA_DIR, saved_cal)

    # 4. Rewind / active branch filtering in guard, recall.py, and context_stats.py
    dst4, pl4 = _mini_conv("rewind-branch", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>keep-user</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 1000,
         "tool_calls": [{"name": "edit_file", "args": {"Path": "keep.py"}}]},
        {"step_index": 2, "type": "GENERIC", "content": "R" * 100},
        {"step_index": 3, "type": "USER_INPUT", "content": "<USER_REQUEST>discarded-user</USER_REQUEST>"},
        {"step_index": 4, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 5000,
         "tool_calls": [{"name": "exit_plan_mode", "args": {"plan": "discarded plan"}},
                        {"name": "edit_file", "args": {"Path": "discarded.py"}}]},
        {"step_index": 5, "type": "GENERIC", "content": "D" * 100},
        {"step_index": 3, "type": "USER_INPUT", "content": "<USER_REQUEST>active-user</USER_REQUEST>"},
        {"step_index": 4, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 2000,
         "tool_calls": [{"name": "exit_plan_mode", "args": {"plan": "active plan"}},
                        {"name": "edit_file", "args": {"Path": "active.py"}}]},
        {"step_index": 5, "type": "GENERIC", "content": "A" * 100},
        {"step_index": 6, "type": "CHECKPOINT", "content": "# Resuming"},
    ])
    pack4 = guard.build_recovery(dst4, pl4["artifactDirectoryPath"], 6, "rewind-branch")
    check("keep-user" in pack4 and "active-user" in pack4 and "active plan" in pack4
          and "active.py" in pack4 and "keep.py" in pack4,
          "rewind: build_recovery keeps active branch users, plan, and files")
    check("discarded-user" not in pack4 and "discarded plan" not in pack4 and "discarded.py" not in pack4,
          "rewind: build_recovery drops discarded branch users, plan, and files")

    ru_active = subprocess.run([sys.executable, RECALL, "--transcript", dst4, "users"],
                               capture_output=True, text=True, encoding="utf-8")
    ru_all = subprocess.run([sys.executable, RECALL, "--transcript", dst4, "users", "--all-branches"],
                            capture_output=True, text=True, encoding="utf-8")
    rf = subprocess.run([sys.executable, RECALL, "--transcript", dst4, "files"],
                        capture_output=True, text=True, encoding="utf-8")
    r_bad_rx = subprocess.run([sys.executable, RECALL, "--transcript", dst4, "grep", "["],
                              capture_output=True, text=True, encoding="utf-8")
    check(ru_active.returncode == 0 and "active-user" in ru_active.stdout and "discarded-user" not in ru_active.stdout,
          "recall users filters discarded branch by default")
    check(ru_all.returncode == 0 and "discarded-user" in ru_all.stdout and "active-user" in ru_all.stdout,
          "recall users --all-branches includes discarded branch")
    check(rf.returncode == 0 and "active.py" in rf.stdout and "discarded.py" not in rf.stdout,
          "recall files drops discarded branch")
    check(r_bad_rx.returncode == 2 and "正则表达式无效" in r_bad_rx.stderr,
          "recall grep handles invalid regex cleanly")

    flat_dir = os.path.join(TMP, "flat-recall")
    os.makedirs(flat_dir, exist_ok=True)
    flat_tr = os.path.join(flat_dir, "custom.jsonl")
    shutil.copy(dst4, flat_tr)
    r_flat = subprocess.run([sys.executable, RECALL, "--transcript", flat_tr, "--limit", "10", "users"],
                            capture_output=True, text=True, encoding="utf-8")
    check(r_flat.returncode == 0 and os.path.join(flat_dir, "recall_users_") in r_flat.stdout,
          "recall overflow on non-standard path writes beside transcript instead of climbing to /scratch")

    stats_script = os.path.join(PLUGIN, "scripts", "context_stats.py")
    r_stats = subprocess.run(
        [sys.executable, "-B", stats_script, "--transcript", dst4.replace("\\", "/"), "--conv", "rewind-branch", "--json"],
        capture_output=True, text=True, encoding="utf-8",
    )
    stats_data = json.loads(r_stats.stdout) if r_stats.returncode == 0 else {}
    check(r_stats.returncode == 0 and "trimmed" not in (stats_data.get("window") or {}).get("cats", {}),
          "context_stats drops discarded branch and accepts --conv / forward-slash path")

    # 5. AgentCollector 1:1 tool queue alignment & lowercase/snake_case arg fallbacks
    col = C.AgentCollector()
    for o in [
        {"step_index": 1, "type": "PLANNER_RESPONSE", "tool_calls": [
            {"name": "read_file", "args": {"Path": "foo.txt"}},
            {"name": "invoke_subagent", "args": {"role": "Helper", "agent_config_name": "research", "prompt": "check x"}},
            {"name": "send_message", "args": {"recipient": "11111111-1111-1111-1111-111111111111", "message": "hi"}},
        ]},
        {"step_index": 2, "type": "GENERIC", "content": '{"conversationId": "99999999-9999-9999-9999-999999999999"}'},
        {"step_index": 3, "type": "GENERIC", "content": 'Conversation ID: {"conversationId": "22222222-2222-2222-2222-222222222222"}'},
    ]:
        col.feed(o)
    check(len(col.dispatched) == 1
          and col.dispatched[0]["conv"] == "22222222-2222-2222-2222-222222222222"
          and col.dispatched[0]["label"] == "Helper"
          and col.dispatched[0]["type"] == "research"
          and col.dispatched[0]["text"] == "check x",
          "AgentCollector: 1:1 queue prevents read_file output from stealing subagent conv id + supports lowercase args")
    check(col.outgoing == [{"step": 1, "to": "11111111-1111-1111-1111-111111111111", "text": "hi"}],
          "AgentCollector: send_message supports lowercase recipient/message fallbacks")

    # 6. Synthetic plan & structure_ok + plan file persistence
    dst5, pl5 = _mini_conv("plan-structure", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST># User Heading\n### Fake Section</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "tool_calls": [
            {"name": "exit_plan_mode", "args": {"plan": "# Old Plan\nold body"}},
        ]},
        {"step_index": 2, "type": "GENERIC", "content": "ok"},
        {"step_index": 3, "type": "PLANNER_RESPONSE", "tool_calls": [
            {"name": "exit_plan_mode", "args": {"plan": "# Plan Title\n### Step 1\nFinal plan body"}},
        ]},
        {"step_index": 4, "type": "GENERIC", "content": "ok"},
        {"step_index": 5, "type": "PLANNER_RESPONSE", "tool_calls": [
            {"name": "enter_plan_mode", "args": {"reason": "Handing back for review"}},
        ]},
        {"step_index": 6, "type": "GENERIC", "content": "ok"},
        {"step_index": 7, "type": "CHECKPOINT", "content": "# Resuming"},
    ])
    os.makedirs(pl5["artifactDirectoryPath"], exist_ok=True)
    with open(os.path.join(pl5["artifactDirectoryPath"], "task_state.md"), "w", encoding="utf-8") as f:
        f.write("## Inner Heading\nkept\n")
    pack5 = guard.build_recovery(dst5, pl5["artifactDirectoryPath"], 7, "plan-structure")
    ok5, bad5 = structure_ok(pack5)
    check(ok5, f"synthetic pack structure intact (offending line: {bad5!r})")
    check("### 当前执行计划（第 3 步" in pack5 and "Final plan body" in pack5 and "### 最近一次交回" in pack5,
          "synthetic pack contains latest plan and enter_plan_mode report")
    plan_file = os.path.join(guard.DATA_DIR, "plans", "plan-structure.md")
    check(os.path.isfile(plan_file) and open(plan_file, encoding="utf-8").read() == "# Plan Title\n### Step 1\nFinal plan body",
          "full plan saved to file")

    # 7. Synthetic recall step & grep checks (deterministic without local brain)
    r_step = subprocess.run([sys.executable, RECALL, "--transcript", dst4, "step", "0"],
                            capture_output=True, text=True, encoding="utf-8")
    r_grep = subprocess.run([sys.executable, RECALL, "--transcript", dst4, "grep", "active-user"],
                            capture_output=True, text=True, encoding="utf-8")
    check(r_step.returncode == 0 and "keep-user" in r_step.stdout, "recall step works on synthetic transcript")
    check(r_grep.returncode == 0 and "active-user" in r_grep.stdout, "recall grep works on synthetic transcript")

    # 8. guard.log rotation at > 2 MiB
    log_p = os.path.join(guard.DATA_DIR, "guard.log")
    rot_p = log_p + ".1"
    for p in (log_p, rot_p):
        if os.path.exists(p):
            os.remove(p)
    os.makedirs(guard.DATA_DIR, exist_ok=True)
    with open(log_p, "wb") as f:
        f.write(b"x" * (guard.LOG_MAX_BYTES + 100))
    guard.log("after-rotate")
    check(os.path.isfile(rot_p) and os.path.getsize(rot_p) > guard.LOG_MAX_BYTES,
          "log rotation: oversized guard.log rotated to guard.log.1")
    check(os.path.isfile(log_p) and os.path.getsize(log_p) < 1024
          and "after-rotate" in open(log_p, encoding="utf-8").read(),
          "log rotation: new guard.log contains fresh entry")

    # 9. conversationId path-traversal validation
    state_dir = os.path.join(guard.DATA_DIR, "state")
    before_states = set(os.listdir(state_dir)) if os.path.isdir(state_dir) else set()
    for bad_conv in ("", ".", "..", "../escape", "a/b", "a\\b", "bad id", 123):
        r_inv = call({"conversationId": bad_conv, "transcriptPath": dst4, "hookEventName": "PreInvocation"})
        r_tool = call({"conversationId": bad_conv, "transcriptPath": dst4, "hookEventName": "PreToolUse",
                       "toolCall": {"name": "edit_file"}})
        check(r_inv == {} and r_tool == {"decision": "ask"},
              f"invalid conversationId {bad_conv!r} safely ignored")
    after_states = set(os.listdir(state_dir)) if os.path.isdir(state_dir) else set()
    check(before_states == after_states, "invalid conversationId creates no state or lock file")

    # 10. update_usage() rebuilds on cross-batch and intra-batch rewind
    import usage as U
    dst_rw, pl_rw = _mini_conv("usage-rewind", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>start</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 10000},
        {"step_index": 2, "type": "USER_INPUT", "content": "<USER_REQUEST>branch-a</USER_REQUEST>"},
        {"step_index": 3, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 20000,
         "tool_calls": [{"name": "read_file", "args": {"Path": "big.txt"}}]},
        {"step_index": 4, "type": "GENERIC", "content": "X" * 50000},
        {"step_index": 5, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 35000},
    ])
    call(dict(pl_rw, hookEventName="PreInvocation"))
    st_rw1 = guard.load_state("usage-rewind")
    check("tool:read_file" in st_rw1["usage"]["window"]["cats"] and "read_file" in U.top_line(st_rw1["usage"]),
          "usage rewind setup: initial branch records read_file")
    _append(dst_rw,
            {"step_index": 2, "type": "USER_INPUT", "content": "<USER_REQUEST>branch-b</USER_REQUEST>"},
            {"step_index": 3, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 12000})
    call(dict(pl_rw, hookEventName="PreInvocation"))
    st_rw2 = guard.load_state("usage-rewind")
    cats_rw2 = st_rw2["usage"]["window"]["cats"]
    check("tool:read_file" not in cats_rw2 and "trimmed" not in cats_rw2
          and "read_file" not in U.top_line(st_rw2["usage"])
          and st_rw2["usage"]["window"]["prompt"] == 12000,
          f"usage cross-batch rewind drops discarded branch tools and trimmed ({cats_rw2})")

    # Intra-batch rewind (both discarded and retried branches arrive in the same batch)
    dst_irw, pl_irw = _mini_conv("usage-intrabatch-rewind", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>start</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 10000},
    ])
    call(dict(pl_irw, hookEventName="PreInvocation"))
    _append(dst_irw,
            {"step_index": 2, "type": "USER_INPUT", "content": "<USER_REQUEST>discard</USER_REQUEST>"},
            {"step_index": 3, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 20000,
             "tool_calls": [{"name": "read_file", "args": {"Path": "discard.txt"}}]},
            {"step_index": 4, "type": "GENERIC", "content": "Z" * 40000},
            {"step_index": 2, "type": "USER_INPUT", "content": "<USER_REQUEST>keep</USER_REQUEST>"},
            {"step_index": 3, "type": "PLANNER_RESPONSE", "model": "m", "input_tokens": 11000})
    call(dict(pl_irw, hookEventName="PreInvocation"))
    st_irw = guard.load_state("usage-intrabatch-rewind")
    check("tool:read_file" not in st_irw["usage"]["window"]["cats"]
          and st_irw["usage"]["window"]["prompt"] == 11000,
          "usage intra-batch rewind rebuilds cleanly without discarded branch")

    # 11. Light iter_active_steps drops huge non-agent GENERIC content & matches full build_recovery
    orch_fx = os.path.join(FIX, "orchestrator.jsonl")
    light_orch = C.iter_active_steps(orch_fx, light=True)
    full_orch = C.iter_active_steps(orch_fx, light=False)
    check(any(o.get("type") == "GENERIC" and "content" not in o for o in light_orch)
          and any(o.get("type") == "GENERIC" and "content" in o for o in light_orch),
          "light iter_active_steps drops ordinary GENERIC content while keeping agent/board GENERIC")
    # Compare build_recovery with light=True vs light=False
    orig_iter = C.iter_active_steps
    try:
        pack_light = guard.build_recovery(orch_fx, "", 999, "orch-light")
        C.iter_active_steps = lambda p, **kw: orig_iter(p, light=False)
        pack_full = guard.build_recovery(orch_fx, "", 999, "orch-full")
        check(pack_light == pack_full, "light build_recovery matches full build_recovery on orchestrator.jsonl")
    finally:
        C.iter_active_steps = orig_iter

    dst_big, pl_big = _mini_conv("light-big", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>task</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "tool_calls": [
            {"name": "read_file", "args": {"Path": "a.py", "big": "B" * 1000}},
            {"name": "invoke_subagent", "args": {"Role": "Scout", "TypeName": "research", "Prompt": "go"}},
            {"name": "edit_file", "args": {"Path": "edited.py", "Replacement": "X" * 2000}},
            {"name": "exit_plan_mode", "args": {"plan": "my plan"}},
        ]},
        {"step_index": 2, "type": "GENERIC", "content": "HUGE" * 80000},
        {"step_index": 3, "type": "GENERIC", "content": '{"conversationId": "33333333-3333-3333-3333-333333333333"}'},
        {"step_index": 4, "type": "GENERIC", "content": "ok"},
        {"step_index": 5, "type": "GENERIC", "content": "ok"},
        {"step_index": 6, "type": "CHECKPOINT", "content": "# Resuming"},
    ])
    light_big = C.iter_active_steps(dst_big, light=True)
    check("content" not in light_big[2] and "33333333-3333-3333-3333-333333333333" in (light_big[3].get("content") or ""),
          "light iter_active_steps drops 320KB read_file GENERIC and keeps invoke_subagent GENERIC")
    try:
        p_big_light = guard.build_recovery(dst_big, "", 6, "big-conv")
        C.iter_active_steps = lambda p, **kw: orig_iter(p, light=False)
        p_big_full = guard.build_recovery(dst_big, "", 6, "big-conv")
        check(p_big_light == p_big_full, "light build_recovery matches full build_recovery on synthetic large transcript")
    finally:
        C.iter_active_steps = orig_iter

    # 12. Python/JS alignment: call_mcp_tool fallback, compact JSON separators, compact_step equivalence
    u_mcp = U.new_usage()
    U.feed(u_mcp, {
        "step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000,
        "tool_calls": [
            {"name": "call_mcp_tool", "args": {}},
            {"name": "call_mcp_tool", "args": {"ServerName": "srv"}},
            {"name": "call_mcp_tool", "args": {"ToolName": "fn"}},
        ],
    })
    check(u_mcp["mcp"] == {"unknown/unknown": 1, "srv/unknown": 1, "unknown/fn": 1},
          f"usage: call_mcp_tool missing fields fall back to unknown ({u_mcp['mcp']})")

    tc_sample = {"name": "edit_file", "args": {"Path": "a.py", "Target": "x"}}
    step_sample = {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000, "tool_calls": [tc_sample]}
    expected_arg_chars = len(json.dumps(tc_sample["args"], ensure_ascii=False, separators=(",", ":")))
    expected_tc_chars = len(json.dumps(tc_sample, ensure_ascii=False, separators=(",", ":")))
    u_sep = U.new_usage()
    U.feed(u_sep, step_sample)
    check(C.step_chars(step_sample) == expected_tc_chars + 2,  # '[' + ']' around single element
          "step_chars uses compact JSON separators matching JS JSON.stringify")
    check(u_sep["tools"]["edit_file"]["arg_chars"] == expected_arg_chars
          and u_sep["pending"][0][1] == expected_tc_chars,
          "usage: arg_chars and pending args char count use compact JSON separators")

    raw_steps = [
        {"step_index": 0, "type": "USER_INPUT", "content": "hello world"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 5000, "thinking": "hmm", "content": "ans",
         "tool_calls": [{"name": "call_mcp_tool", "args": {"ServerName": "s", "ToolName": "t"}}]},
        {"step_index": 2, "type": "GENERIC", "content": "R" * 1234},
        {"step_index": 3, "type": "SYSTEM_MESSAGE", "source": "SYSTEM_SDK", "content": "hook msg"},
        {"step_index": 4, "type": "ERROR_MESSAGE", "error": "boom"},
        {"step_index": 5, "type": "PLANNER_RESPONSE", "input_tokens": 7000},
    ]
    u_raw, u_cmp = U.new_usage(), U.new_usage()
    for row in raw_steps:
        U.feed(u_raw, row)
        U.feed(u_cmp, U.compact_step(row))
    check(u_raw == u_cmp, "usage: compact_step produces identical accounting to raw steps")

    # 13. Item 1: Coordinator board.jsonl, Worker steps/<n>/output.txt + traversal guard, and 42k global cap
    c_root = os.path.join(TMP, "board-coord")
    c_logs = os.path.join(c_root, ".system_generated", "logs")
    c_swarm = os.path.join(c_root, ".system_generated", "swarm")
    os.makedirs(c_logs, exist_ok=True)
    os.makedirs(c_swarm, exist_ok=True)
    c_tr = os.path.join(c_logs, "transcript_full.jsonl")
    with open(c_tr, "w", encoding="utf-8") as f:
        for row in [
            {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>协调任务</USER_REQUEST>"},
            {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 2000,
             "tool_calls": [{"name": "read_board", "args": {"since": 0}}]},
            {"step_index": 2, "type": "GENERIC",
             "content": "The tool's output was truncated due to being too long. file:///C:/nope/output.txt\n{\"messages\":[]"},
            {"step_index": 3, "type": "CHECKPOINT", "content": "# Resuming"},
        ]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(c_swarm, "board.jsonl"), "w", encoding="utf-8") as f:
        for seq in range(1, 21):
            txt = (f"MSG-{seq:02d}-\n# Fake Heading\n### Fake Section\n" + "Z" * 40
                   if seq == 20 else f"MSG-{seq:02d}-" + "Z" * 80)
            f.write(json.dumps({"seq": seq, "agent": f"w-{seq % 3}", "text": txt},
                               ensure_ascii=False) + "\n")
    pack_coord = guard.build_recovery(c_tr, c_root, 3, "board-coord")
    check("MSG-20-" in pack_coord and "MSG-06-" in pack_coord and "MSG-05-" not in pack_coord
          and "> # Fake Heading" in pack_coord and "> ### Fake Section" in pack_coord
          and structure_ok(pack_coord)[0],
          "swarm: coordinator reads full board.jsonl and keeps latest 15 messages")
    r_msg_c = subprocess.run([sys.executable, RECALL, "--transcript", c_tr, "messages"],
                             capture_output=True, text=True, encoding="utf-8")
    check(r_msg_c.returncode == 0 and "#20" in r_msg_c.stdout and "MSG-20-" in r_msg_c.stdout,
          "recall messages: coordinator reads full board.jsonl")

    w_root = os.path.join(TMP, "board-worker")
    w_logs = os.path.join(w_root, ".system_generated", "logs")
    w_step5 = os.path.join(w_root, ".system_generated", "steps", "5")
    w_outside = os.path.join(w_root, "outside")
    os.makedirs(w_logs, exist_ok=True)
    os.makedirs(w_step5, exist_ok=True)
    os.makedirs(w_outside, exist_ok=True)
    w_out5 = os.path.join(w_step5, "output.txt")
    with open(w_out5, "w", encoding="utf-8") as f:
        f.write("Created At: 2026-11-28T00:00:00+08:00\nCompleted At: 2026-11-28T00:00:01+08:00\n")
        f.write(json.dumps({"messages": [
            {"seq": 1, "agent": "coordinator", "text": "ASSIGN-1"},
            {"seq": 2, "agent": "worker", "text": "WORKER-OUT-2"},
        ]}, ensure_ascii=False) + "\n")
    w_bad_out = os.path.join(w_outside, "output.txt")
    with open(w_bad_out, "w", encoding="utf-8") as f:
        f.write("Created At: x\nCompleted At: y\n" + json.dumps({"messages": [{"seq": 99, "agent": "evil", "text": "OUTSIDE-SECRET"}]}))
    w_missing = os.path.join(w_root, ".system_generated", "steps", "999", "output.txt")
    to_uri = lambda p: "file:///" + p.replace("\\", "/").lstrip("/")  # noqa: E731
    w_tr = os.path.join(w_logs, "transcript_full.jsonl")
    with open(w_tr, "w", encoding="utf-8") as f:
        for row in [
            {"step_index": 0, "type": "SYSTEM_MESSAGE",
             "content": "[Message] timestamp=2026-11-28T00:00:00Z sender=11111111-1111-1111-1111-111111111111 "
                        "priority=MESSAGE_PRIORITY_HIGH content=Your assignment (recorded on the board as #1)"},
            {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000,
             "tool_calls": [{"name": "read_board", "args": {"since": 0}},
                            {"name": "read_board", "args": {"since": 0}},
                            {"name": "read_board", "args": {"since": 0}}]},
            {"step_index": 2, "type": "GENERIC",
             "content": f"The tool's output was truncated. See {to_uri(w_out5)}\n" + '{"messages":[{"seq":1}'},
            {"step_index": 3, "type": "GENERIC",
             "content": f"The tool's output was truncated. See {to_uri(w_bad_out)}\n"},
            {"step_index": 4, "type": "GENERIC",
             "content": f"The tool's output was truncated. See {to_uri(w_missing)}\n"},
            {"step_index": 5, "type": "CHECKPOINT", "content": "# Resuming"},
        ]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pack_w = guard.build_recovery(w_tr, w_root, 5, "board-worker")
    check("WORKER-OUT-2" in pack_w and "OUTSIDE-SECRET" not in pack_w and structure_ok(pack_w)[0],
          "swarm: worker recovers truncated steps/<n>/output.txt and blocks paths outside steps/")

    cap_root = os.path.join(TMP, "cap-42k")
    cap_logs = os.path.join(cap_root, ".system_generated", "logs")
    cap_swarm = os.path.join(cap_root, ".system_generated", "swarm")
    os.makedirs(cap_logs, exist_ok=True)
    os.makedirs(cap_swarm, exist_ok=True)
    cap_tr = os.path.join(cap_logs, "transcript_full.jsonl")
    with open(os.path.join(cap_root, "task_state.md"), "w", encoding="utf-8") as f:
        f.write("# task_state\nTS-KEEP-MARKER\n" + ("S" * 7400))
    with open(os.path.join(cap_swarm, "board.jsonl"), "w", encoding="utf-8") as f:
        for seq in range(1, 16):
            f.write(json.dumps({"seq": seq, "agent": f"w{seq}", "text": "B" * 1000}) + "\n")
    with open(cap_tr, "w", encoding="utf-8") as f:
        s_idx = 0
        for i in range(45):
            f.write(json.dumps({"step_index": s_idx, "type": "USER_INPUT",
                                "content": f"<USER_REQUEST>early-user-{i} " + ("U" * 800) + "</USER_REQUEST>"}) + "\n")
            s_idx += 1
        for tag in ("TAIL-USER-A", "TAIL-USER-B", "TAIL-USER-C"):
            f.write(json.dumps({"step_index": s_idx, "type": "USER_INPUT",
                                "content": f"<USER_REQUEST>{tag} " + ("T" * 2600) + "</USER_REQUEST>"}) + "\n")
            s_idx += 1
        for i in range(12):
            f.write(json.dumps({"step_index": s_idx, "type": "SYSTEM_MESSAGE",
                                "content": f"[Message] timestamp=2026-11-28T00:00:00Z sender=22222222-2222-2222-2222-222222222222 "
                                           f"priority=HIGH content=IN-{i}-" + ("I" * 1000)}) + "\n")
            s_idx += 1
        tcs = [{"name": "edit_file", "args": {"Path": f"src/file_{i}.py"}} for i in range(15)]
        tcs += [{"name": "invoke_subagent", "args": {"Role": f"Sub{i}", "TypeName": "research", "Prompt": "P" * 300}} for i in range(12)]
        tcs += [{"name": "send_message", "args": {"Recipient": "22222222-2222-2222-2222-222222222222", "Message": "O" * 300}} for i in range(6)]
        tcs += [{"name": "exit_plan_mode", "args": {"plan": "P" * 4000}},
                {"name": "enter_plan_mode", "args": {"reason": "R" * 2000}}]
        f.write(json.dumps({"step_index": s_idx, "type": "PLANNER_RESPONSE", "tool_calls": tcs}) + "\n")
        s_idx += 1
        f.write(json.dumps({"step_index": s_idx, "type": "CHECKPOINT", "content": "# Resuming"}) + "\n")
    pack_cap = guard.build_recovery(cap_tr, cap_root, s_idx, "cap-42k")
    check(len(pack_cap) <= guard.PACK_MAX_CHARS and structure_ok(pack_cap)[0]
          and all(tag in pack_cap for tag in ("TAIL-USER-A", "TAIL-USER-B", "TAIL-USER-C", "TS-KEEP-MARKER")),
          f"global cap: pack len {len(pack_cap)} <= {guard.PACK_MAX_CHARS} while preserving last 3 users & task_state.md")

    # 14. Item 2: task_state.md staleness metadata & Worker rule exclusivity
    ts_root = os.path.join(TMP, "ts-stale")
    ts_logs = os.path.join(ts_root, ".system_generated", "logs")
    os.makedirs(ts_logs, exist_ok=True)
    ts_file = os.path.join(ts_root, "task_state.md")
    with open(ts_file, "w", encoding="utf-8") as f:
        f.write("# task_state\n当前目标：测试\n")
    t_old = 1764288000  # fixed past epoch
    os.utime(ts_file, (t_old, t_old))
    ts_tr_fresh = os.path.join(ts_logs, "transcript_full.jsonl")
    with open(ts_tr_fresh, "w", encoding="utf-8") as f:
        for row in [
            {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>hi</USER_REQUEST>"},
            {"step_index": 10, "type": "PLANNER_RESPONSE", "created_at": "2025-11-28T00:00:00Z",
             "tool_calls": [{"name": "write_file", "args": {"TargetFile": ts_file}}]},
            {"step_index": 12, "type": "PLANNER_RESPONSE", "created_at": "2025-11-28T00:01:00Z"},
            {"step_index": 15, "type": "CHECKPOINT", "content": "# Resuming"},
        ]:
            f.write(json.dumps(row) + "\n")
    pack_fresh = guard.build_recovery(ts_tr_fresh, ts_root, 15, "ts-fresh")
    check("上次更新于第 10 步，滞后 5 步 / 1 次模型调用" in pack_fresh
          and "可能已过时" not in pack_fresh and structure_ok(pack_fresh)[0],
          "task_state staleness: fresh file annotates step/lag with no stale warning")

    with open(ts_tr_fresh, "w", encoding="utf-8") as f:
        rows_stale = [
            {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>hi</USER_REQUEST>"},
            {"step_index": 2, "type": "PLANNER_RESPONSE", "created_at": "2025-11-28T00:00:00Z",
             "tool_calls": [{"name": "write_file", "args": {"TargetFile": ts_file}}]},
        ]
        for s_i in range(3, 9):
            rows_stale.append({"step_index": s_i, "type": "PLANNER_RESPONSE", "created_at": "2025-11-28T00:01:00Z"})
        rows_stale.append({"step_index": 30, "type": "CHECKPOINT", "content": "# Resuming"})
        for row in rows_stale:
            f.write(json.dumps(row) + "\n")
    pack_stale = guard.build_recovery(ts_tr_fresh, ts_root, 30, "ts-stale")
    check("上次更新于第 2 步，滞后 28 步 / 6 次模型调用" in pack_stale
          and "可能已过时" in pack_stale and structure_ok(pack_stale)[0],
          "task_state staleness: stale file emits strong lag warning")

    t_newer = 1764288000 + 600
    os.utime(ts_file, (t_newer, t_newer))
    pack_ext = guard.build_recovery(ts_tr_fresh, ts_root, 30, "ts-ext")
    check("可能已由命令或外部更新" in pack_ext and "可能已过时" not in pack_ext and structure_ok(pack_ext)[0],
          "task_state staleness: newer disk mtime avoids false stale warning")

    sw_excl_d = os.path.join(TMP, "swarm-worker-excl", ".system_generated", "logs")
    os.makedirs(sw_excl_d, exist_ok=True)
    shutil.copy(os.path.join(FIX, "swarm_worker.jsonl"), os.path.join(sw_excl_d, "transcript_full.jsonl"))
    worker_rule_msg = text_of(call({
        "conversationId": "swarm-worker-excl",
        "transcriptPath": os.path.join(sw_excl_d, "transcript.jsonl"),
        "artifactDirectoryPath": os.path.join(TMP, "swarm-worker-excl"),
        "hookEventName": "PreInvocation",
    }))
    check("严禁改写 Coordinator 或其他会话的 task_state.md" in worker_rule_msg,
          "swarm worker: rule footer forbids overwriting Coordinator task_state.md")

    # 15. Item 4: Conditional monotonic warning lock (Plan<->Execute switch, legacy state, rewind)
    saved_cal = K.load(guard.DATA_DIR)
    K.save(guard.DATA_DIR, {"models": {
        "neon-plan": {"samples": [{"tokens": 517000, "conv": "sw", "step": 100}]},
        "hydro-exec": {"override": 512000},
    }})
    dst_sw, pl_sw = _mini_conv("warn-switch", [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>plan</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "model": "neon-plan", "input_tokens": 200000},
    ])
    call(dict(pl_sw, hookEventName="PreInvocation"))
    _append(dst_sw, {"step_index": 2, "type": "PLANNER_RESPONSE", "model": "neon-plan", "input_tokens": 448000})
    w_sw1 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    _append(dst_sw, {"step_index": 3, "type": "PLANNER_RESPONSE", "model": "hydro-exec", "input_tokens": 408000})
    w_sw2 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    _append(dst_sw, {"step_index": 4, "type": "PLANNER_RESPONSE", "model": "hydro-exec", "input_tokens": 425000})
    w_sw3 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    _append(dst_sw, {"step_index": 5, "type": "PLANNER_RESPONSE", "model": "neon-plan", "input_tokens": 431000})
    w_sw4 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    check("Context Guard 紧急" in w_sw1 and not w_sw2 and not w_sw3 and not w_sw4,
          "warn lock: Plan<->Execute model switch does not re-trigger L2 warning")

    # Legacy state without warned_target
    st_leg = guard.load_state("warn-switch")
    st_leg.pop("warned_target", None)
    guard.save_state("warn-switch", st_leg)
    _append(dst_sw, {"step_index": 6, "type": "PLANNER_RESPONSE", "model": "hydro-exec", "input_tokens": 405000})
    w_leg1 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    _append(dst_sw, {"step_index": 7, "type": "PLANNER_RESPONSE", "model": "hydro-exec", "input_tokens": 425000})
    w_leg2 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    check(not w_leg1 and not w_leg2 and guard.load_state("warn-switch").get("warned_target") is not None,
          "warn lock: legacy state without warned_target also avoids duplicate L2 and backfills warned_target")

    # Rewind unlocks warning when new post-rewind prompt is smaller, and warns again once threshold crossed
    _append(dst_sw, {"step_index": 1, "type": "USER_INPUT", "content": "<USER_REQUEST>edited prompt</USER_REQUEST>"})
    w_rw0 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    _append(dst_sw, {"step_index": 2, "type": "PLANNER_RESPONSE", "model": "hydro-exec", "input_tokens": 80000})
    w_rw1 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    st_rw = guard.load_state("warn-switch")
    _append(dst_sw, {"step_index": 3, "type": "PLANNER_RESPONSE", "model": "hydro-exec", "input_tokens": 360000})
    w_rw2 = text_of(call(dict(pl_sw, hookEventName="PreInvocation")))
    check(not w_rw0 and not w_rw1 and st_rw.get("model_warned") == 0 and "Context Guard 预警" in w_rw2,
          "warn lock: rewind resets warning level after smaller prompt and allows re-warning later")
    K.save(guard.DATA_DIR, saved_cal)

    # 16. Item 5: CLI UTF-8 stderr/stdout under non-UTF8 console encoding (no PYTHONUTF8=1 injection)
    r_miss_cs = subprocess.run([sys.executable, "-B", stats_script, "--transcript", os.path.join(TMP, "missing.jsonl")],
                               capture_output=True)
    check(r_miss_cs.returncode != 0 and "找不到".encode("utf-8") in r_miss_cs.stderr,
          "context_stats: missing transcript emits valid UTF-8 stderr without encoding crash")
    cn_dir = os.path.join(TMP, "中文会话", ".system_generated", "logs")
    os.makedirs(cn_dir, exist_ok=True)
    cn_tr = os.path.join(cn_dir, "transcript_full.jsonl")
    with open(cn_tr, "w", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>测试中文</USER_REQUEST>"}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 1000,
                            "tool_calls": [{"name": "read_file", "args": {"Path": "中文.py"}}]}, ensure_ascii=False) + "\n")
    r_json_cs = subprocess.run([sys.executable, "-B", stats_script, "--transcript", cn_tr, "--json"],
                               capture_output=True)
    check(r_json_cs.returncode == 0 and json.loads(r_json_cs.stdout.decode("utf-8")).get("window", {}).get("prompt") == 1000,
          "context_stats: --json handles non-ASCII paths cleanly on stdout")


def main():
    replay_local = "--replay-local" in sys.argv[1:]
    ckpts = {}
    all_times = []
    if replay_local:
        for p in glob.glob(os.path.join(BRAIN, "*", ".system_generated", "logs", "transcript_full.jsonl")):
            conv = p.split(os.sep)[-4]
            if conv.startswith(("replay-", "fx")):
                continue
            trailing = None
            for line in open(p, encoding="utf-8", errors="replace"):
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("type") == "CHECKPOINT":
                    trailing = int(o["step_index"])
                elif o.get("type") == "PLANNER_RESPONSE" and trailing is not None:
                    # only compactions followed by a model call can receive a pack
                    ckpts.setdefault(conv, []).append(trailing)
                    trailing = None
        if ckpts:
            print(f"PASS found historical compactions in {len(ckpts)} conversations")
        else:
            print("SKIP no local conversation with a compaction yet: real-transcript replay skipped")

        for conv, steps in ckpts.items():
            ev, times = replay(conv)
            all_times += times
            delivered = sorted(ev["pack_inject"] + ev["pack_deny"])
            print(f"  {conv[:8]} ckpts={steps} deny={ev['pack_deny']} inject={ev['pack_inject']} "
                  f"note={ev['note']} w1={len(ev['w1'])} w2={len(ev['w2'])} max_pack={max(ev['sizes'] or [0])}")
            check(len(delivered) == len(steps), f"{conv[:8]} pack delivered exactly once per compaction")
            check(not ev["leak"], f"{conv[:8]} no mutating tool ran before the pack (leaks={ev['leak'][:3]})")
            check(len(ev["note"]) == len(ev["pack_deny"]), f"{conv[:8]} follow-up note after each deny-delivery")
            check(all(sz <= 45000 for sz in ev["sizes"]), f"{conv[:8]} pack size <= 45000")

        all_times.sort()
        p95 = all_times[int(len(all_times) * 0.95)] if all_times else 0
        check(p95 < 0.2, f"in-process p95 PreInvocation latency {p95*1000:.1f}ms < 200ms")

    env = dict(os.environ)
    for name, data, want in [
        ("bad json (invocation)", b"{not json", b"{}"),
        ("bad json (tool)", b'{"hookEventName":"PreToolUse", bad', b'{"decision": "ask"}'),
        ("missing transcript (tool)", json.dumps({"hookEventName": "PreToolUse", "conversationId": "x",
                                                  "transcriptPath": "Z:/nope/t.jsonl"}).encode(), b'{"decision": "ask"}'),
        ("missing transcript (invocation)", json.dumps({"conversationId": "x", "transcriptPath": "Z:/nope/t.jsonl"}).encode(), b"{}"),
    ]:
        r = subprocess.run([sys.executable, GUARD], input=data, capture_output=True, env=env)
        check(r.returncode == 0 and r.stdout.strip() == want, f"subprocess {name} -> {want.decode()} ({r.stdout[:60]!r})")

    if replay_local:
        conv = next(iter(ckpts), None)
        for args, needle in ([] if conv is None else [(["users"], b"1. [step"), (["step", "1"], b"step 1"),
                             (["grep", "Resuming from a compaction", "--max", "2"], b"CHECKPOINT"), (["files"], b"")]):
            r = subprocess.run([sys.executable, RECALL, "--conv", conv] + args, capture_output=True)
            check(r.returncode == 0 and needle in r.stdout, f"recall {args[0]} works")

    multi_agent_tests()
    sidecar_linkage_tests()
    manual_and_stale_tests()
    calibration_tests()
    usage_tests()
    deny_and_head_tests()
    nudge_tests()
    regression_tests()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAIL else str(len(FAIL)) + ' FAILED'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
