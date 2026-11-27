"""Model-aware compaction thresholds with self-calibration (context-guard).

Antigravity compacts at a model-dependent prompt size (Gemini: ~512k since 2026-11-16,
~256k before; Claude: unknown, the conversation DB reports contextLimit=128000).
Fixed thresholds therefore never fire for smaller-window models. Resolution order for
the compaction target of a model:

  override (hand-set in calibration.json)
  > observed (median of the last auto-compaction prompt sizes for this model)
  > builtin (known families)
  > db_limit (gen_metadata contextLimit from the conversation DB)
  > default (512k)

and the result is raised to max_prompt * 1.05 whenever a larger prompt was seen without
compaction (the window is evidently bigger). Warnings fire at 68% / 82% of the target
(= the historical 350k / 420k at 512k).
"""
import json
import os
import sqlite3
import statistics
import time

DEFAULT_TARGET = 512_000
# Prefix -> target. Observed baseline: gemini-2.5-pro auto-compacted around
# 510k-514k; gemini-2.5-flash reached 505k without compaction.
BUILTIN = [
    ("gemini", 512_000),
    # Claude: the DB contextLimit (128000) is NOT where Antigravity compacts — opus-high ran
    # at 195k and opus-medium at 162k without compaction (2026-11-21/24). Use the Gemini value
    # per user request; a real auto-compaction sample still overrides it via "observed".
    ("claude-opus", 512_000),
    ("claude-sonnet", 512_000),
]
WARN_FRACTIONS = (350 / 512, 420 / 512)   # ~68% / ~82%
CHAR_LEVELS_AT_DEFAULT = ((700_000, 1), (900_000, 2))
RANGE_FRACTIONS = (480 / 512, 524 / 512)  # 480k-524k at 512k
MIN_SAMPLE = 30_000       # ignore tiny prompts (tests, broken usage records)
KEEP_SAMPLES = 5
USE_LAST = 3              # median of the most recent samples (window may change with app updates)
GROW_STEP = 2_000         # persist max_prompt only when it grew this much
SEEN_MARGIN = 1.05
DB_RETRY_SEC = 600


def _now():
    return round(time.time(), 3)


def path(data_dir):
    return os.path.join(data_dir, "calibration.json")


def load(data_dir):
    try:
        with open(path(data_dir), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save(data_dir, cal):
    os.makedirs(data_dir, exist_ok=True)
    p = path(data_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def _entry(cal, model):
    models = cal.setdefault("models", {})
    return models.setdefault(model, {})


def _median_recent(samples):
    vals = [int(s["tokens"]) for s in samples[-USE_LAST:] if isinstance(s, dict) and s.get("tokens")]
    return int(statistics.median(vals)) if vals else None


def record_auto(cal, model, tokens, conv="", step=None):
    """An automatic compaction happened right after a prompt of `tokens` on `model`."""
    if not model or not tokens or tokens < MIN_SAMPLE:
        return False
    e = _entry(cal, model)
    s = e.setdefault("samples", [])
    if step is not None and any(isinstance(x, dict) and x.get("conv") == conv[:8] and x.get("step") == step
                                for x in s):
        return False
    s.append({"tokens": int(tokens), "conv": conv[:8], "step": step, "at": _now()})
    del s[:-KEEP_SAMPLES]
    # The window shrank (app update / model change): an old larger max_prompt must not
    # keep pushing the target above where compaction really happens.
    if int(e.get("max_prompt") or 0) > tokens / 0.9:
        e["max_prompt"] = int(tokens)
    return True


def note_prompt(cal, model, tokens):
    """A prompt of `tokens` ran on `model` without compaction."""
    if not model or not tokens or tokens < MIN_SAMPLE:
        return False
    e = _entry(cal, model)
    changed = False
    if tokens >= int(e.get("max_prompt") or 0) + GROW_STEP:
        e["max_prompt"] = int(tokens)
        changed = True
    med = _median_recent(e.get("samples") or [])
    if med and tokens > med * 1.08:
        # Compaction did not happen well beyond the observed point: window grew.
        e["stale_samples"] = (e.get("samples") or [])[-KEEP_SAMPLES:]
        e["samples"] = []
        changed = True
    return changed


def needs_db(cal, model):
    if not model or any(model.startswith(p) for p, _ in BUILTIN):
        return False
    e = (cal.get("models") or {}).get(model) or {}
    if e.get("override") or e.get("samples") or e.get("db_limit"):
        return False
    return time.time() - float(e.get("db_checked") or 0) > DB_RETRY_SEC


# --- conversation DB (read-only; same protobuf fields the context_meter sidecar parses) ---

def _varint(b, p):
    r = s = 0
    while p < len(b):
        c = b[p]
        p += 1
        r |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return r, p
    return None, p


def _fields(b):
    p, out = 0, []
    while p < len(b):
        t, p = _varint(b, p)
        if t is None:
            break
        fn, wt = t >> 3, t & 7
        if wt == 0:
            v, p = _varint(b, p)
            if v is None:
                break
            out.append((fn, wt, v))
        elif wt == 2:
            n, p = _varint(b, p)
            if n is None or p + n > len(b):
                break
            out.append((fn, wt, b[p:p + n]))
            p += n
        elif wt == 1:
            p += 8
        elif wt == 5:
            p += 4
        else:
            break
    return out


def _gen_info(b):
    name = variant = limit = None
    for fn, wt, v in _fields(b):
        if fn == 3 and wt == 2:
            for sfn, swt, sv in _fields(v):
                if sfn == 6 and swt == 0:
                    limit = sv
                elif sfn == 28 and swt == 2:
                    variant = sv.decode("utf-8", "replace")
        elif fn == 1 and wt == 2 and len(v) < 1024 and not name:
            for sfn, swt, sv in _fields(v):
                if sfn == 19 and swt == 2:
                    name = sv.decode("utf-8", "replace")
    return name, variant, limit


def db_path_for(full_transcript, conv):
    """<appData>/brain/<conv>/.system_generated/logs/transcript_full.jsonl -> <appData>/conversations/<conv>.db"""
    cands = []
    if full_transcript:
        app = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(full_transcript))))))
        cands.append(os.path.join(app, "conversations", f"{conv}.db"))
    cands.append(os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "conversations", f"{conv}.db"))
    return next((c for c in cands if os.path.isfile(c)), None)


def read_db_limits(db_path):
    """Return {model_name: contextLimit} from gen_metadata, {} on any failure."""
    out = {}
    if not db_path:
        return out
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1)
    except Exception:
        return out
    try:
        for (data,) in db.execute("SELECT data FROM gen_metadata"):
            if not data:
                continue
            name, variant, limit = _gen_info(bytes(data))
            if limit and limit > 1000:
                for k in (variant, name):
                    if k:
                        out[k] = int(limit)
    except Exception:
        pass
    finally:
        try:
            db.close()
        except Exception:
            pass
    return out


def apply_db(cal, model, limits):
    e = _entry(cal, model)
    e["db_checked"] = _now()
    if limits.get(model):
        e["db_limit"] = limits[model]
    return True


# --- resolution ---

def target_for(cal, model):
    e = ((cal.get("models") or {}).get(model) or {}) if model else {}
    if e.get("override"):
        return int(e["override"]), "override"
    med = _median_recent(e.get("samples") or [])
    if med:
        base, src = med, "observed"
    else:
        hit = next((t for p, t in BUILTIN if model and model.startswith(p)), None)
        if hit:
            base, src = hit, "builtin"
        elif e.get("db_limit"):
            base, src = int(e["db_limit"]), "db_limit"
        else:
            base, src = DEFAULT_TARGET, "default"
    seen = int(e.get("max_prompt") or 0)
    # Observed targets are kept as is (note_prompt drops samples once clearly exceeded).
    if src != "observed" and seen > base:
        return int(seen * SEEN_MARGIN), src + "+seen"
    return base, src


def levels_for(cal, model):
    target, src = target_for(cal, model)
    scale = target / DEFAULT_TARGET
    return {
        "token": [[int(round(target * f, -3)), i + 1] for i, f in enumerate(WARN_FRACTIONS)],
        "char": [[int(round(c * scale, -3)), lv] for c, lv in CHAR_LEVELS_AT_DEFAULT],
        "target": int(target),
        "range": [int(round(target * RANGE_FRACTIONS[0], -3)), int(round(target * RANGE_FRACTIONS[1], -3))],
        "model": model or None,
        "source": src,
    }
