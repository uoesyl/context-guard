/**
 * Pure transcript and Context Guard state reader for context_meter.
 * Read-only: never writes or mutates conversation transcripts or guard state.
 */
import { existsSync, openSync, readFileSync, readSync, closeSync, readdirSync, realpathSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, isAbsolute, join, sep } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { pathToFileURL } from 'node:url';

export const ID_RE = /^[0-9a-f-]{8,}$/i;
export const COMPACTION_TARGET_TOKENS = 512_000;
export const COMPACTION_RANGE = [480_000, 524_000];
// Defaults only: guard.py writes the authoritative levels into each state file (`levels`).
export const TOKEN_LEVELS = [
  [350_000, 1],
  [420_000, 2],
];
export const CHAR_LEVELS = [
  [700_000, 1],
  [900_000, 2],
];

const MAX_SERIES_POINTS = 24;

function safeReaddir(p) {
  try {
    return readdirSync(p);
  } catch {
    return [];
  }
}

function statOf(p) {
  try {
    const s = statSync(p);
    return { exists: true, isFile: s.isFile(), isDirectory: s.isDirectory(), mtimeMs: s.mtimeMs, size: s.size };
  } catch {
    return { exists: false, isFile: false, isDirectory: false, mtimeMs: 0, size: -1 };
  }
}

export function isWithinDir(candidate, rootDir) {
  try {
    const realRoot = realpathSync(rootDir);
    const realCand = realpathSync(candidate);
    const normRoot = realRoot.endsWith(sep) ? realRoot : realRoot + sep;
    return realCand === realRoot || realCand.startsWith(normRoot);
  } catch {
    return false;
  }
}

export function findAppDataDirs(env = process.env) {
  const dataDir = env.ANTIGRAVITY_EXECUTABLE_DATA_DIR;
  if (dataDir) {
    for (let d = dataDir; d !== dirname(d); d = dirname(d)) {
      if (basename(d) !== 'sidecar_data') continue;
      const appData = dirname(d);
      return existsSync(join(appData, 'brain')) ? [appData] : [];
    }
    return [];
  }
  const root = join(homedir(), '.gemini');
  return safeReaddir(root)
    .sort()
    .map((name) => join(root, name))
    .filter((dir) => existsSync(join(dir, 'brain')));
}

export function getGuardDataDir(env = process.env) {
  return env.CONTEXT_GUARD_DATA || join(homedir(), '.gemini', 'antigravity', 'plugin_data', 'context-guard');
}

// Hook state older than the transcript by more than this = hook did not run.
export const HOOK_STALE_MS = 3 * 60 * 1000;

const hookConfigCache = { key: '', result: null };

/**
 * Health check of the global hooks.json that runs guard.py. Antigravity silently ignores
 * the whole file when it does not parse (e.g. a UTF-8 BOM), which disables every hook.
 */
export function checkHookConfig(path = join(homedir(), '.gemini', 'config', 'hooks.json')) {
  const st = statOf(path);
  if (!st.exists) return { ok: false, path, issues: ['hooks.json 不存在'] };
  const key = `${path}|${st.mtimeMs}|${st.size}`;
  if (key === hookConfigCache.key && hookConfigCache.result) return hookConfigCache.result;
  const issues = [];
  try {
    const buf = readFileSync(path);
    if (buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf) {
      issues.push('hooks.json 带 UTF-8 BOM，Antigravity 会忽略整个文件（所有 hook 失效）');
    }
    let cfg = null;
    try {
      cfg = JSON.parse(buf.toString('utf-8').replace(/^\uFEFF/, ''));
    } catch (e) {
      issues.push(`hooks.json 不是合法 JSON：${e.message}`);
    }
    if (cfg) {
      const cg = cfg['context-guard'];
      if (!cg) issues.push('hooks.json 缺少 context-guard 条目');
      else {
        if (cg.enabled === false) issues.push('context-guard hook 被设为 enabled:false');
        const needs = ['PreInvocation', 'PreToolUse'];
        for (const ev of needs) {
          const txt = JSON.stringify(cg[ev] || []);
          if (!txt.includes('guard.py')) issues.push(`context-guard 缺少 ${ev} → guard.py`);
        }
      }
    }
  } catch (e) {
    issues.push(`无法读取 hooks.json：${e.message}`);
  }
  const result = { ok: issues.length === 0, path, issues };
  hookConfigCache.key = key;
  hookConfigCache.result = result;
  return result;
}

export function stepChars(o) {
  let n = (o?.content?.length || 0) + (o?.thinking?.length || 0);
  if (o?.tool_calls) {
    n += JSON.stringify(o.tool_calls).length;
  }
  return n;
}

export function stepUsage(o) {
  if (o?.type !== 'PLANNER_RESPONSE') return null;
  const input = Number(o.input_tokens || 0);
  const cacheRead = Number(o.cache_read_tokens || 0);
  const output = Number(o.output_tokens || 0);
  if (!Number.isFinite(input) || !Number.isFinite(cacheRead) || !Number.isFinite(output)) return null;
  const inp = Math.max(0, input);
  const cr = Math.max(0, cacheRead);
  const out = Math.max(0, output);
  const prompt = inp + cr;
  if (prompt <= 0 && out <= 0) return null;
  const cacheHitRate = prompt > 0 ? Math.round((cr / prompt) * 10000) / 10000 : 0;
  return { prompt, input: inp, cacheRead: cr, output: out, cacheHitRate };
}

export function ctxTokens(o) {
  const u = stepUsage(o);
  return u && u.prompt > 0 ? u.prompt : null;
}

function createEmptyBucket() {
  return {
    steps: 0,
    inputTokens: 0,
    cacheReadTokens: 0,
    promptTokens: 0,
    outputTokens: 0,
    cacheHitRate: 0,
    cacheMisses: 0,
  };
}

function accumBucket(bucket, u) {
  bucket.steps += 1;
  bucket.inputTokens += u.input;
  bucket.cacheReadTokens += u.cacheRead;
  bucket.promptTokens += u.prompt;
  bucket.outputTokens += u.output;
  if (u.prompt > 0 && u.cacheRead === 0) {
    bucket.cacheMisses += 1;
  }
  bucket.cacheHitRate = bucket.promptTokens > 0
    ? Math.round((bucket.cacheReadTokens / bucket.promptTokens) * 10000) / 10000
    : 0;
}

const UUID_GLOBAL_RE = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;

function readPbVarint(buf, pos) {
  let val = 0n;
  let shift = 0n;
  while (pos < buf.length) {
    const b = buf[pos++];
    val |= BigInt(b & 0x7f) << shift;
    if ((b & 0x80) === 0) return [Number(val), pos];
    shift += 7n;
    if (shift > 63n) break;
  }
  return null;
}

function parsePbFields(buf) {
  const fields = [];
  let pos = 0;
  while (pos < buf.length) {
    const tagRes = readPbVarint(buf, pos);
    if (!tagRes) break;
    const [tag, p1] = tagRes;
    pos = p1;
    const fn = tag >>> 3;
    const wt = tag & 7;
    if (wt === 0) {
      const vRes = readPbVarint(buf, pos);
      if (!vRes) break;
      fields.push([fn, wt, vRes[0]]);
      pos = vRes[1];
    } else if (wt === 2) {
      const lRes = readPbVarint(buf, pos);
      if (!lRes) break;
      const len = lRes[0];
      pos = lRes[1];
      if (len < 0 || pos + len > buf.length) break;
      fields.push([fn, wt, buf.subarray(pos, pos + len)]);
      pos += len;
    } else if (wt === 1) {
      if (pos + 8 > buf.length) break;
      pos += 8;
    } else if (wt === 5) {
      if (pos + 4 > buf.length) break;
      pos += 4;
    } else {
      break;
    }
  }
  return fields;
}

function extractGenModelInfo(buf) {
  let uuid = null;
  let modelEnum = null;
  let modelName = null;
  let variant = null;
  let contextLimit = null;
  for (const [fn, wt, val] of parsePbFields(buf)) {
    if (fn === 4 && wt === 2 && val.length === 36) {
      uuid = val.toString('ascii');
    } else if (fn === 3 && wt === 2) {
      for (const [sfn, swt, sval] of parsePbFields(val)) {
        if (sfn === 1 && swt === 0) modelEnum = sval;
        else if (sfn === 6 && swt === 0) contextLimit = sval;
        else if (sfn === 28 && swt === 2) variant = sval.toString('utf-8');
      }
    } else if (fn === 1 && wt === 2 && val.length < 1024 && !modelName) {
      for (const [sfn, swt, sval] of parsePbFields(val)) {
        if (sfn === 3 && swt === 0 && modelEnum === null) modelEnum = sval;
        else if (sfn === 19 && swt === 2) modelName = sval.toString('utf-8');
      }
    }
  }
  const canonicalName = modelName || variant || (modelEnum !== null ? `model-${modelEnum}` : 'unknown');
  return { uuid, modelEnum, name: canonicalName, variant: variant || modelName || null, contextLimit };
}

function decodePbTimestamp(buf) {
  if (!buf || buf.length < 4 || buf.length > 16) return null;
  let sec = 0;
  let nanos = 0;
  for (const [fn, wt, val] of parsePbFields(buf)) {
    if (fn === 1 && wt === 0) sec = val;
    else if (fn === 2 && wt === 0) nanos = val;
  }
  if (sec < 1_700_000_000 || sec > 2_200_000_000) return null;
  return sec + nanos / 1e9;
}

function extractStepTiming(buf) {
  let tStart = null;
  let tFirst = null;
  let tEnd = null;
  for (const [fn, wt, val] of parsePbFields(buf)) {
    if (wt !== 2) continue;
    if (fn === 1) tStart = decodePbTimestamp(val);
    else if (fn === 6) tFirst = decodePbTimestamp(val);
    else if (fn === 7) tEnd = decodePbTimestamp(val);
  }
  const durationSec = tStart && tEnd && tEnd > tStart ? Math.round((tEnd - tStart) * 1000) / 1000 : null;
  const ttftSec = tStart && tFirst && tFirst >= tStart ? Math.round((tFirst - tStart) * 1000) / 1000 : null;
  const endAtMs = tEnd ? Math.round(tEnd * 1000) : null;
  return { durationSec, ttftSec, endAtMs };
}

export function readStepModels(dbPath, cache = new Map()) {
  if (!dbPath) return { map: new Map(), mtime: 0, size: 0 };
  const stDb = statOf(dbPath);
  if (!stDb.exists) return { map: new Map(), mtime: 0, size: 0 };
  const stWal = statOf(`${dbPath}-wal`);
  const stamp = `${stDb.mtimeMs}:${stDb.size}:${stWal.mtimeMs}:${stWal.size}`;
  const cached = cache.get(`db:${dbPath}`);
  if (cached && cached.stamp === stamp) {
    return { map: cached.map, mtime: Math.max(stDb.mtimeMs, stWal.mtimeMs), size: stDb.size + Math.max(0, stWal.size) };
  }

  const stepMap = new Map();
  let db = null;
  try {
    db = new DatabaseSync(dbPath, { readOnly: true });
    const uuidMap = new Map();
    const gmRows = db.prepare('SELECT idx, data FROM gen_metadata ORDER BY idx').all();
    for (const row of gmRows) {
      if (!row?.data) continue;
      const info = extractGenModelInfo(Buffer.from(row.data));
      if (info.uuid) uuidMap.set(info.uuid.toLowerCase(), info);
    }
    const stepRows = db.prepare('SELECT idx, metadata FROM steps WHERE step_type = 15').all();
    for (const row of stepRows) {
      if (!row?.metadata) continue;
      const metaBuf = Buffer.from(row.metadata);
      const timing = extractStepTiming(metaBuf);
      const text = metaBuf.toString('latin1');
      const matches = text.match(UUID_GLOBAL_RE) || [];
      let matched = null;
      for (const u of matches) {
        const hit = uuidMap.get(u.toLowerCase());
        if (hit) {
          const key = hit.modelEnum !== null ? `enum:${hit.modelEnum}` : `name:${hit.name}`;
          matched = { ...hit, key, ...timing };
          break;
        }
      }
      if (matched) {
        stepMap.set(Number(row.idx), matched);
      } else if (timing.durationSec !== null || timing.ttftSec !== null) {
        stepMap.set(Number(row.idx), { key: 'unknown', name: 'unknown', modelEnum: null, variant: null, contextLimit: null, ...timing });
      }
    }
  } catch {
    // Graceful fallback if DB is locked, malformed, or unavailable.
  } finally {
    try { db?.close(); } catch {}
  }

  cache.set(`db:${dbPath}`, { stamp, map: stepMap });
  return { map: stepMap, mtime: Math.max(stDb.mtimeMs, stWal.mtimeMs), size: stDb.size + Math.max(0, stWal.size) };
}

const CHARS_PER_TOKEN = 3.0;
const SKILL_RE = /[\\/]skills[\\/]([^\\/]+)[\\/](.+)$/i;
const MSG_SENDER_RE = /sender=\S+?\/(task|[0-9a-f]{8})/i;

export function newUsage() {
  return {
    v: 1,
    window: { cats: {}, first_prompt: null, prompt: null, since_step: null },
    total: { cats: {} },
    tools: {},
    skills: {},
    mcp: {},
    pending: [],
    queue: [],
    last_prompt: null,
    compactions: 0,
  };
}

function _bumpUsage(cats, cat, chars = 0, tokens = 0, n = 1) {
  const e = (cats[cat] ||= { chars: 0, tokens: 0, n: 0 });
  e.chars += Math.trunc(chars);
  e.tokens += Math.round(tokens);
  e.n += n;
}

function _bookUsage(u, cat, chars = 0, tokens = 0, n = 1) {
  _bumpUsage(u.window.cats, cat, chars, tokens, n);
  _bumpUsage(u.total.cats, cat, chars, tokens, n);
}

function _toolUsage(u, name) {
  return (u.tools[name] ||= {
    calls: 0,
    arg_chars: 0,
    result_chars: 0,
    result_tokens: 0,
    arg_tokens: 0,
    errors: 0,
  });
}

function _systemUsageCat(o) {
  const content = o.content || '';
  if (o.source === 'SYSTEM_SDK' || content.includes('injected by a PreInvocation hook')) {
    return 'system:context_guard';
  }
  if (MSG_SENDER_RE.test(content)) {
    return 'system:task_notice';
  }
  return 'system:other';
}

function summarizeToolCall(tc) {
  if (tc && tc._summarized) return tc;
  const name = String(tc.name || 'unknown');
  const args = tc.args && typeof tc.args === 'object' && !Array.isArray(tc.args) ? tc.args : {};
  let mcpKey = null;
  if (name === 'call_mcp_tool' || name.startsWith('mcp_')) {
    mcpKey = name === 'call_mcp_tool'
      ? `${args.ServerName || 'unknown'}/${args.ToolName || 'unknown'}`
      : name.slice(4).replace('_', '/');
  }
  let skill = null;
  for (const k of ['Path', 'AbsolutePath', 'path', 'CommandLine', 'ScriptPath']) {
    const v = args[k];
    if (typeof v !== 'string') continue;
    const m = SKILL_RE.exec(v);
    if (m) {
      skill = { name: m[1], isLoad: m[2].trim().toLowerCase().startsWith('skill.md') };
      break;
    }
  }
  return {
    _summarized: true,
    name,
    _argChars: JSON.stringify(args).length,
    _tcChars: JSON.stringify(tc).length,
    _mcpKey: mcpKey,
    _skill: skill,
  };
}

function _noteToolCall(u, tc, step) {
  const sum = summarizeToolCall(tc);
  const t = _toolUsage(u, sum.name);
  t.calls += 1;
  t.arg_chars += sum._argChars;
  if (sum._mcpKey) {
    u.mcp[sum._mcpKey] = Number(u.mcp[sum._mcpKey] || 0) + 1;
  }
  if (sum._skill) {
    const s = (u.skills[sum._skill.name] ||= { loads: 0, resource_uses: 0, first_step: step, last_step: step });
    if (sum._skill.isLoad) s.loads += 1;
    else s.resource_uses += 1;
    s.last_step = step;
  }
  return sum;
}

function _creditToolUsage(u, cat, share) {
  if (cat.startsWith('tool:')) {
    _toolUsage(u, cat.slice(5)).result_tokens += Math.round(share);
  } else if (cat.startsWith('args:')) {
    const t = _toolUsage(u, cat.slice(5));
    t.arg_tokens = Number(t.arg_tokens || 0) + Math.round(share);
  }
}

function _settleUsage(u, prompt) {
  const w = u.window;
  const pending = u.pending;
  const chars = pending.reduce((sum, p) => sum + p[1], 0);
  if (u.last_prompt === null) {
    const est = Math.min(prompt, chars / CHARS_PER_TOKEN);
    for (const [cat, c] of pending) {
      const share = chars ? (est * c) / chars : 0;
      _bookUsage(u, cat, c, share);
      _creditToolUsage(u, cat, share);
    }
    _bookUsage(u, 'baseline', 0, prompt - est);
    w.first_prompt = prompt;
  } else {
    let delta = prompt - u.last_prompt;
    if (delta < 0) {
      _bookUsage(u, 'trimmed', 0, delta);
      delta = 0;
    }
    if (!pending.length && delta) {
      _bookUsage(u, 'unattributed', 0, delta);
    }
    for (const [cat, c] of pending) {
      const share = chars ? (delta * c) / chars : delta / pending.length;
      _bookUsage(u, cat, c, share);
      _creditToolUsage(u, cat, share);
    }
  }
  u.pending = [];
  u.last_prompt = prompt;
  w.prompt = prompt;
}

export function feedUsage(u, o) {
  if (!o || typeof o !== 'object') return;
  const typ = o.type;
  const step = o.step_index ?? null;
  const contentLen = typeof o._contentLen === 'number'
    ? o._contentLen
    : (typeof o.content === 'string' ? o.content : (o.content ? String(o.content) : '')).length;
  if (typ === 'CHECKPOINT') {
    u.compactions += 1;
    u.window = { cats: {}, first_prompt: null, prompt: null, since_step: step };
    u.last_prompt = null;
    u.queue = [];
    u.pending = [['checkpoint', contentLen]];
    return;
  }
  if (typ === 'PLANNER_RESPONSE') {
    const inp = Number(o.input_tokens || 0);
    const cr = Number(o.cache_read_tokens || 0);
    const prompt = Number.isFinite(inp) && Number.isFinite(cr) ? Math.max(0, inp) + Math.max(0, cr) : 0;
    if (prompt > 0) _settleUsage(u, prompt);
    const thinkingLen = typeof o._thinkingLen === 'number'
      ? o._thinkingLen
      : (o.thinking ? String(o.thinking).length : 0);
    if (thinkingLen > 0) u.pending.push(['model:thinking', thinkingLen]);
    if (contentLen > 0) u.pending.push(['model:text', contentLen]);
    const tcs = Array.isArray(o.tool_calls) ? o.tool_calls.filter((tc) => tc && typeof tc === 'object') : [];
    if (tcs.length) {
      const sums = tcs.map((tc) => _noteToolCall(u, tc, step));
      u.queue = sums.map((s) => s.name);
      for (const s of sums) {
        u.pending.push([`args:${s.name}`, s._tcChars]);
      }
    }
    return;
  }
  if (typ === 'GENERIC') {
    const name = u.queue.length ? u.queue.shift() : 'unknown';
    const t = _toolUsage(u, name);
    t.result_chars += contentLen;
    if (o.status === 'ERROR' || o.error) t.errors += 1;
    u.pending.push([`tool:${name}`, contentLen]);
  } else if (typ === 'USER_INPUT') {
    u.pending.push(['user_input', contentLen]);
  } else if (typ === 'SYSTEM_MESSAGE') {
    u.pending.push([o._sysCat || _systemUsageCat(o), contentLen]);
  } else if (typ === 'ERROR_MESSAGE') {
    const errLen = typeof o._errorLen === 'number'
      ? o._errorLen
      : String(o.error || (typeof o.content === 'string' ? o.content : '')).length;
    u.pending.push(['error', errLen]);
  }
}

function compactReplayStep(o, resolvedStep) {
  const typ = o?.type;
  const contentStr = typeof o?.content === 'string' ? o.content : (o?.content ? String(o.content) : '');
  const thinkingStr = typeof o?.thinking === 'string' ? o.thinking : (o?.thinking ? String(o.thinking) : '');
  const tcs = Array.isArray(o?.tool_calls) ? o.tool_calls.filter((tc) => tc && typeof tc === 'object') : [];
  return {
    type: typ,
    step_index: resolvedStep,
    _chars: stepChars(o),
    _contentLen: contentStr.length,
    _thinkingLen: thinkingStr.length,
    _sysCat: typ === 'SYSTEM_MESSAGE' ? _systemUsageCat(o) : null,
    _errorLen: typ === 'ERROR_MESSAGE' ? String(o?.error || contentStr).length : 0,
    input_tokens: o?.input_tokens,
    cache_read_tokens: o?.cache_read_tokens,
    status: o?.status,
    error: Boolean(o?.error),
    tool_calls: tcs.length ? tcs.map(summarizeToolCall) : null,
  };
}

function createEmptyAccumulator() {
  return {
    offset: 0,
    chars: 0,
    tokens: null,
    inputTokens: null,
    cacheReadTokens: null,
    outputTokens: null,
    cacheHitRate: null,
    lastStepIndex: 0,
    stepsSinceCheckpoint: 0,
    lastCheckpointStep: null,
    compactionCount: 0,
    checkpoints: [],
    series: [],
    stepEvents: [],
    plannerStartOffsets: [],
    activeSteps: [],
    cycle: createEmptyBucket(),
    totals: createEmptyBucket(),
    usage: newUsage(),
    seenStep: false,
    // Model calls on branches the user discarded by editing / retrying an earlier message.
    rewound: { count: 0, steps: 0, promptTokens: 0, outputTokens: 0 },
  };
}

function bucketInput(ev) {
  return { prompt: ev.promptTokens, input: ev.inputTokens, cacheRead: ev.cacheReadTokens, output: ev.outputTokens };
}

/**
 * The user edited / retried the message at step `r`: Antigravity appends the new branch to
 * transcript_full.jsonl reusing step indices >= r, while the old rows stay in the file. Drop
 * the superseded model calls so they are neither counted twice nor matched against the new
 * branch's DB metadata (which is keyed by step index).
 */
function rewindTo(acc, r) {
  const keep = [];
  let dropped = 0;
  for (const ev of acc.stepEvents) {
    if (ev.step >= r) {
      dropped += 1;
      acc.rewound.steps += 1;
      acc.rewound.promptTokens += ev.promptTokens;
      acc.rewound.outputTokens += ev.outputTokens;
    } else {
      keep.push(ev);
    }
  }
  if (dropped) acc.rewound.count += 1;
  acc.stepEvents = keep;
  acc.checkpoints = acc.checkpoints.filter((c) => !(Number.isFinite(c.step) && c.step >= r));
  acc.compactionCount = acc.checkpoints.length;
  acc.lastCheckpointStep = acc.checkpoints.length ? acc.checkpoints[acc.checkpoints.length - 1].step : null;
  acc.series = keep
    .filter((ev) => ev.cycleIndex === acc.compactionCount && ev.promptTokens > 0)
    .slice(-MAX_SERIES_POINTS)
    .map((ev) => ({
      step: ev.step,
      tokens: ev.promptTokens,
      inputTokens: ev.inputTokens,
      cacheReadTokens: ev.cacheReadTokens,
      outputTokens: ev.outputTokens > 0 ? ev.outputTokens : null,
      cacheHitRate: ev.cacheHitRate,
      createdAt: ev.createdAt,
    }));
  acc.totals = createEmptyBucket();
  acc.cycle = createEmptyBucket();
  let last = null;
  for (const ev of keep) {
    const u = bucketInput(ev);
    accumBucket(acc.totals, u);
    if (ev.cycleIndex === acc.compactionCount) {
      accumBucket(acc.cycle, u);
      if (ev.promptTokens > 0) last = ev;
    }
  }
  acc.tokens = last ? last.promptTokens : null;
  acc.inputTokens = last ? last.inputTokens : null;
  acc.cacheReadTokens = last ? last.cacheReadTokens : null;
  acc.cacheHitRate = last ? last.cacheHitRate : null;
  acc.outputTokens = last && last.outputTokens > 0 ? last.outputTokens : null;

  const keptSteps = [];
  let inDiscarded = false;
  for (const s of acc.activeSteps) {
    if (s.step_index >= 0) {
      inDiscarded = s.step_index >= r;
    }
    if (!inDiscarded) keptSteps.push(s);
  }
  acc.activeSteps = keptSteps;
  acc.usage = newUsage();
  acc.chars = 0;
  let lastCkptIdx = -1;
  for (let i = 0; i < acc.activeSteps.length; i += 1) {
    const s = acc.activeSteps[i];
    feedUsage(acc.usage, s);
    if (s.type === 'CHECKPOINT') {
      acc.chars = s._contentLen;
      lastCkptIdx = i;
    } else {
      acc.chars += s._chars;
    }
  }
  acc.stepsSinceCheckpoint = acc.activeSteps.length - lastCkptIdx - 1;
  acc.lastStepIndex = r;
}

function applyStep(acc, o, lineStart = 0) {
  const rawStep = o?.step_index;
  const stepIdx = rawStep !== null && rawStep !== undefined && rawStep !== '' ? Number(rawStep) : NaN;
  const hasStep = Number.isFinite(stepIdx) && stepIdx >= 0;
  if (o?.type === 'USER_INPUT' && hasStep && acc.seenStep && stepIdx <= acc.lastStepIndex) {
    rewindTo(acc, stepIdx);
  }
  if (hasStep) acc.seenStep = true;
  if (hasStep && stepIdx > acc.lastStepIndex) {
    acc.lastStepIndex = stepIdx;
  }
  const replayRow = compactReplayStep(o, hasStep ? stepIdx : -1);
  acc.activeSteps.push(replayRow);
  feedUsage(acc.usage, replayRow);
  if (o?.type === 'PLANNER_RESPONSE') {
    acc.plannerStartOffsets.push(lineStart);
  }
  if (o?.type === 'CHECKPOINT') {
    acc.compactionCount += 1;
    acc.lastCheckpointStep = Number.isFinite(stepIdx) ? stepIdx : acc.lastCheckpointStep;
    acc.checkpoints.push({
      step: acc.lastCheckpointStep,
      createdAt: o.created_at || null,
    });
    acc.chars = (o.content || '').length;
    acc.tokens = null;
    acc.inputTokens = null;
    acc.cacheReadTokens = null;
    acc.outputTokens = null;
    acc.cacheHitRate = null;
    acc.stepsSinceCheckpoint = 0;
    acc.series = [];
    acc.cycle = createEmptyBucket();
    return;
  }
  acc.stepsSinceCheckpoint += 1;
  acc.chars += stepChars(o);
  const u = stepUsage(o);
  if (u) {
    accumBucket(acc.cycle, u);
    accumBucket(acc.totals, u);
    const resolvedStep = Number.isFinite(stepIdx) ? stepIdx : acc.lastStepIndex;
    const durRaw = Number(o.duration_sec);
    acc.stepEvents.push({
      step: resolvedStep,
      cycleIndex: acc.compactionCount,
      createdAt: o.created_at || null,
      promptTokens: u.prompt,
      inputTokens: u.input,
      cacheReadTokens: u.cacheRead,
      outputTokens: u.output,
      cacheHitRate: u.cacheHitRate,
      modelHint: typeof o.model === 'string' && o.model ? o.model : null,
      durationSec: Number.isFinite(durRaw) && durRaw > 0 ? durRaw : null,
    });
    if (u.prompt > 0) {
      acc.tokens = u.prompt;
      acc.inputTokens = u.input;
      acc.cacheReadTokens = u.cacheRead;
      acc.cacheHitRate = u.cacheHitRate;
      acc.outputTokens = u.output > 0 ? u.output : null;
      acc.series.push({
        step: resolvedStep,
        tokens: u.prompt,
        inputTokens: u.input,
        cacheReadTokens: u.cacheRead,
        outputTokens: acc.outputTokens,
        cacheHitRate: u.cacheHitRate,
        createdAt: o.created_at || null,
      });
      if (acc.series.length > MAX_SERIES_POINTS) {
        acc.series = acc.series.slice(-MAX_SERIES_POINTS);
      }
    } else if (u.output > 0) {
      acc.outputTokens = u.output;
    }
  }
}

/**
 * Incrementally reads JSONL lines from transcriptPath using a per-path cache.
 */
export function parseTranscriptIncremental(transcriptPath, cache = new Map()) {
  const st = statOf(transcriptPath);
  if (!st.exists) return null;

  let acc = cache.get(transcriptPath);
  if (!acc || st.size < acc.offset) {
    acc = createEmptyAccumulator();
  }

  if (st.size > acc.offset) {
    const toRead = st.size - acc.offset;
    const buf = Buffer.allocUnsafe(toRead);
    const fd = openSync(transcriptPath, 'r');
    let bytesRead = 0;
    try {
      bytesRead = readSync(fd, buf, 0, toRead, acc.offset);
    } finally {
      closeSync(fd);
    }
    const slice = buf.subarray(0, bytesRead);
    const lastNl = slice.lastIndexOf(0x0a);
    if (lastNl >= 0) {
      let pos = 0;
      while (pos <= lastNl) {
        const nl = slice.indexOf(0x0a, pos);
        if (nl < 0 || nl > lastNl) break;
        const lineStart = acc.offset + pos;
        const line = slice.subarray(pos, nl).toString('utf-8').trim();
        pos = nl + 1;
        if (!line) continue;
        try {
          applyStep(acc, JSON.parse(line), lineStart);
        } catch {
          // Skip partial or malformed lines safely.
        }
      }
      acc.offset += lastNl + 1;
    }
  }

  cache.set(transcriptPath, acc);
  const { activeSteps: _activeSteps, ...restAcc } = acc;
  return {
    ...restAcc,
    cycle: { ...acc.cycle },
    totals: { ...acc.totals },
    usage: acc.usage,
    checkpoints: [...acc.checkpoints],
    series: [...acc.series],
    stepEvents: [...acc.stepEvents],
    rewound: { ...acc.rewound },
    plannerStartOffsets: acc.plannerStartOffsets,
    transcriptMtime: st.mtimeMs,
    transcriptSize: st.size,
  };
}

export function resolveConversationFiles(convId, appDataDirs) {
  if (!convId || !ID_RE.test(convId)) return null;
  for (const appData of appDataDirs) {
    const brainDir = join(appData, 'brain');
    const convDir = join(brainDir, convId);
    if (!existsSync(convDir) || !isWithinDir(convDir, brainDir)) continue;
    const logsDir = join(convDir, '.system_generated', 'logs');
    const fullPath = join(logsDir, 'transcript_full.jsonl');
    const compactPath = join(logsDir, 'transcript.jsonl');
    const transcriptPath = existsSync(fullPath) ? fullPath : (existsSync(compactPath) ? compactPath : null);
    const convsDir = join(appData, 'conversations');
    const candidateDb = join(convsDir, `${convId}.db`);
    const dbPath = existsSync(candidateDb) && isWithinDir(candidateDb, convsDir) ? candidateDb : null;
    return { appData, convDir, transcriptPath, dbPath };
  }
  return null;
}

export function readGuardState(guardDataDir, convId) {
  if (!convId || !ID_RE.test(convId)) return null;
  const p = join(guardDataDir, 'state', `${convId}.json`);
  const st = statOf(p);
  if (!st.exists) return null;
  try {
    const data = JSON.parse(readFileSync(p, 'utf-8'));
    return { data, mtimeMs: st.mtimeMs };
  } catch {
    return null;
  }
}

function isValidLocalDir(dirPath) {
  if (typeof dirPath !== 'string') return false;
  const trimmed = dirPath.trim();
  if (!trimmed || trimmed.includes('\0')) return false;
  if (trimmed.startsWith('\\\\') || trimmed.startsWith('//')) return false;
  if (!isAbsolute(trimmed)) return false;
  return statOf(trimmed).isDirectory;
}

export function readTaskStateInfo(convDir, guardArtifactDir = null) {
  const dirs = [];
  if (isValidLocalDir(guardArtifactDir)) dirs.push(guardArtifactDir.trim());
  if (convDir) dirs.push(convDir);
  for (const dir of dirs) {
    const p = join(dir, 'task_state.md');
    const st = statOf(p);
    if (st.exists && st.isFile && isWithinDir(p, dir)) {
      return {
        exists: true,
        size: st.size,
        mtime: st.mtimeMs,
        uri: pathToFileURL(realpathSync(p)).href,
      };
    }
  }
  return { exists: false, size: 0, mtime: null, uri: null };
}

function validLevels(v) {
  return Array.isArray(v) && v.length >= 2 && v.every((x) => Array.isArray(x) && x.length === 2 && x.every(Number.isFinite));
}

/** Levels from guard state (single source of truth), falling back to built-in defaults. */
export function resolveLevels(g) {
  const L = g && g.levels;
  return {
    token: validLevels(L?.token) ? L.token : TOKEN_LEVELS,
    char: validLevels(L?.char) ? L.char : CHAR_LEVELS,
    target: Number.isFinite(L?.target) ? L.target : COMPACTION_TARGET_TOKENS,
    range: Array.isArray(L?.range) && L.range.length === 2 ? L.range : COMPACTION_RANGE,
    model: typeof L?.model === 'string' && L.model ? L.model : null,
    source: typeof L?.source === 'string' && L.source ? L.source : null,
  };
}

export function computeThresholdLevel(tokens, chars, levels = resolveLevels(null)) {
  if (tokens && tokens > 0) {
    let lv = 0;
    for (const [th, level] of levels.token) {
      if (tokens >= th) lv = level;
    }
    return lv;
  }
  let lv = 0;
  for (const [th, level] of levels.char) {
    if ((chars || 0) >= th) lv = level;
  }
  return lv;
}

function computeTrend(series, tokens, target = COMPACTION_TARGET_TOKENS) {
  if (!series.length) {
    return {
      avgDeltaPerStep: null,
      lastDelta: null,
      estimatedStepsLeft: null,
      avgOutputPerStep: null,
      avgInputPerStep: null,
      recentCacheHitRate: null,
    };
  }
  const deltas = [];
  let sumOut = 0;
  let countOut = 0;
  let sumIn = 0;
  let sumCache = 0;
  let sumPrompt = 0;
  for (let i = 0; i < series.length; i += 1) {
    const p = series[i];
    if (i > 0) {
      const d = p.tokens - series[i - 1].tokens;
      if (d >= 0) deltas.push(d);
    }
    if (Number.isFinite(p.outputTokens)) {
      sumOut += p.outputTokens;
      countOut += 1;
    }
    if (Number.isFinite(p.inputTokens)) sumIn += p.inputTokens;
    if (Number.isFinite(p.cacheReadTokens)) sumCache += p.cacheReadTokens;
    if (Number.isFinite(p.tokens)) sumPrompt += p.tokens;
  }
  const lastDelta = series.length >= 2 ? series[series.length - 1].tokens - series[series.length - 2].tokens : null;
  const avgDelta = deltas.length ? Math.round(deltas.reduce((a, b) => a + b, 0) / deltas.length) : null;
  let estimatedStepsLeft = null;
  if (tokens && avgDelta && avgDelta > 0) {
    estimatedStepsLeft = Math.max(0, Math.round((target - tokens) / avgDelta));
  }
  const avgOutputPerStep = countOut ? Math.round(sumOut / countOut) : null;
  const avgInputPerStep = series.length ? Math.round(sumIn / series.length) : null;
  const recentCacheHitRate = sumPrompt > 0 ? Math.round((sumCache / sumPrompt) * 10000) / 10000 : null;
  return {
    avgDeltaPerStep: avgDelta,
    lastDelta,
    estimatedStepsLeft,
    avgOutputPerStep,
    avgInputPerStep,
    recentCacheHitRate,
  };
}

function aggregateModels(stepEvents, stepModelMap, currentCycleIndex, sessionTotals, rewound = null) {
  const byKey = new Map();
  let activeModel = null;
  let allTimedOutput = 0;
  let allDurationSec = 0;
  let lastTps = null;
  let lastDurationSec = null;
  let cycleLastActivity = null;
  const cycleModels = new Set();

  // One row per model id as written in the transcript (`model`, e.g. gemini-2.5-pro-high).
  // The DB only fills in metadata (enum, timing, context limit): keying rows by the DB enum
  // split one model into an `enum:` row and a `name:` row whenever some steps lacked DB
  // metadata, and showed two identically named rows for enums that differ only by reasoning
  // level (the DB name omits the -high / -xhigh suffix).
  const enumHints = new Map();
  for (const ev of stepEvents) {
    const meta = stepModelMap.get(ev.step);
    if (!ev.modelHint || meta?.modelEnum === null || meta?.modelEnum === undefined) continue;
    const counts = enumHints.get(meta.modelEnum) || new Map();
    counts.set(ev.modelHint, (counts.get(ev.modelHint) || 0) + 1);
    enumHints.set(meta.modelEnum, counts);
  }
  const hintForEnum = (en) => {
    const counts = enumHints.get(en);
    if (!counts) return null;
    return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  };
  const usableName = (n) => (n && n !== 'unknown' && !n.startsWith('model-') ? n : null);

  for (const ev of stepEvents) {
    const meta = stepModelMap.get(ev.step);
    const modelEnum = meta?.modelEnum ?? null;
    const resolved = ev.modelHint || (modelEnum !== null ? hintForEnum(modelEnum) : null);
    const fallbackName = usableName(meta?.name);
    const key = resolved
      ? `name:${resolved}`
      : (modelEnum !== null ? `enum:${modelEnum}` : (fallbackName ? `name:${fallbackName}` : 'unknown'));
    const name = resolved || meta?.name || 'unknown';
    const variant = meta?.variant || null;
    const contextLimit = meta?.contextLimit ?? null;
    const durationSec = meta?.durationSec ?? ev.durationSec ?? null;
    const ttftSec = meta?.ttftSec ?? null;
    const stepTps = durationSec && durationSec > 0 && ev.outputTokens > 0
      ? Math.round((ev.outputTokens / durationSec) * 10) / 10
      : null;

    let entry = byKey.get(key);
    if (!entry) {
      entry = {
        key,
        name,
        modelEnum,
        modelEnums: new Set(),
        variants: new Set(),
        contextLimit,
        bucket: createEmptyBucket(),
        cycle: createEmptyBucket(),
        maxPromptTokens: 0,
        lastPromptTokens: 0,
        firstStepIndex: ev.step,
        lastStepIndex: ev.step,
        timedOutputTokens: 0,
        totalDurationSec: 0,
        ttftSum: 0,
        ttftCount: 0,
        lastTps: null,
        lastDurationSec: null,
        stepDetails: [],
      };
      byKey.set(key, entry);
    }
    if (entry.modelEnum === null && modelEnum !== null) entry.modelEnum = modelEnum;
    if (modelEnum !== null) entry.modelEnums.add(modelEnum);
    // Prefer a human name if an earlier step of an enum-keyed row fell back to model-<enum>
    if (entry.name.startsWith('model-') && name && !name.startsWith('model-')) entry.name = name;
    if (variant) entry.variants.add(variant);
    if (contextLimit && !entry.contextLimit) entry.contextLimit = contextLimit;

    const u = {
      prompt: ev.promptTokens,
      input: ev.inputTokens,
      cacheRead: ev.cacheReadTokens,
      output: ev.outputTokens,
    };
    accumBucket(entry.bucket, u);
    if (ev.cycleIndex === currentCycleIndex) {
      accumBucket(entry.cycle, u);
    }
    if (ev.promptTokens > entry.maxPromptTokens) entry.maxPromptTokens = ev.promptTokens;
    entry.lastPromptTokens = ev.promptTokens;
    entry.lastStepIndex = ev.step;
    if (durationSec && durationSec > 0 && ev.outputTokens > 0) {
      entry.timedOutputTokens += ev.outputTokens;
      entry.totalDurationSec += durationSec;
      allTimedOutput += ev.outputTokens;
      allDurationSec += durationSec;
      entry.lastTps = stepTps;
      entry.lastDurationSec = durationSec;
      lastTps = stepTps;
      lastDurationSec = durationSec;
    } else {
      lastTps = null;
      lastDurationSec = durationSec;
    }
    if (ttftSec !== null && ttftSec >= 0) {
      entry.ttftSum += ttftSec;
      entry.ttftCount += 1;
    }
    entry.stepDetails.push({
      step: ev.step,
      cycleIndex: ev.cycleIndex,
      createdAt: ev.createdAt,
      inputTokens: ev.inputTokens,
      cacheReadTokens: ev.cacheReadTokens,
      promptTokens: ev.promptTokens,
      outputTokens: ev.outputTokens,
      cacheHitRate: ev.cacheHitRate,
      durationSec,
      ttftSec,
      tps: stepTps,
      variant,
    });
    activeModel = { key, name: entry.name, modelEnum, variant };
    if (ev.cycleIndex === currentCycleIndex) {
      const parsedCreated = ev.createdAt ? Date.parse(ev.createdAt) : NaN;
      const atMs = Number.isFinite(meta?.endAtMs) && meta.endAtMs > 0
        ? meta.endAtMs
        : (Number.isFinite(parsedCreated) && parsedCreated > 0 ? parsedCreated : null);
      cycleLastActivity = {
        model: entry.name,
        step: ev.step,
        atMs,
        cacheHitRate: ev.cacheHitRate,
      };
      if (entry.name && entry.name !== 'unknown') cycleModels.add(entry.name);
    }
  }

  const totalPrompt = sessionTotals.promptTokens || 0;
  const totalOutput = sessionTotals.outputTokens || 0;
  const list = [...byKey.values()]
    .map((e) => ({
      key: e.key,
      name: e.name,
      modelEnum: e.modelEnum,
      modelEnums: [...e.modelEnums],
      variants: [...e.variants],
      contextLimit: e.contextLimit,
      ...e.bucket,
      avgInputPerStep: e.bucket.steps ? Math.round(e.bucket.inputTokens / e.bucket.steps) : 0,
      avgCacheReadPerStep: e.bucket.steps ? Math.round(e.bucket.cacheReadTokens / e.bucket.steps) : 0,
      avgOutputPerStep: e.bucket.steps ? Math.round(e.bucket.outputTokens / e.bucket.steps) : 0,
      avgPromptPerStep: e.bucket.steps ? Math.round(e.bucket.promptTokens / e.bucket.steps) : 0,
      tps: e.totalDurationSec > 0 ? Math.round((e.timedOutputTokens / e.totalDurationSec) * 10) / 10 : null,
      lastTps: e.lastTps,
      lastDurationSec: e.lastDurationSec,
      totalDurationSec: Math.round(e.totalDurationSec * 10) / 10,
      avgTtftSec: e.ttftCount > 0 ? Math.round((e.ttftSum / e.ttftCount) * 100) / 100 : null,
      maxPromptTokens: e.maxPromptTokens,
      lastPromptTokens: e.lastPromptTokens,
      firstStepIndex: e.firstStepIndex,
      lastStepIndex: e.lastStepIndex,
      shareOfPrompt: totalPrompt > 0 ? Math.round((e.bucket.promptTokens / totalPrompt) * 10000) / 10000 : 0,
      shareOfOutput: totalOutput > 0 ? Math.round((e.bucket.outputTokens / totalOutput) * 10000) / 10000 : 0,
      cycle: e.cycle,
      stepDetails: e.stepDetails,
    }))
    .sort((a, b) => b.promptTokens - a.promptTokens || b.steps - a.steps);

  const lastActivityModel = cycleLastActivity?.model || null;
  const isHydrogen = Boolean(lastActivityModel && /hydrogen/i.test(lastActivityModel));
  const hasPlannerInCycle = [...cycleModels].some((m) => !/hydrogen/i.test(m));

  return {
    activeModel,
    lastTps,
    lastDurationSec,
    activity: {
      lastModel: lastActivityModel,
      lastStep: cycleLastActivity?.step ?? null,
      lastStepAtMs: cycleLastActivity?.atMs ?? null,
      lastCacheHitRate: cycleLastActivity?.cacheHitRate ?? null,
      cycleModels: [...cycleModels],
      crossModelCompactRisk: Boolean(isHydrogen && hasPlannerInCycle),
    },
    list,
    totals: {
      ...sessionTotals,
      modelCount: list.length,
      rewound: rewound && rewound.steps ? { ...rewound } : null,
      totalDurationSec: Math.round(allDurationSec * 10) / 10,
      tps: allDurationSec > 0 ? Math.round((allTimedOutput / allDurationSec) * 10) / 10 : null,
    },
  };
}

// --- Context composition (from guard state `usage`, written by hooks/usage.py) ---

/** Semantic groups: few, stable colors so the distribution reads at a glance. */
export const BREAKDOWN_GROUPS = [
  { id: 'tool', label: '工具结果', hint: '读文件、命令输出等工具返回的内容' },
  { id: 'args', label: '工具参数', hint: '写入/编辑的内容、命令行等调用参数' },
  { id: 'model', label: '模型输出', hint: '思考过程与回复正文' },
  { id: 'baseline', label: '系统基线', hint: '系统提示、工具定义、规则、技能列表' },
  { id: 'msg', label: '消息与注入', hint: '用户输入、后台通知、Hook 注入、压缩摘要、错误' },
];

const CAT_LABELS = {
  baseline: '系统基线',
  user_input: '用户输入',
  'model:thinking': '模型思考',
  'model:text': '回复正文',
  'model:tool_args': '工具参数',
  'system:context_guard': 'Hook 注入',
  'system:task_notice': '后台任务通知',
  'system:other': '其他系统消息',
  checkpoint: '压缩摘要',
  error: '错误消息',
  unattributed: '未归因',
};

export function catGroup(cat) {
  if (cat.startsWith('tool:')) return 'tool';
  if (cat.startsWith('args:') || cat === 'model:tool_args') return 'args';
  if (cat.startsWith('model:')) return 'model';
  if (cat === 'baseline') return 'baseline';
  return 'msg';
}

export function catLabel(cat) {
  if (cat.startsWith('tool:') || cat.startsWith('args:')) return cat.slice(5);
  return CAT_LABELS[cat] || cat;
}

function breakdownScope(cats, targetTotal = null) {
  const trimmed = Math.round(Number(cats?.trimmed?.tokens) || 0);
  let items = [];
  for (const [cat, e] of Object.entries(cats || {})) {
    if (cat === 'trimmed' || !e || !(Number(e.tokens) > 0)) continue;
    items.push({ cat, label: catLabel(cat), group: catGroup(cat), tokens: Math.round(e.tokens), chars: Number(e.chars) || 0, n: Number(e.n) || 0 });
  }
  const rawTotal = items.reduce((a, b) => a + b.tokens, 0);
  if (targetTotal > 0 && trimmed < 0 && rawTotal > targetTotal) {
    const ratio = targetTotal / rawTotal;
    items = items
      .map((it) => ({ ...it, tokens: Math.round(it.tokens * ratio) }))
      .filter((it) => it.tokens > 0);
  }
  const total = items.reduce((a, b) => a + b.tokens, 0);
  const groups = BREAKDOWN_GROUPS.map((g) => {
    const children = items.filter((it) => it.group === g.id).sort((a, b) => b.tokens - a.tokens);
    const tokens = children.reduce((a, b) => a + b.tokens, 0);
    return { ...g, tokens, share: total ? tokens / total : 0, children: children.map((c) => ({ ...c, share: total ? c.tokens / total : 0 })) };
  }).filter((g) => g.tokens > 0).sort((a, b) => b.tokens - a.tokens);
  const top = [...items].sort((a, b) => b.tokens - a.tokens)[0] || null;
  return { total, rawTotal, trimmed, groups, top: top ? { ...top, share: total ? top.tokens / total : 0 } : null };
}

export function buildBreakdown(u) {
  if (!u || typeof u !== 'object' || u.v !== 1) return null;
  const tools = Object.entries(u.tools || {}).map(([name, t]) => ({
    name,
    calls: Number(t.calls) || 0,
    resultTokens: Number(t.result_tokens) || 0,
    argTokens: Number(t.arg_tokens) || 0,
    resultChars: Number(t.result_chars) || 0,
    argChars: Number(t.arg_chars) || 0,
    errors: Number(t.errors) || 0,
  })).sort((a, b) => (b.resultTokens + b.argTokens) - (a.resultTokens + a.argTokens) || b.calls - a.calls);
  const skills = Object.entries(u.skills || {}).map(([name, s]) => ({
    name, loads: Number(s.loads) || 0, resourceUses: Number(s.resource_uses) || 0,
    firstStep: s.first_step ?? null, lastStep: s.last_step ?? null,
  })).sort((a, b) => (b.loads + b.resourceUses) - (a.loads + a.resourceUses));
  const mcp = Object.entries(u.mcp || {}).map(([key, n]) => ({ key, calls: Number(n) || 0 })).sort((a, b) => b.calls - a.calls);
  return {
    window: { prompt: u.window?.prompt ?? null, sinceStep: u.window?.since_step ?? null, ...breakdownScope(u.window?.cats, u.window?.prompt) },
    total: breakdownScope(u.total?.cats),
    compactions: Number(u.compactions) || 0,
    tools,
    totalCalls: tools.reduce((a, b) => a + b.calls, 0),
    skills,
    mcp,
  };
}

export function buildContextSnapshot({
  convId,
  appDataDirs = findAppDataDirs(),
  guardDataDir = getGuardDataDir(),
  cache = new Map(),
  hooksJsonPath = undefined,
} = {}) {
  if (!convId || !ID_RE.test(convId)) {
    return { conversationId: convId || null, found: false, reason: 'invalid_or_empty_conversation' };
  }
  const files = resolveConversationFiles(convId, appDataDirs);
  if (!files || !files.transcriptPath) {
    return { conversationId: convId, found: false, reason: 'transcript_not_found' };
  }

  const metrics = parseTranscriptIncremental(files.transcriptPath, cache);
  if (!metrics) {
    return { conversationId: convId, found: false, reason: 'transcript_unreadable' };
  }

  const dbModels = readStepModels(files.dbPath, cache);
  const modelsSummary = aggregateModels(metrics.stepEvents, dbModels.map, metrics.compactionCount, metrics.totals, metrics.rewound);

  const guardWrap = readGuardState(guardDataDir, convId);
  const g = guardWrap?.data || null;
  const taskState = readTaskStateInfo(files.convDir, g?.artifact_dir);

  const tokens = metrics.tokens ?? g?.tokens ?? null;
  const chars = metrics.chars;
  const levels = resolveLevels(g);
  const liveLevel = computeThresholdLevel(tokens, chars, levels);
  const hookConfig = checkHookConfig(hooksJsonPath);
  // Hook liveness: guard.py only runs on PreInvocation / PreToolUse, so a slow
  // tool, background notification, or idle turn can leave transcriptMtime > state mtime
  // by >3 min with at most 1 PLANNER_RESPONSE after guard's byte offset. Only mark
  // stale when 2+ model invocations happened after guard's offset (or no offset exists).
  const hookLagMs = guardWrap ? Math.max(0, metrics.transcriptMtime - guardWrap.mtimeMs) : null;
  let unseenPlanners = 0;
  let hookStale = false;
  if (!guardWrap) {
    hookStale = metrics.lastStepIndex > 10;
  } else if (typeof g?.offset === 'number' && Number.isFinite(g.offset)) {
    const effectiveOffset = g.offset <= metrics.transcriptSize ? Math.max(0, g.offset) : 0;
    const offsets = metrics.plannerStartOffsets || [];
    for (let i = offsets.length - 1; i >= 0; i -= 1) {
      if (offsets[i] >= effectiveOffset) {
        unseenPlanners += 1;
      } else {
        break;
      }
    }
    hookStale = hookLagMs > HOOK_STALE_MS && unseenPlanners >= 2;
  } else {
    hookStale = hookLagMs > HOOK_STALE_MS;
  }
  const isRecovery = Boolean(g?.pending) && !hookStale;
  const effectiveLevel = isRecovery ? 3 : Math.max(liveLevel, hookStale ? 0 : Number(g?.warned || 0));

  const lv1Txt = tokens !== null
    ? `≥${Math.round(levels.token[0][0] / 1000)}k`
    : `≥${Math.round(levels.char[0][0] / 1000)}k 字符`;
  const lv2Txt = tokens !== null
    ? `≥${Math.round(levels.token[1][0] / 1000)}k`
    : `≥${Math.round(levels.char[1][0] / 1000)}k 字符`;
  const levelLabels = {
    0: '正常',
    1: `一级预警 (${lv1Txt})`,
    2: `二级紧急 (${lv2Txt})`,
    3: '压缩恢复中',
  };

  const percent = tokens !== null
    ? Math.min(100, Math.round((tokens / levels.target) * 1000) / 10)
    : Math.min(100, Math.round((chars / levels.char[levels.char.length - 1][0]) * 1000) / 10);

  const trendStats = computeTrend(metrics.series, tokens, levels.target);
  const etag = [
    metrics.transcriptMtime,
    metrics.transcriptSize,
    dbModels.mtime,
    dbModels.size,
    guardWrap?.mtimeMs || 0,
    taskState.mtime || 0,
    hookConfig.ok ? 1 : 0,
  ].join(':');

  const hasLiveUsage = Boolean(
    metrics.usage && (
      metrics.usage.window.prompt
      || Object.keys(metrics.usage.total.cats).length
      || Object.keys(metrics.usage.tools).length
    )
  );
  const breakdown = buildBreakdown(hasLiveUsage ? metrics.usage : g?.usage);

  return {
    conversationId: convId,
    found: true,
    etag,
    usage: {
      tokens,
      inputTokens: metrics.inputTokens ?? g?.input_tokens ?? null,
      cacheReadTokens: metrics.cacheReadTokens ?? g?.cache_read_tokens ?? null,
      outputTokens: metrics.outputTokens ?? g?.output_tokens ?? null,
      cacheHitRate: metrics.cacheHitRate ?? g?.cache_hit_rate ?? null,
      lastTps: modelsSummary.lastTps,
      lastDurationSec: modelsSummary.lastDurationSec,
      overTarget: tokens !== null && tokens > levels.target,
      chars,
      targetTokens: levels.target,
      compactionRange: levels.range,
      levelsModel: levels.model,
      levelsSource: levels.source,
      warnTokenLevel1: levels.token[0][0],
      warnTokenLevel2: levels.token[1][0],
      warnCharLevel1: levels.char[0][0],
      warnCharLevel2: levels.char[1][0],
      percent,
      remainingToTarget: tokens !== null ? Math.max(0, levels.target - tokens) : null,
      source: tokens !== null ? 'tokens' : 'chars',
      cycle: metrics.cycle,
      totals: metrics.totals,
    },
    trend: {
      lastStepIndex: metrics.lastStepIndex,
      stepsSinceCheckpoint: metrics.stepsSinceCheckpoint,
      avgDeltaPerStep: trendStats.avgDeltaPerStep,
      lastDelta: trendStats.lastDelta,
      estimatedStepsLeft: trendStats.estimatedStepsLeft,
      avgOutputPerStep: trendStats.avgOutputPerStep,
      avgInputPerStep: trendStats.avgInputPerStep,
      recentCacheHitRate: trendStats.recentCacheHitRate,
      series: metrics.series,
    },
    models: modelsSummary,
    compaction: {
      count: Math.max(metrics.compactionCount, Number(g?.compaction_count || 0)),
      lastStep: metrics.lastCheckpointStep ?? g?.last_ckpt ?? null,
      checkpoints: metrics.checkpoints,
    },
    guard: {
      connected: Boolean(g),
      level: effectiveLevel,
      levelLabel: levelLabels[effectiveLevel] || '正常',
      warned: Number(g?.warned || 0),
      modelWarned: Number(g?.model_warned ?? g?.warned ?? 0),
      pending: g?.pending ?? null,
      lastEvent: g?.last_event || (isRecovery ? 'compacted' : effectiveLevel > 0 ? `warn_${effectiveLevel}` : 'normal'),
      updatedAt: g?.updated_at || null,
      stale: hookStale,
      lagMs: hookLagMs,
      unseenPlanners,
      config: hookConfig,
      manualCompaction: Boolean(g?.ckpt_manual),
      // 'ok' | 'failed' | null — whether the last recovery pack was seen in the transcript
      lastDelivery: g?.last_delivery ?? null,
      verifying: Boolean(g?.verify),
      levelsFromGuard: Boolean(g?.levels),
    },
    taskState,
    breakdown,
  };
}
