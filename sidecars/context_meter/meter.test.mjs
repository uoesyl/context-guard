import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, appendFileSync, rmSync, utimesSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import {
  buildBreakdown,
  buildContextSnapshot,
  checkHookConfig,
  computeThresholdLevel,
  parseTranscriptIncremental,
  resolveLevels,
} from './meter.mjs';

test('buildBreakdown groups categories, ranks tools and drops trimmed/zero entries', () => {
  assert.equal(buildBreakdown(null), null);
  assert.equal(buildBreakdown({ v: 2 }), null);
  const u = {
    v: 1,
    window: {
      prompt: 1000,
      since_step: 7,
      cats: {
        'tool:read_file': { chars: 900, tokens: 500, n: 3 },
        'tool:run_command': { chars: 300, tokens: 100, n: 2 },
        'args:edit_file': { chars: 400, tokens: 200, n: 1 },
        baseline: { chars: 0, tokens: 150, n: 1 },
        'model:thinking': { chars: 80, tokens: 50, n: 2 },
        trimmed: { chars: 0, tokens: -40, n: 1 },
        user_input: { chars: 0, tokens: 0, n: 1 },
      },
    },
    total: { cats: { 'tool:read_file': { chars: 900, tokens: 500, n: 3 } } },
    tools: {
      edit_file: { calls: 4, arg_tokens: 200, result_tokens: 10 },
      read_file: { calls: 3, result_tokens: 500, arg_tokens: 5 },
    },
    skills: { pdf: { loads: 1, resource_uses: 2, first_step: 3, last_step: 9 } },
    mcp: { 'blender/get_objects_summary': 2 },
    compactions: 1,
  };
  const b = buildBreakdown(u);
  assert.equal(b.window.total, 1000);
  assert.deepEqual(b.window.groups.map((g) => g.id), ['tool', 'args', 'baseline', 'model']);
  assert.equal(b.window.groups[0].tokens, 600);
  assert.equal(b.window.groups[0].children[0].label, 'read_file');
  assert.equal(b.window.top.cat, 'tool:read_file');
  assert.ok(Math.abs(b.window.top.share - 0.5) < 1e-9);
  assert.equal(b.window.sinceStep, 7);
  assert.deepEqual(b.tools.map((t) => t.name), ['read_file', 'edit_file']);
  assert.equal(b.totalCalls, 7);
  assert.deepEqual(b.skills[0], { name: 'pdf', loads: 1, resourceUses: 2, firstStep: 3, lastStep: 9 });
  assert.deepEqual(b.mcp, [{ key: 'blender/get_objects_summary', calls: 2 }]);
});

test('buildBreakdown scales window categories to real prompt when trimmed is negative', () => {
  const u = {
    v: 1,
    window: {
      prompt: 2000,
      since_step: 10,
      cats: {
        'tool:read_file': { chars: 3000, tokens: 1500, n: 2 },
        'args:edit_file': { chars: 1500, tokens: 900, n: 1 },
        baseline: { chars: 0, tokens: 600, n: 1 },
        trimmed: { chars: 0, tokens: -1000, n: 1 },
      },
    },
    total: {
      cats: {
        'tool:read_file': { chars: 3000, tokens: 1500, n: 2 },
        'args:edit_file': { chars: 1500, tokens: 900, n: 1 },
        baseline: { chars: 0, tokens: 600, n: 1 },
        trimmed: { chars: 0, tokens: -1000, n: 1 },
      },
    },
    tools: {},
    skills: {},
    mcp: {},
    compactions: 0,
  };
  const b = buildBreakdown(u);
  assert.ok(Math.abs(b.window.total - 2000) <= 2, `window total ${b.window.total} should match prompt 2000`);
  assert.equal(b.window.rawTotal, 3000);
  assert.equal(b.window.trimmed, -1000);
  assert.equal(b.window.groups[0].tokens, 1000); // 1500 * (2/3)
  assert.ok(Math.abs(b.window.groups[0].share - 0.5) < 1e-6);
  // Cumulative total stays unscaled
  assert.equal(b.total.total, 3000);
});

test('resolveLevels prefers guard state levels and falls back to defaults', () => {
  const d = resolveLevels(null);
  assert.deepEqual(d.token, [[350_000, 1], [420_000, 2]]);
  assert.equal(d.target, 512_000);
  const g = { levels: { token: [[100, 1], [200, 2]], char: [[10, 1], [20, 2]], target: 300, range: [250, 310] } };
  const L = resolveLevels(g);
  assert.equal(L.target, 300);
  assert.equal(computeThresholdLevel(150, 0, L), 1);
  assert.equal(computeThresholdLevel(250, 0, L), 2);
  assert.deepEqual(resolveLevels({ levels: { token: 'bad' } }).token, d.token);
  assert.equal(d.model, null);
  const m = resolveLevels({ levels: { ...g.levels, model: 'claude-opus-4', source: 'db_limit+seen' } });
  assert.equal(m.model, 'claude-opus-4');
  assert.equal(m.source, 'db_limit+seen');
});

test('parseTranscriptIncremental handles checkpoints and incremental appends', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxmeter-'));
  try {
    const p = join(root, 'transcript_full.jsonl');
    writeFileSync(p, [
      JSON.stringify({ step_index: 1, type: 'USER_INPUT', content: 'hello' }),
      JSON.stringify({ step_index: 2, type: 'PLANNER_RESPONSE', input_tokens: 12000, cache_read_tokens: 3000, output_tokens: 400, content: 'ok' }),
    ].join('\n') + '\n');

    const cache = new Map();
    const m1 = parseTranscriptIncremental(p, cache);
    assert.equal(m1.tokens, 15000);
    assert.equal(m1.inputTokens, 12000);
    assert.equal(m1.cacheReadTokens, 3000);
    assert.equal(m1.outputTokens, 400);
    assert.equal(m1.cacheHitRate, 0.2);
    assert.equal(m1.cycle.promptTokens, 15000);
    assert.equal(m1.totals.promptTokens, 15000);
    assert.equal(m1.compactionCount, 0);
    assert.equal(m1.series.length, 1);

    // Append checkpoint + post-compaction step
    appendFileSync(p, [
      JSON.stringify({ step_index: 3, type: 'CHECKPOINT', content: '# Resuming', created_at: '2026-11-15T10:00:00Z' }),
      JSON.stringify({ step_index: 4, type: 'PLANNER_RESPONSE', input_tokens: 25000, cache_read_tokens: 5000, output_tokens: 200 }),
    ].join('\n') + '\n');

    const m2 = parseTranscriptIncremental(p, cache);
    assert.equal(m2.compactionCount, 1);
    assert.equal(m2.lastCheckpointStep, 3);
    assert.equal(m2.tokens, 30000);
    assert.equal(m2.cycle.promptTokens, 30000);
    assert.equal(m2.cycle.outputTokens, 200);
    assert.equal(m2.totals.promptTokens, 45000);
    assert.equal(m2.totals.cacheReadTokens, 8000);
    assert.equal(m2.totals.outputTokens, 600);
    assert.equal(m2.series.length, 1);
    assert.equal(m2.stepsSinceCheckpoint, 1);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
test('buildContextSnapshot fuses transcript, guard state, and task_state.md', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxsnap-'));
  try {
    const convId = '12345678-1234-1234-1234-123456789abc';
    const appData = join(root, 'antigravity');
    const convDir = join(appData, 'brain', convId);
    const logsDir = join(convDir, '.system_generated', 'logs');
    const guardDir = join(root, 'guard_data');
    mkdirSync(logsDir, { recursive: true });
    mkdirSync(join(guardDir, 'state'), { recursive: true });

    writeFileSync(join(logsDir, 'transcript_full.jsonl'), [
      JSON.stringify({ step_index: 1, type: 'PLANNER_RESPONSE', input_tokens: 350000, cache_read_tokens: 0, output_tokens: 300 }),
      JSON.stringify({ step_index: 2, type: 'PLANNER_RESPONSE', input_tokens: 365000, cache_read_tokens: 0, output_tokens: 350 }),
    ].join('\n') + '\n');

    writeFileSync(join(guardDir, 'state', `${convId}.json`), JSON.stringify({
      warned: 1,
      model_warned: 1,
      compaction_count: 0,
      last_event: 'warn_1',
      updated_at: 1700000000,
    }));

    writeFileSync(join(convDir, 'task_state.md'), '# 目标\ntest');

    const snap = buildContextSnapshot({
      convId,
      appDataDirs: [appData],
      guardDataDir: guardDir,
      cache: new Map(),
    });

    assert.equal(snap.found, true);
    assert.equal(snap.usage.tokens, 365000);
    assert.equal(snap.usage.inputTokens, 365000);
    assert.equal(snap.usage.cacheReadTokens, 0);
    assert.equal(snap.usage.outputTokens, 350);
    assert.equal(snap.usage.cacheHitRate, 0);
    assert.equal(snap.usage.overTarget, false);
    assert.equal(snap.usage.totals.cacheMisses, 2);
    assert.equal(snap.usage.totals.outputTokens, 650);
    assert.equal(snap.trend.avgOutputPerStep, 325);
    assert.equal(snap.usage.source, 'tokens');
    assert.equal(snap.guard.level, 1);
    assert.equal(snap.guard.connected, true);
    assert.equal(snap.trend.lastDelta, 15000);
    assert.equal(snap.taskState.exists, true);
    assert.match(snap.taskState.uri, /^file:\/\//);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
test('buildContextSnapshot rejects path traversal and handles char fallback', () => {
  const bad = buildContextSnapshot({ convId: '../../etc/passwd', appDataDirs: [], guardDataDir: '' });
  assert.equal(bad.found, false);
  assert.equal(computeThresholdLevel(null, 720000), 1);
  assert.equal(computeThresholdLevel(430000, 100), 2);
});

test('checkHookConfig flags BOM, bad JSON and missing guard events', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxmeter-hooks-'));
  try {
    const good = {
      'context-guard': {
        PreInvocation: [{ type: 'command', command: 'python x/guard.py' }],
        PreToolUse: [{ matcher: '*', hooks: [{ type: 'command', command: 'python x/guard.py' }] }],
      },
    };
    const p1 = join(root, 'ok.json');
    writeFileSync(p1, JSON.stringify(good));
    assert.equal(checkHookConfig(p1).ok, true);

    const p2 = join(root, 'bom.json');
    writeFileSync(p2, '\uFEFF' + JSON.stringify(good));
    const r2 = checkHookConfig(p2);
    assert.equal(r2.ok, false);
    assert.match(r2.issues.join(), /BOM/);

    const p3 = join(root, 'missing.json');
    writeFileSync(p3, JSON.stringify({ 'context-guard': { PreInvocation: good['context-guard'].PreInvocation } }));
    assert.match(checkHookConfig(p3).issues.join(), /PreToolUse/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
test('buildContextSnapshot marks the hook stale when guard state lags the transcript', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxmeter-stale-'));
  try {
    const appData = join(root, 'antigravity');
    const conv = 'abcdef12-0000-0000-0000-000000000001';
    const logs = join(appData, 'brain', conv, '.system_generated', 'logs');
    mkdirSync(logs, { recursive: true });
    const tp = join(logs, 'transcript_full.jsonl');
    writeFileSync(tp, JSON.stringify({ step_index: 1, type: 'PLANNER_RESPONSE', input_tokens: 1000 }) + '\n');
    const guardDir = join(root, 'guard');
    mkdirSync(join(guardDir, 'state'), { recursive: true });
    const sp = join(guardDir, 'state', `${conv}.json`);
    writeFileSync(sp, JSON.stringify({ pending: 5, warned: 2 }));
    const old = new Date(Date.now() - 10 * 60 * 1000);
    utimesSync(sp, old, old);
    const hooks = join(root, 'hooks.json');
    writeFileSync(hooks, '{}');
    const snap = buildContextSnapshot({ convId: conv, appDataDirs: [appData], guardDataDir: guardDir, hooksJsonPath: hooks });
    assert.equal(snap.guard.stale, true);
    assert.equal(snap.guard.level, 0, 'stale pending/warned must not be shown as live');
    assert.equal(snap.guard.config.ok, false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
test('buildContextSnapshot does not false-alarm on >3m lag when <=1 PLANNER_RESPONSE follows guard offset', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxmeter-slowstep-'));
  try {
    const appData = join(root, 'antigravity');
    const conv = 'abcdef12-0000-0000-0000-000000000002';
    const logs = join(appData, 'brain', conv, '.system_generated', 'logs');
    mkdirSync(logs, { recursive: true });
    const tp = join(logs, 'transcript_full.jsonl');
    const step0 = JSON.stringify({ step_index: 0, type: 'USER_INPUT', content: 'run slow task' }) + '\n';
    const step1 = JSON.stringify({ step_index: 1, type: 'PLANNER_RESPONSE', input_tokens: 360000, cache_read_tokens: 0, output_tokens: 100 }) + '\n';
    const step2 = JSON.stringify({ step_index: 2, type: 'SYSTEM_MESSAGE', content: 'background task done' }) + '\n';
    writeFileSync(tp, step0 + step1 + step2);

    const guardDir = join(root, 'guard');
    mkdirSync(join(guardDir, 'state'), { recursive: true });
    const sp = join(guardDir, 'state', `${conv}.json`);
    // Guard ran at PreInvocation right after step0 (before step1 was written), then >3m elapsed.
    writeFileSync(sp, JSON.stringify({ offset: Buffer.byteLength(step0), warned: 2, model_warned: 2 }));
    const old = new Date(Date.now() - 10 * 60 * 1000);
    utimesSync(sp, old, old);

    const hooks = join(root, 'hooks.json');
    writeFileSync(hooks, JSON.stringify({
      'context-guard': {
        PreInvocation: [{ type: 'command', command: 'python guard.py' }],
        PreToolUse: [{ matcher: '*', hooks: [{ type: 'command', command: 'python guard.py' }] }],
      },
    }));

    const cache = new Map();
    const snapOk = buildContextSnapshot({ convId: conv, appDataDirs: [appData], guardDataDir: guardDir, hooksJsonPath: hooks, cache });
    assert.equal(snapOk.guard.stale, false);
    assert.equal(snapOk.guard.unseenPlanners, 1);
    assert.equal(snapOk.guard.level, 2);

    // Once a 2nd PLANNER_RESPONSE runs without guard advancing offset, it IS stale.
    const step3 = JSON.stringify({ step_index: 3, type: 'USER_INPUT', content: 'next' }) + '\n';
    const step4 = JSON.stringify({ step_index: 4, type: 'PLANNER_RESPONSE', input_tokens: 370000, cache_read_tokens: 0, output_tokens: 100 }) + '\n';
    appendFileSync(tp, step3 + step4);
    const snapStale = buildContextSnapshot({ convId: conv, appDataDirs: [appData], guardDataDir: guardDir, hooksJsonPath: hooks, cache });
    assert.equal(snapStale.guard.stale, true);
    assert.equal(snapStale.guard.unseenPlanners, 2);
    assert.equal(snapStale.guard.level, 1, 'stale g.warned=2 suppressed to liveLevel=1');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
function pbVarint(n) {
  const bytes = [];
  let v = BigInt(n);
  while (v >= 0x80n) {
    bytes.push(Number((v & 0x7fn) | 0x80n));
    v >>= 7n;
  }
  bytes.push(Number(v));
  return Buffer.from(bytes);
}

function pbBytes(fn, buf) {
  return Buffer.concat([pbVarint((fn << 3) | 2), pbVarint(buf.length), buf]);
}

function pbInt(fn, n) {
  return Buffer.concat([pbVarint((fn << 3) | 0), pbVarint(n)]);
}

function pbTs(sec, nanos = 0) {
  return Buffer.concat([pbInt(1, sec), ...(nanos ? [pbInt(2, nanos)] : [])]);
}

function pbStepMeta(uuid, t1, t6, t7) {
  return Buffer.concat([
    pbBytes(1, pbTs(t1)),
    pbBytes(6, pbTs(t6)),
    pbBytes(7, pbTs(t7)),
    pbBytes(15, Buffer.from(`meta:${uuid}`)),
  ]);
}

test('buildContextSnapshot aggregates per-model stats from conversations DB and preserves full-session totals', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxmodels-'));
  try {
    const appData = join(root, 'antigravity');
    const convId = '11111111-2222-3333-4444-555555555555';
    const logsDir = join(appData, 'brain', convId, '.system_generated', 'logs');
    const convsDir = join(appData, 'conversations');
    mkdirSync(logsDir, { recursive: true });
    mkdirSync(convsDir, { recursive: true });

    writeFileSync(join(logsDir, 'transcript_full.jsonl'), [
      JSON.stringify({ step_index: 1, type: 'PLANNER_RESPONSE', input_tokens: 10000, cache_read_tokens: 0, output_tokens: 500, created_at: '2026-11-16T04:00:00Z' }),
      JSON.stringify({ step_index: 3, type: 'PLANNER_RESPONSE', input_tokens: 2000, cache_read_tokens: 10000, output_tokens: 300, created_at: '2026-11-16T04:01:00Z' }),
      JSON.stringify({ step_index: 4, type: 'CHECKPOINT', content: '# Resuming' }),
      JSON.stringify({ step_index: 5, type: 'PLANNER_RESPONSE', input_tokens: 4000, cache_read_tokens: 16000, output_tokens: 800, created_at: '2026-11-16T04:02:00Z' }),
    ].join('\n') + '\n');

    const db = new DatabaseSync(join(convsDir, `${convId}.db`));
    db.exec('CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB);');
    db.exec('CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, metadata BLOB);');

    const u1 = 'aaaaaaaa-0000-0000-0000-000000000001';
    const u2 = 'aaaaaaaa-0000-0000-0000-000000000002';
    const u3 = 'aaaaaaaa-0000-0000-0000-000000000003';

    const gm1 = Buffer.concat([pbBytes(1, Buffer.concat([pbInt(3, 1525), pbBytes(19, Buffer.from('gemini-2.5-pro'))])), pbBytes(4, Buffer.from(u1))]);
    const gm2 = Buffer.concat([pbBytes(3, Buffer.concat([pbInt(1, 1525), pbInt(6, 524288), pbBytes(28, Buffer.from('gemini-2.5-pro-high'))])), pbBytes(4, Buffer.from(u2))]);
    const gm3 = Buffer.concat([pbBytes(3, Buffer.concat([pbInt(1, 1669), pbInt(6, 524288), pbBytes(28, Buffer.from('gemini-2.5-flash'))])), pbBytes(4, Buffer.from(u3))]);

    db.prepare('INSERT INTO gen_metadata (idx, data) VALUES (?, ?)').run(0, gm1);
    db.prepare('INSERT INTO gen_metadata (idx, data) VALUES (?, ?)').run(1, gm2);
    db.prepare('INSERT INTO gen_metadata (idx, data) VALUES (?, ?)').run(2, gm3);

    // step 1: 500 out in 5s (100 tok/s, TTFT 2s); step 3: 300 out in 3s (100 tok/s, TTFT 1s); step 5: 800 out in 4s (200 tok/s, TTFT 1s)
    db.prepare('INSERT INTO steps (idx, step_type, metadata) VALUES (?, 15, ?)').run(1, pbStepMeta(u1, 1794800000, 1794800002, 1794800005));
    db.prepare('INSERT INTO steps (idx, step_type, metadata) VALUES (?, 15, ?)').run(3, pbStepMeta(u2, 1794800060, 1794800061, 1794800063));
    db.prepare('INSERT INTO steps (idx, step_type, metadata) VALUES (?, 15, ?)').run(5, pbStepMeta(u3, 1794800120, 1794800121, 1794800124));
    db.close();

    const snap = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: join(root, 'g') });
    assert.equal(snap.found, true);
    assert.equal(snap.models.totals.modelCount, 2);
    assert.equal(snap.models.totals.steps, 3);
    assert.equal(snap.models.totals.inputTokens, 16000);
    assert.equal(snap.models.totals.cacheReadTokens, 26000);
    assert.equal(snap.models.totals.outputTokens, 1600);
    assert.equal(snap.models.totals.cacheMisses, 1);
    assert.equal(snap.models.totals.totalDurationSec, 12);
    assert.equal(snap.models.totals.tps, 133.3);
    assert.equal(snap.usage.lastTps, 200);
    assert.equal(snap.models.activeModel.name, 'gemini-2.5-flash');

    const h2 = snap.models.list.find((m) => m.modelEnum === 1525);
    assert.ok(h2);
    assert.equal(h2.name, 'gemini-2.5-pro');
    assert.deepEqual(h2.variants.sort(), ['gemini-2.5-pro', 'gemini-2.5-pro-high']);
    assert.equal(h2.steps, 2);
    assert.equal(h2.inputTokens, 12000);
    assert.equal(h2.cacheReadTokens, 10000);
    assert.equal(h2.outputTokens, 800);
    assert.equal(h2.cacheMisses, 1);
    assert.equal(h2.tps, 100);
    assert.equal(h2.avgTtftSec, 1.5);
    assert.equal(h2.totalDurationSec, 8);
    assert.equal(h2.cycle.steps, 0);
    assert.equal(h2.stepDetails.length, 2);

    const w = snap.models.list.find((m) => m.modelEnum === 1669);
    assert.ok(w);
    assert.equal(w.steps, 1);
    assert.equal(w.cycle.steps, 1);
    assert.equal(w.cacheHitRate, 0.8);
    assert.equal(w.tps, 200);
    assert.equal(w.avgTtftSec, 1);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('buildContextSnapshot computes breakdown from transcript without guard usage', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxlivebd-'));
  try {
    const convId = 'abcdef12-3456-7890-abcd-ef1234567890';
    const appData = join(root, 'antigravity');
    const logsDir = join(appData, 'brain', convId, '.system_generated', 'logs');
    const guardDir = join(root, 'guard_data');
    mkdirSync(logsDir, { recursive: true });
    mkdirSync(join(guardDir, 'state'), { recursive: true });

    writeFileSync(join(guardDir, 'state', `${convId}.json`), JSON.stringify({
      warned: 0,
      model_warned: 0,
      offset: 0,
      updated_at: Date.now() / 1000,
    }));

    const skillPath = 'C:\\Users\\x\\.gemini\\config\\skills\\pdf\\SKILL.md';
    const transcriptPath = join(logsDir, 'transcript_full.jsonl');
    writeFileSync(transcriptPath, [
      JSON.stringify({ step_index: 0, type: 'USER_INPUT', content: 'u'.repeat(300) }),
      JSON.stringify({
        step_index: 1,
        type: 'PLANNER_RESPONSE',
        model: 'm',
        input_tokens: 10000,
        thinking: 't'.repeat(60),
        tool_calls: [
          { name: 'read_file', args: { Path: skillPath } },
          { name: 'call_mcp_tool', args: { ServerName: 'blender', ToolName: 'get_objects_summary' } },
        ],
      }),
      JSON.stringify({ step_index: 2, type: 'GENERIC', content: 'S'.repeat(3000) }),
      JSON.stringify({ step_index: 3, type: 'GENERIC', content: 'M'.repeat(1000) }),
      JSON.stringify({ step_index: 4, type: 'PLANNER_RESPONSE', model: 'm', input_tokens: 12000, content: 'ok' }),
    ].join('\n') + '\n');

    const cache = new Map();
    const s1 = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: guardDir, cache });
    assert.ok(s1.breakdown);
    assert.equal(s1.breakdown.window.total, 12000);
    assert.deepEqual(s1.breakdown.skills.map((s) => ({ name: s.name, loads: s.loads })), [{ name: 'pdf', loads: 1 }]);
    assert.deepEqual(s1.breakdown.mcp, [{ key: 'blender/get_objects_summary', calls: 1 }]);
    assert.equal(s1.breakdown.tools[0].name, 'read_file');

    appendFileSync(transcriptPath, [
      JSON.stringify({ step_index: 5, type: 'CHECKPOINT', content: 'c'.repeat(900) }),
      JSON.stringify({ step_index: 6, type: 'PLANNER_RESPONSE', model: 'm', input_tokens: 3000 }),
    ].join('\n') + '\n');

    const s2 = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: guardDir, cache });
    assert.equal(s2.breakdown.compactions, 1);
    assert.equal(s2.breakdown.window.total, 3000);
    assert.ok(s2.breakdown.total.total > s2.breakdown.window.total);
    assert.equal(s2.breakdown.skills[0].name, 'pdf');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('models: one row per transcript model id; rewound branch is dropped, not double counted', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxrewind-'));
  try {
    const appData = join(root, 'antigravity');
    const convId = '11111111-2222-3333-4444-666666666666';
    const logsDir = join(appData, 'brain', convId, '.system_generated', 'logs');
    const convsDir = join(appData, 'conversations');
    mkdirSync(logsDir, { recursive: true });
    mkdirSync(convsDir, { recursive: true });
    const pr = (s, m, inp, out) => JSON.stringify({ step_index: s, type: 'PLANNER_RESPONSE', model: m, input_tokens: inp, cache_read_tokens: 0, output_tokens: out });
    writeFileSync(join(logsDir, 'transcript_full.jsonl'), [
      JSON.stringify({ step_index: 0, type: 'USER_INPUT', content: 'first try' }),
      pr(1, 'gemini-2.5-pro-high', 1000, 10),
      JSON.stringify({ step_index: 2, type: 'USER_INPUT', content: 'more' }),
      pr(3, 'gemini-2.5-pro-xhigh', 2000, 20), // superseded below (no DB metadata for it)
      JSON.stringify({ step_index: 2, type: 'USER_INPUT', content: 'edited' }), // rewind to step 2
      pr(3, 'gemini-2.5-pro-high', 1500, 15),
      pr(4, 'gemini-2.5-pro-high', 1600, 16), // no DB metadata yet (WAL lag)
    ].join('\n') + '\n');

    const db = new DatabaseSync(join(convsDir, `${convId}.db`));
    db.exec('CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB);');
    db.exec('CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, metadata BLOB);');
    const u1 = 'bbbbbbbb-0000-0000-0000-000000000001';
    // DB name lacks the reasoning suffix, exactly like the real gen_metadata
    const gm = Buffer.concat([pbBytes(1, Buffer.concat([pbInt(3, 1525), pbBytes(19, Buffer.from('gemini-2.5-pro'))])), pbBytes(4, Buffer.from(u1))]);
    db.prepare('INSERT INTO gen_metadata (idx, data) VALUES (?, ?)').run(0, gm);
    db.prepare('INSERT INTO steps (idx, step_type, metadata) VALUES (?, 15, ?)').run(1, pbStepMeta(u1, 1794800000, 1794800001, 1794800002));
    db.prepare('INSERT INTO steps (idx, step_type, metadata) VALUES (?, 15, ?)').run(3, pbStepMeta(u1, 1794800010, 1794800011, 1794800012));
    db.close();

    const snap = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: join(root, 'g') });
    assert.equal(snap.models.list.length, 1, JSON.stringify(snap.models.list.map((m) => m.key)));
    const m = snap.models.list[0];
    assert.equal(m.key, 'name:gemini-2.5-pro-high');
    assert.equal(m.name, 'gemini-2.5-pro-high');
    assert.equal(m.modelEnum, 1525);
    assert.equal(m.steps, 3);
    assert.equal(snap.models.totals.steps, 3);
    assert.equal(snap.models.totals.inputTokens, 4100);
    assert.deepEqual(snap.models.totals.rewound, { count: 1, steps: 1, promptTokens: 2000, outputTokens: 20 });
    assert.equal(snap.usage.tokens, 1600);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('rewind resets usage breakdown and chars without fake negative trimmed or stale tools/skills', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxrewindusage-'));
  try {
    const appData = join(root, 'antigravity');
    const convId = '11111111-2222-3333-4444-777777777777';
    const logsDir = join(appData, 'brain', convId, '.system_generated', 'logs');
    mkdirSync(logsDir, { recursive: true });
    const skillPath = 'C:\\Users\\x\\.gemini\\config\\skills\\pdf\\SKILL.md';
    writeFileSync(join(logsDir, 'transcript_full.jsonl'), [
      JSON.stringify({ step_index: 0, type: 'USER_INPUT', content: 'start' }),
      JSON.stringify({ step_index: 1, type: 'PLANNER_RESPONSE', model: 'm', input_tokens: 10000, output_tokens: 50 }),
      JSON.stringify({ step_index: 2, type: 'USER_INPUT', content: 'discarded branch' }),
      JSON.stringify({
        step_index: 3,
        type: 'PLANNER_RESPONSE',
        model: 'm',
        input_tokens: 20000,
        output_tokens: 80,
        tool_calls: [{ name: 'read_file', args: { Path: skillPath } }],
      }),
      JSON.stringify({ step_index: 4, type: 'GENERIC', content: 'X'.repeat(750000) }),
      JSON.stringify({ step_index: 2, type: 'USER_INPUT', content: 'retried branch' }),
      JSON.stringify({ step_index: 3, type: 'PLANNER_RESPONSE', model: 'm', input_tokens: 12000, output_tokens: 60 }),
    ].join('\n') + '\n');

    const snap = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: join(root, 'g') });
    assert.equal(snap.breakdown.window.trimmed, 0);
    assert.equal(snap.breakdown.window.total, 12000);
    assert.equal(snap.breakdown.tools.length, 0);
    assert.equal(snap.breakdown.skills.length, 0);
    assert.ok(snap.usage.chars < 10000, `chars should exclude discarded 750k generic step, got ${snap.usage.chars}`);
    assert.equal(snap.guard.level, 0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('chars fallback mode uses char thresholds in guard.levelLabel and resolves valid artifact_dir', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxchars-'));
  try {
    const appData = join(root, 'antigravity');
    const convId = '11111111-2222-3333-4444-888888888888';
    const logsDir = join(appData, 'brain', convId, '.system_generated', 'logs');
    const guardDir = join(root, 'guard');
    const customArtDir = join(root, 'custom-artifacts');
    mkdirSync(logsDir, { recursive: true });
    mkdirSync(join(guardDir, 'state'), { recursive: true });
    mkdirSync(customArtDir, { recursive: true });

    const tp = join(logsDir, 'transcript_full.jsonl');
    writeFileSync(tp, JSON.stringify({ step_index: 0, type: 'USER_INPUT', content: 'A'.repeat(710000) }) + '\n');
    writeFileSync(join(customArtDir, 'task_state.md'), '# 目标\nok\n');
    writeFileSync(join(guardDir, 'state', `${convId}.json`), JSON.stringify({
      artifact_dir: customArtDir,
      updated_at: Date.now() / 1000,
    }));

    const snap1 = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: guardDir });
    assert.equal(snap1.usage.source, 'chars');
    assert.equal(snap1.guard.level, 1);
    assert.equal(snap1.guard.levelLabel, '一级预警 (≥700k 字符)');
    assert.equal(snap1.taskState.exists, true);

    appendFileSync(tp, JSON.stringify({ step_index: 1, type: 'GENERIC', content: 'B'.repeat(210000) }) + '\n');
    writeFileSync(join(guardDir, 'state', `${convId}.json`), JSON.stringify({
      artifact_dir: '../relative-invalid',
      updated_at: Date.now() / 1000,
    }));
    const snap2 = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: guardDir });
    assert.equal(snap2.guard.level, 2);
    assert.equal(snap2.guard.levelLabel, '二级紧急 (≥900k 字符)');
    assert.equal(snap2.taskState.exists, false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('call_mcp_tool falls back to unknown for missing ServerName or ToolName', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxmcp-'));
  try {
    const convId = '11111111-2222-3333-4444-999999999999';
    const appData = join(root, 'antigravity');
    const logsDir = join(appData, 'brain', convId, '.system_generated', 'logs');
    mkdirSync(logsDir, { recursive: true });
    writeFileSync(join(logsDir, 'transcript_full.jsonl'), [
      JSON.stringify({
        step_index: 1,
        type: 'PLANNER_RESPONSE',
        input_tokens: 1000,
        tool_calls: [
          { name: 'call_mcp_tool', args: {} },
          { name: 'call_mcp_tool', args: { ServerName: 'srv' } },
          { name: 'call_mcp_tool', args: { ToolName: 'fn' } },
        ],
      }),
    ].join('\n') + '\n');
    const snap = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: join(root, 'g') });
    assert.deepEqual(
      snap.breakdown.mcp.map((x) => [x.key, x.calls]).sort((a, b) => a[0].localeCompare(b[0])),
      [['srv/unknown', 1], ['unknown/fn', 1], ['unknown/unknown', 1]],
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('models.activity tracks cycle model activity, DB tEnd preference, hydrogen compact hint, and CHECKPOINT reset', () => {
  const root = mkdtempSync(join(tmpdir(), 'ctxact-'));
  try {
    const convId = '11111111-2222-3333-4444-aaaaaaaaaaaa';
    const appData = join(root, 'antigravity');
    const logsDir = join(appData, 'brain', convId, '.system_generated', 'logs');
    const convsDir = join(appData, 'conversations');
    mkdirSync(logsDir, { recursive: true });
    mkdirSync(convsDir, { recursive: true });

    const uuid1 = 'aaaaaaaa-1111-2222-3333-444444444444';
    const dbEndSec = 1795800500;
    const db = new DatabaseSync(join(convsDir, `${convId}.db`));
    db.exec('CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB);');
    db.exec('CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, metadata BLOB);');
    const gm1 = Buffer.concat([
      pbBytes(1, pbBytes(19, Buffer.from('gemini-4-hydrogen-high'))),
      pbBytes(4, Buffer.from(uuid1, 'ascii')),
    ]);
    db.prepare('INSERT INTO gen_metadata (idx, data) VALUES (?, ?)').run(1, gm1);
    const meta2 = pbStepMeta(uuid1, dbEndSec - 10, dbEndSec - 5, dbEndSec);
    db.prepare('INSERT INTO steps (idx, step_type, metadata) VALUES (?, ?, ?)').run(2, 15, meta2);
    db.close();

    const tp = join(logsDir, 'transcript_full.jsonl');
    writeFileSync(tp, JSON.stringify({
      step_index: 1,
      type: 'PLANNER_RESPONSE',
      model: 'gemini-4-hydrogen-high',
      input_tokens: 10000,
      cache_read_tokens: 90000,
      created_at: '2026-11-28T01:00:00Z',
    }) + '\n');
    const snapOnlyH = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: join(root, 'g') });
    assert.equal(snapOnlyH.models.activity.lastModel, 'gemini-4-hydrogen-high');
    assert.equal(snapOnlyH.models.activity.lastStepAtMs, Date.parse('2026-11-28T01:00:00Z'));
    assert.equal(snapOnlyH.models.activity.crossModelCompactRisk, false, 'hydrogen alone without planner in cycle');

    // Add a planner model first? Wait: test planner -> hydrogen in same cycle with DB tEnd on step 2
    writeFileSync(tp, [
      JSON.stringify({
        step_index: 1,
        type: 'PLANNER_RESPONSE',
        model: 'gemini-4-neon-xhigh',
        input_tokens: 10000,
        cache_read_tokens: 90000,
        created_at: '2026-11-28T01:00:00Z',
      }),
      JSON.stringify({
        step_index: 2,
        type: 'PLANNER_RESPONSE',
        model: 'gemini-4-hydrogen-high',
        input_tokens: 20000,
        cache_read_tokens: 80000,
        created_at: '2026-11-28T01:01:00Z',
      }),
    ].join('\n') + '\n');
    const snapSwitch = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: join(root, 'g') });
    assert.equal(snapSwitch.models.activity.lastModel, 'gemini-4-hydrogen-high');
    assert.equal(snapSwitch.models.activity.lastStepAtMs, dbEndSec * 1000, 'prefers DB tEnd over created_at');
    assert.equal(snapSwitch.models.activity.crossModelCompactRisk, true);

    appendFileSync(tp, JSON.stringify({
      step_index: 3,
      type: 'PLANNER_RESPONSE',
      model: 'gemini-4-neon-xhigh',
      input_tokens: 5000,
      cache_read_tokens: 95000,
      created_at: '2026-11-28T01:02:00Z',
    }) + '\n');
    const snapNeon = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: join(root, 'g') });
    assert.equal(snapNeon.models.activity.lastModel, 'gemini-4-neon-xhigh');
    assert.equal(snapNeon.models.activity.crossModelCompactRisk, false);

    appendFileSync(tp, JSON.stringify({
      step_index: 4,
      type: 'CHECKPOINT',
      content: '# Resuming',
      created_at: '2026-11-28T01:03:00Z',
    }) + '\n');
    const snapAfterCkpt = buildContextSnapshot({ convId, appDataDirs: [appData], guardDataDir: join(root, 'g') });
    assert.equal(snapAfterCkpt.models.activity.lastModel, null);
    assert.equal(snapAfterCkpt.models.activity.lastStepAtMs, null);
    assert.equal(snapAfterCkpt.models.activity.crossModelCompactRisk, false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('sidecar UI subtraction removes remind-state route, interactive buttons, and sets allows_user_interactions false', () => {
  const dir = new URL('.', import.meta.url);
  const sidecarJson = JSON.parse(readFileSync(new URL('./sidecar.json', dir), 'utf-8'));
  const mainSrc = readFileSync(new URL('./main.mjs', dir), 'utf-8');
  const htmlSrc = readFileSync(new URL('./index.html', dir), 'utf-8');
  const appSrc = readFileSync(new URL('./app.js', dir), 'utf-8');
  assert.equal(sidecarJson.agent_permissions?.allows_user_interactions, false);
  assert.equal(mainSrc.includes('/api/actions/remind-state'), false);
  assert.equal(mainSrc.includes('sendMessage'), false);
  assert.equal(htmlSrc.includes('btn-copy-compact'), false);
  assert.equal(htmlSrc.includes('btn-remind-ts'), false);
  assert.equal(htmlSrc.includes('cache-activity'), true);
  assert.equal(appSrc.includes('btn-open-ts'), true);
});
