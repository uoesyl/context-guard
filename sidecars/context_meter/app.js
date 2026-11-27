(() => {
  const POLL_MS = 1500;
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);

  const state = {
    etag: '',
    snap: null,
    selectedModelKey: null,
    showBreakdown: false,
    toastTimer: null,
    remindConfirmUntil: 0,
    compScope: 'window',
    compOpen: new Set(),
    compOpenInit: false,
    toolsAll: false,
  };

  function currentConv() {
    return window.sidecar?.conversationId || params.get('conversationId') || '';
  }

  function syncScheme() {
    const probe = document.createElement('span');
    probe.style.cssText = 'position:absolute;visibility:hidden;color:var(--background, #f7f7f8)';
    document.documentElement.appendChild(probe);
    const c = getComputedStyle(probe).color;
    probe.remove();
    const nums = (c.match(/[\d.]+/g) || []).map(Number);
    let theme;
    if (nums.length < 3) {
      theme = window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    } else {
      const scale = c.startsWith('color(') ? 1 : 255;
      const [r, g, b] = nums.slice(0, 3).map((v) => v / scale);
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      theme = lum < 0.5 ? 'dark' : 'light';
    }
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
  }
  syncScheme();
  new MutationObserver(syncScheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['style'],
  });

  async function apiFetch(path, options = {}) {
    const f = window.sidecar?.fetch ? window.sidecar.fetch.bind(window.sidecar) : fetch;
    const res = await f(path, options);
    if (!res.ok) throw new Error(`${path}: ${res.status}`);
    return res.json();
  }

  function fmtK(n) {
    if (n === null || n === undefined || !Number.isFinite(n)) return '--';
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
    return String(Math.round(n));
  }

  function fmtPct(r) {
    if (r === null || r === undefined || !Number.isFinite(r)) return '--';
    return `${(r * 100).toFixed(1)}%`;
  }

  function fmtTps(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return '--';
    return `${v.toFixed(1)} tok/s`;
  }

  function fmtSec(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return '--';
    return v >= 100 ? `${Math.round(v)}s` : `${v.toFixed(1)}s`;
  }

  function fmtAgo(ms) {
    if (!ms) return '';
    const diffSec = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (diffSec < 60) return `${diffSec} 秒前`;
    const mins = Math.round(diffSec / 60);
    if (mins < 60) return `${mins} 分钟前`;
    const hrs = Math.round(mins / 60);
    return `${hrs} 小时前`;
  }

  function showToast(msg) {
    const el = $('toast');
    el.textContent = msg;
    el.classList.remove('hidden');
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => el.classList.add('hidden'), 2400);
  }

  function hitClass(rate) {
    if (rate === null || rate === undefined || !Number.isFinite(rate)) return '';
    if (rate >= 0.85) return 'ok';
    if (rate >= 0.5) return 'warn';
    return 'crit';
  }

  function renderModelList(models, activeModel) {
    const listEl = $('model-list');
    while (listEl.firstChild) listEl.removeChild(listEl.firstChild);
    const items = models?.list || [];
    $('model-count-tag').textContent = items.length ? `${items.length} 模型` : '暂无数据';

    if (!items.length) {
      const empty = document.createElement('div');
      empty.className = 'memory-meta';
      empty.textContent = '尚无模型调用统计';
      listEl.appendChild(empty);
      return;
    }

    for (const mod of items) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'model-row';
      btn.dataset.modelKey = mod.key;

      const top = document.createElement('div');
      top.className = 'model-row-top';

      const nameWrap = document.createElement('div');
      nameWrap.className = 'model-name-wrap';
      const nameSpan = document.createElement('span');
      nameSpan.className = 'model-name';
      nameSpan.textContent = mod.name;
      nameWrap.appendChild(nameSpan);
      if (activeModel && activeModel.key === mod.key) {
        const badge = document.createElement('span');
        badge.className = 'active-tag';
        badge.textContent = '当前';
        nameWrap.appendChild(badge);
      }

      const right = document.createElement('div');
      right.className = 'model-row-right';
      const stepsTag = document.createElement('span');
      stepsTag.className = 'muted-tag';
      stepsTag.textContent = `${mod.steps} 次${mod.cacheMisses ? ` · ${mod.cacheMisses}归零` : ''}`;
      const hitPill = document.createElement('span');
      const hCls = hitClass(mod.cacheHitRate);
      hitPill.className = `hit-pill ${hCls}`;
      hitPill.textContent = fmtPct(mod.cacheHitRate);
      const chev = document.createElement('span');
      chev.className = 'chevron';
      chev.textContent = '›';
      right.appendChild(stepsTag);
      right.appendChild(hitPill);
      right.appendChild(chev);

      top.appendChild(nameWrap);
      top.appendChild(right);

      const barTrack = document.createElement('div');
      barTrack.className = 'hit-bar-track';
      const barFill = document.createElement('div');
      barFill.className = `hit-bar-fill ${hCls}`;
      barFill.style.width = `${Math.max(2, Math.min(100, (mod.cacheHitRate || 0) * 100))}%`;
      barTrack.appendChild(barFill);

      const metrics = document.createElement('div');
      metrics.className = 'model-row-metrics';
      const cols = [
        ['未命中', fmtK(mod.inputTokens), false],
        ['缓存读', fmtK(mod.cacheReadTokens), false],
        ['输出', fmtK(mod.outputTokens), false],
        ['TPS', mod.tps !== null && mod.tps !== undefined ? `${mod.tps.toFixed(1)}/s` : '--', true],
      ];
      for (const [labelTxt, valTxt, accent] of cols) {
        const cell = document.createElement('div');
        cell.className = 'model-metric';
        const kEl = document.createElement('span');
        kEl.className = 'model-metric-k';
        kEl.textContent = labelTxt;
        const vEl = document.createElement('span');
        vEl.className = `model-metric-v${accent ? ' accent' : ''}`;
        vEl.textContent = valTxt;
        cell.appendChild(kEl);
        cell.appendChild(vEl);
        metrics.appendChild(cell);
      }

      btn.appendChild(top);
      btn.appendChild(barTrack);
      btn.appendChild(metrics);
      btn.addEventListener('click', () => {
        state.showBreakdown = false;
        state.selectedModelKey = mod.key;
        window.scrollTo(0, 0);
        if (state.snap) render(state.snap);
      });
      listEl.appendChild(btn);
    }
  }

  function renderModelDetail(snap, mod) {
    $('dashboard').classList.add('hidden');
    $('breakdown-detail').classList.add('hidden');
    $('empty-state').classList.add('hidden');
    $('model-detail').classList.remove('hidden');

    $('detail-model-name').textContent = mod.name;
    const metaParts = [];
    if (mod.modelEnum !== null && mod.modelEnum !== undefined) metaParts.push(`enum:${mod.modelEnum}`);
    if (mod.contextLimit) metaParts.push(`上限 ${fmtK(mod.contextLimit)}`);
    if (mod.variants?.length > 1) metaParts.push(mod.variants.join(' / '));
    $('detail-model-meta').textContent = metaParts.join(' · ') || mod.key;

    const pill = $('detail-model-hit-pill');
    pill.className = `hit-pill ${hitClass(mod.cacheHitRate)}`;
    pill.textContent = `命中 ${fmtPct(mod.cacheHitRate)}`;

    $('detail-model-sub').textContent =
      `占全会话输入 ${fmtPct(mod.shareOfPrompt)} · 占全会话输出 ${fmtPct(mod.shareOfOutput)} · 步号 #${mod.firstStepIndex}–#${mod.lastStepIndex}`;

    $('detail-calls').textContent = `${mod.steps} 次 / ${mod.cycle?.steps || 0} 次`;
    $('detail-hit-miss').textContent = `${fmtPct(mod.cacheHitRate)} · ${mod.cacheMisses || 0} 次`;
    $('detail-in').textContent = `${fmtK(mod.inputTokens)} / 均 ${fmtK(mod.avgInputPerStep)}`;
    $('detail-cr').textContent = `${fmtK(mod.cacheReadTokens)} / 均 ${fmtK(mod.avgCacheReadPerStep)}`;
    $('detail-out').textContent = `${fmtK(mod.outputTokens)} / 均 ${fmtK(mod.avgOutputPerStep)}`;
    $('detail-ctx').textContent = `${fmtK(mod.maxPromptTokens)} / ${fmtK(mod.lastPromptTokens)}`;
    $('detail-tps').textContent = mod.tps !== null && mod.tps !== undefined
      ? `${fmtTps(mod.tps)}${mod.lastTps !== null && mod.lastTps !== undefined ? ` / 末次 ${mod.lastTps.toFixed(1)}` : ''}`
      : '--';
    $('detail-timing').textContent = mod.totalDurationSec > 0
      ? `${fmtSec(mod.avgTtftSec)} / ${fmtSec(mod.totalDurationSec)}`
      : '--';

    const steps = [...(mod.stepDetails || [])].reverse();
    $('detail-steps-count').textContent = `共 ${steps.length} 步`;
    const listEl = $('detail-steps-list');
    while (listEl.firstChild) listEl.removeChild(listEl.firstChild);

    for (const st of steps) {
      const row = document.createElement('div');
      const isMiss = st.promptTokens > 0 && st.cacheReadTokens === 0;
      const isLow = !isMiss && st.cacheHitRate < 0.5;
      row.className = `detail-step-row${isMiss ? ' miss' : isLow ? ' low' : ''}`;

      const top = document.createElement('div');
      top.className = 'detail-step-top';
      const left = document.createElement('span');
      const cycTxt = (snap.compaction?.count || 0) > 0 ? ` · 周期#${st.cycleIndex}` : '';
      const timeTxt = st.createdAt ? ` · ${new Date(st.createdAt).toLocaleTimeString()}` : '';
      left.textContent = `step #${st.step}${cycTxt}${timeTxt}`;

      const badges = document.createElement('div');
      badges.className = 'detail-step-badges';
      if (st.tps !== null && st.tps !== undefined) {
        const tpsEl = document.createElement('span');
        tpsEl.className = 'tps-pill';
        tpsEl.textContent = `${st.tps.toFixed(1)}/s`;
        badges.appendChild(tpsEl);
      }
      const hit = document.createElement('span');
      hit.className = `hit-pill ${hitClass(st.cacheHitRate)}`;
      hit.textContent = isMiss ? '0.0% 归零' : fmtPct(st.cacheHitRate);
      badges.appendChild(hit);
      top.appendChild(left);
      top.appendChild(badges);

      const sub = document.createElement('div');
      sub.className = 'detail-step-metrics';
      const stepCols = [
        `上下文 ${fmtK(st.promptTokens)}`,
        `未命中 ${fmtK(st.inputTokens)}`,
        `缓存 ${fmtK(st.cacheReadTokens)}`,
        `输出 ${fmtK(st.outputTokens)}`,
      ];
      if (st.durationSec !== null && st.durationSec !== undefined) {
        stepCols.push(`耗时 ${fmtSec(st.durationSec)}`);
      }
      for (const txt of stepCols) {
        const s = document.createElement('span');
        s.textContent = txt;
        sub.appendChild(s);
      }

      row.appendChild(top);
      row.appendChild(sub);
      listEl.appendChild(row);
    }
  }

  // ---------- Context composition ----------
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  const pct = (r) => (r > 0 && r < 0.0005 ? '<0.1%' : `${(r * 100).toFixed(r >= 0.1 ? 0 : 1)}%`);

  function setBarFocus(groupId) {
    const bar = $('comp-bar');
    bar.classList.toggle('dim', Boolean(groupId));
    for (const s of bar.children) s.classList.toggle('hot', s.dataset.group === groupId);
  }

  function renderComposition(bd) {
    const card = $('comp-card');
    if (!bd || (!bd.window?.total && !bd.total?.total)) {
      card.classList.add('hidden');
      return;
    }
    const scope = state.compScope === 'total' ? bd.total : bd.window;
    card.classList.remove('hidden');
    $('comp-scope-tag').textContent = state.compScope === 'total' ? '全对话' : '当前窗口';

    const head = $('comp-headline');
    const bar = $('comp-bar');
    const list = $('comp-legend');
    clear(head);
    clear(bar);
    clear(list);

    if (!scope || !scope.total) {
      head.textContent = state.compScope === 'window' && bd.compactions > 0
        ? '当前窗口刚压缩，等待首轮模型响应…'
        : '暂无数据';
      bar.setAttribute('aria-label', '上下文构成：暂无数据');
      $('comp-foot').textContent = state.compScope === 'window' && bd.compactions > 0
        ? `自第 ${bd.window.sinceStep ?? 0} 步起 · 可切换「全对话」查看累计记录`
        : '';
      return;
    }

    const isExpandable = (g) => !(g.children.length <= 1 && (!g.children[0] || g.children[0].label === g.label));

    // Headline: the one sentence that answers "what is filling my context?"
    const g0 = scope.groups[0];
    if (g0) {
      const dot = el('span', `hl-dot g-${g0.id}`);
      head.appendChild(dot);
      head.appendChild(document.createTextNode(`${g0.label}占 `));
      head.appendChild(el('b', '', pct(g0.share)));
      if (scope.top) {
        const topName = scope.top.group === 'tool' ? `${scope.top.label} 的结果`
          : scope.top.group === 'args' ? `${scope.top.label} 的参数` : scope.top.label;
        head.appendChild(document.createTextNode(' · 最大单项 '));
        head.appendChild(el('b', '', topName));
        head.appendChild(document.createTextNode(` ${fmtK(scope.top.tokens)}（${pct(scope.top.share)}）`));
      }
    }

    // Stacked bar: one segment per group
    const barDesc = [];
    for (const g of scope.groups) {
      const expandable = isExpandable(g);
      const seg = el('div', `comp-seg g-${g.id}${expandable ? ' clickable' : ''}`);
      seg.style.flexGrow = String(g.tokens);
      seg.dataset.group = g.id;
      seg.title = `${g.label} ${fmtK(g.tokens)} · ${pct(g.share)}${expandable ? '（点击展开明细）' : ''}`;
      seg.addEventListener('mouseenter', () => setBarFocus(g.id));
      seg.addEventListener('mouseleave', () => setBarFocus(null));
      if (expandable) {
        seg.addEventListener('click', () => {
          if (state.compOpen.has(g.id)) state.compOpen.delete(g.id);
          else state.compOpen.add(g.id);
          renderComposition(state.snap?.breakdown);
        });
      }
      bar.appendChild(seg);
      barDesc.push(`${g.label} ${pct(g.share)}`);
    }
    bar.setAttribute('aria-label', `上下文构成：${barDesc.join('，')}`);

    // Legend with expandable details
    for (const g of scope.groups) {
      const expandable = isExpandable(g);
      const li = el('li', `g-${g.id}`);
      const row = el(expandable ? 'button' : 'div', expandable ? 'comp-row' : 'comp-row static');
      const open = expandable && state.compOpen.has(g.id);
      if (expandable) {
        row.type = 'button';
        row.setAttribute('aria-expanded', open ? 'true' : 'false');
      }
      row.title = g.hint;
      row.appendChild(el('span', 'comp-dot'));
      const name = el('span', 'comp-name');
      name.appendChild(el('div', 'comp-label', g.label));
      const subTxt = !expandable
        ? g.hint
        : g.children.slice(0, 3).map((c) => `${c.label} ${fmtK(c.tokens)}`).join(' · ')
          + (g.children.length > 3 ? ` · +${g.children.length - 3}` : '');
      name.appendChild(el('div', 'comp-sub', subTxt));
      row.appendChild(name);
      row.appendChild(el('span', 'comp-val', fmtK(g.tokens)));
      row.appendChild(el('span', 'comp-pct', pct(g.share)));
      row.appendChild(el('span', 'comp-chev', expandable ? '▶' : ''));
      if (expandable) {
        row.addEventListener('click', () => {
          if (state.compOpen.has(g.id)) state.compOpen.delete(g.id);
          else state.compOpen.add(g.id);
          renderComposition(state.snap?.breakdown);
        });
      }
      row.addEventListener('mouseenter', () => setBarFocus(g.id));
      row.addEventListener('mouseleave', () => setBarFocus(null));
      li.appendChild(row);
      if (open) {
        const sub = el('ul', 'comp-children');
        const maxT = g.children[0]?.tokens || 1;
        for (const c of g.children) {
          const r = el('li', 'comp-child');
          r.title = `${c.label}：${c.tokens.toLocaleString()} token≈ · ${c.chars.toLocaleString()} 字符 · ${c.n} 条`;
          r.appendChild(el('span', 'comp-child-name', c.label));
          const track = el('span', 'mini-track');
          const fill = el('span', 'mini-fill');
          fill.style.display = 'block';
          fill.style.width = `${Math.max(2, (c.tokens / maxT) * 100)}%`;
          track.appendChild(fill);
          r.appendChild(track);
          r.appendChild(el('span', 'comp-child-val', `${fmtK(c.tokens)} · ${pct(c.share)}`));
          sub.appendChild(r);
        }
        li.appendChild(sub);
      }
      list.appendChild(li);
    }

    $('comp-foot').textContent = state.compScope === 'total'
      ? `累计进入过上下文的内容（含已被压缩的部分）· 合计 ${fmtK(scope.total)} · 压缩 ${bd.compactions} 次`
      : `自第 ${bd.window.sinceStep ?? 0} 步起 · 合计 ${fmtK(scope.total)}（= 当前真实 prompt${bd.window.trimmed < 0 ? `；含切模型/裁剪折算 -${fmtK(-bd.window.trimmed)}` : ''}）· token 按字符比例估算`;
  }

  function renderTools(bd) {
    const card = $('tools-card');
    if (!bd || (!bd.tools.length && !bd.skills.length)) {
      card.classList.add('hidden');
      return;
    }
    card.classList.remove('hidden');
    $('tools-meta').textContent = `${bd.totalCalls} 次调用 · ${bd.tools.length} 个工具`;
    const list = $('tool-list');
    clear(list);
    const LIMIT = 8;
    const shown = state.toolsAll ? bd.tools : bd.tools.slice(0, LIMIT);
    const maxT = Math.max(1, ...bd.tools.map((t) => t.resultTokens + t.argTokens));
    for (const t of shown) {
      const li = el('li', 'tool-row');
      li.title = `${t.name}：调用 ${t.calls} 次 · 结果 ≈${t.resultTokens.toLocaleString()} token（${t.resultChars.toLocaleString()} 字符）`
        + ` · 参数 ≈${t.argTokens.toLocaleString()} token（${t.argChars.toLocaleString()} 字符）${t.errors ? ` · 错误 ${t.errors}` : ''}`;
      const name = el('div', 'tool-name');
      name.appendChild(document.createTextNode(t.name));
      li.appendChild(name);
      const mid = el('div');
      const track = el('div', 'tool-track');
      const r = el('span', `g-tool${t.resultTokens > 0 ? ' nz' : ''}`);
      r.style.width = `${(t.resultTokens / maxT) * 100}%`;
      const a = el('span', `g-args${t.argTokens > 0 ? ' nz' : ''}`);
      a.style.width = `${(t.argTokens / maxT) * 100}%`;
      track.appendChild(r);
      track.appendChild(a);
      mid.appendChild(track);
      const calls = el('div', 'tool-calls', `×${t.calls}`);
      if (t.errors) calls.appendChild(el('span', 'tool-err', `${t.errors} 错误`));
      mid.appendChild(calls);
      li.appendChild(mid);
      li.appendChild(el('div', 'tool-val', fmtK(t.resultTokens + t.argTokens)));
      list.appendChild(li);
    }
    const more = $('tool-more');
    if (bd.tools.length > LIMIT) {
      more.classList.remove('hidden');
      more.textContent = state.toolsAll ? '收起' : `展开全部 ${bd.tools.length} 个工具`;
    } else {
      more.classList.add('hidden');
    }

    const sk = $('skill-chips');
    clear(sk);
    if (!bd.skills.length) {
      sk.appendChild(el('span', 'chip empty', '本对话未使用 Skill'));
    }
    for (const s of bd.skills) {
      const c = el('span', 'chip');
      c.title = `第 ${s.firstStep} – ${s.lastStep} 步`;
      c.appendChild(el('b', '', s.name));
      c.appendChild(el('span', 'chip-k', `加载 ${s.loads}${s.resourceUses ? ` · 资源 ${s.resourceUses}` : ''}`));
      sk.appendChild(c);
    }
    const mb = $('mcp-block');
    const mc = $('mcp-chips');
    clear(mc);
    mb.classList.toggle('hidden', !bd.mcp.length);
    for (const m of bd.mcp) {
      const c = el('span', 'chip');
      c.appendChild(el('b', '', m.key));
      c.appendChild(el('span', 'chip-k', `×${m.calls}`));
      mc.appendChild(c);
    }
  }

  function renderBreakdownSummary(bd) {
    const card = $('comp-summary');
    if (!bd || (!bd.window?.total && !bd.total?.total)) {
      card.classList.add('hidden');
      return;
    }
    const scope = bd.window?.total ? bd.window : bd.total;
    card.classList.remove('hidden');

    const head = $('comp-mini-headline');
    clear(head);
    const g0 = scope.groups[0];
    if (g0) {
      head.appendChild(el('span', `hl-dot g-${g0.id}`));
      head.appendChild(document.createTextNode(`${g0.label}占 `));
      head.appendChild(el('b', '', pct(g0.share)));
      if (scope.top) {
        const topName = scope.top.group === 'tool' ? `${scope.top.label} 结果`
          : scope.top.group === 'args' ? `${scope.top.label} 参数` : scope.top.label;
        head.appendChild(document.createTextNode(' · 最大 '));
        head.appendChild(el('b', '', topName));
        head.appendChild(document.createTextNode(` ${fmtK(scope.top.tokens)}`));
      }
    }

    const bar = $('comp-mini-bar');
    clear(bar);
    for (const g of scope.groups) {
      const seg = el('div', `comp-seg g-${g.id}`);
      seg.style.flexGrow = String(g.tokens);
      bar.appendChild(seg);
    }

    const leg = $('comp-mini-legend');
    clear(leg);
    for (const g of scope.groups) {
      const item = el('span', `comp-mini-item g-${g.id}`);
      item.appendChild(el('i'));
      item.appendChild(document.createTextNode(`${g.label} `));
      item.appendChild(el('b', '', pct(g.share)));
      leg.appendChild(item);
    }

    const metaParts = [];
    if (bd.tools.length) metaParts.push(`工具 ${bd.totalCalls} 次（${bd.tools.length} 个）`);
    if (bd.skills.length) metaParts.push(`Skill ${bd.skills.length}`);
    if (bd.mcp.length) metaParts.push(`MCP ${bd.mcp.length}`);
    $('comp-mini-meta').textContent = metaParts.join(' · ') || '点击查看构成明细';
  }

  function renderBreakdownDetail(bd) {
    $('dashboard').classList.add('hidden');
    $('model-detail').classList.add('hidden');
    $('empty-state').classList.add('hidden');
    const view = $('breakdown-detail');
    view.classList.remove('hidden');

    if (!state.compOpenInit && bd?.window?.groups?.length) {
      const firstExpandable = bd.window.groups.find(
        (g) => !(g.children.length <= 1 && (!g.children[0] || g.children[0].label === g.label))
      );
      if (firstExpandable) state.compOpen.add(firstExpandable.id);
      state.compOpenInit = true;
    }

    for (const b of view.querySelectorAll('.seg-btn')) {
      const on = b.dataset.scope === state.compScope;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    }
    renderComposition(bd);
    renderTools(bd);
  }

  for (const b of document.querySelectorAll('#breakdown-detail .seg-btn')) {
    b.addEventListener('click', () => {
      state.compScope = b.dataset.scope;
      renderBreakdownDetail(state.snap?.breakdown);
    });
  }
  $('comp-summary').addEventListener('click', () => {
    state.selectedModelKey = null;
    state.showBreakdown = true;
    window.scrollTo(0, 0);
    if (state.snap) render(state.snap);
  });
  $('btn-back-breakdown').addEventListener('click', () => {
    state.showBreakdown = false;
    window.scrollTo(0, 0);
    if (state.snap) render(state.snap);
  });
  $('tool-more').addEventListener('click', () => {
    state.toolsAll = !state.toolsAll;
    renderTools(state.snap?.breakdown);
  });

  function render(snap) {
    state.snap = snap;
    const conv = snap.conversationId || currentConv();
    $('conv-badge').textContent = conv ? conv.slice(0, 8) : '--';

    if (!snap.found) {
      state.selectedModelKey = null;
      state.showBreakdown = false;
      $('dashboard').classList.add('hidden');
      $('model-detail').classList.add('hidden');
      $('breakdown-detail').classList.add('hidden');
      $('empty-state').classList.remove('hidden');
      $('empty-desc').textContent = conv
        ? '尚未检测到本会话的 transcript_full.jsonl（发送首条消息后自动开始监控）。'
        : '当前面板未绑定对话 ID。';
      return;
    }

    if (state.selectedModelKey) {
      const foundMod = (snap.models?.list || []).find((m) => m.key === state.selectedModelKey);
      if (foundMod) {
        renderModelDetail(snap, foundMod);
        return;
      }
      state.selectedModelKey = null;
    }

    if (state.showBreakdown && snap.breakdown) {
      renderBreakdownDetail(snap.breakdown);
      return;
    }

    $('model-detail').classList.add('hidden');
    $('breakdown-detail').classList.add('hidden');
    $('empty-state').classList.add('hidden');
    $('dashboard').classList.remove('hidden');

    const { usage, compaction, guard, taskState } = snap;
    const lvClass = `level-${guard.level}`;

    const badge = $('guard-badge');
    badge.className = `guard-badge ${lvClass}`;
    badge.textContent = guard.levelLabel;

    const banner = $('alert-banner');
    const k = (n) => `${Math.round(n / 1000)}k`;
    const range = `${k(usage.compactionRange[0])}–${k(usage.compactionRange[1])}`;
    const isTokens = usage.source === 'tokens';
    const maxVal = isTokens ? usage.targetTokens : usage.warnCharLevel2;
    const lv1 = isTokens ? usage.warnTokenLevel1 : usage.warnCharLevel1;
    const lv2 = isTokens ? usage.warnTokenLevel2 : usage.warnCharLevel2;
    const unitSuffix = isTokens ? '' : ' 字符';
    if (maxVal > 0) {
      [[1, lv1, '一级预警'], [2, lv2, '二级紧急']].forEach(([i, v, name]) => {
        const pct = Math.min(100, Math.round((v / maxVal) * 100));
        const tick = $(`tick-${i}`);
        const label = $(`tick-label-${i}`);
        if (tick) { tick.style.left = `${pct}%`; tick.title = `${name} ${k(v)}${unitSuffix} (${pct}%)`; }
        if (label) {
          label.style.left = `${pct}%`;
          label.textContent = k(v);
          label.classList.toggle('hidden', !isTokens && pct >= 98);
        }
      });
      const max = $('tick-label-max');
      if (max) max.textContent = `${k(maxVal)}${unitSuffix}`;
    }
    const cfgIssues = guard.config && !guard.config.ok ? guard.config.issues || [] : [];
    if (cfgIssues.length || guard.stale) {
      banner.className = 'alert-banner level-2';
      banner.textContent = cfgIssues.length
        ? `⚠ Context Guard hook 配置异常：${cfgIssues.join('；')}。修复后需重启 Antigravity 或新开对话。`
        : '⚠ Context Guard hook 未在本会话运行（状态落后于对话记录）：预警和压缩恢复都不会生效。检查 ~/.gemini/config/hooks.json，然后重启 Antigravity。';
    } else if (guard.lastDelivery === 'failed' && guard.level !== 3) {
      banner.className = 'alert-banner level-2';
      banner.textContent = '⚠ 上一次恢复包没有出现在对话记录里（注入被丢弃）。Context Guard 已重新挂起补发；如果仍失败，请让代理用 recall.py 自查。';
    } else if (guard.level === 1) {
      banner.className = 'alert-banner level-1';
      banner.textContent = `🟠 已达一级预警 (≥${k(lv1)}${unitSuffix})：建议尽快更新 task_state.md，并在自然断点执行 /compact。`;
    } else if (guard.level === 2) {
      banner.className = 'alert-banner level-2';
      banner.textContent = isTokens
        ? `🔴 已达二级紧急预警 (≥${k(lv2)})：即将自动压缩（~${range}），请立即保存记忆或执行 /compact！`
        : `🔴 已达二级紧急预警 (≥${k(lv2)} 字符)：即将自动压缩，请立即保存记忆或执行 /compact！`;
    } else if (guard.level === 3) {
      banner.className = 'alert-banner level-3';
      banner.textContent = `🔄 检测到${guard.manualCompaction ? '手动压缩（/compact）' : '自动压缩'}：Context Guard 将在下一执行步注入无损恢复包。`;
    } else {
      banner.className = 'alert-banner hidden';
      banner.textContent = '';
    }

    const SRC = { observed: '实测', builtin: '内置', db_limit: '数据库上限', default: '默认', override: '手动' };
    const srcRaw = usage.levelsSource || '';
    const srcTxt = srcRaw
      ? `${SRC[srcRaw.replace('+seen', '')] || srcRaw}${srcRaw.endsWith('+seen') ? '·已上调' : ''}`
      : '';
    const basis = usage.levelsModel ? `；阈值依据 ${usage.levelsModel}${srcTxt ? ` · ${srcTxt}` : ''}` : '';
    if (usage.source === 'tokens') {
      $('usage-value').textContent = fmtK(usage.tokens);
      $('usage-unit').textContent = `/ ~${fmtK(usage.targetTokens)} tokens`;
      $('usage-pct').textContent = `${usage.percent.toFixed(1)}%`;
      $('usage-sub').textContent = usage.overTarget
        ? `已超 ${k(usage.targetTokens)} 基准 +${fmtK(usage.tokens - usage.targetTokens)}（实际上限以当前模型为准${basis}）`
        : `距 ${k(usage.targetTokens)} 目标余量 ${fmtK(usage.remainingToTarget)}（预计压缩区间 ${range}${basis}）`;
    } else {
      $('usage-value').textContent = fmtK(usage.chars);
      $('usage-unit').textContent = `/ ${k(usage.warnCharLevel2)} chars (回退估算)`;
      $('usage-pct').textContent = `${usage.percent.toFixed(1)}%`;
      $('usage-sub').textContent = '首轮 MODEL 响应完成后将自动切换为精确 Token 计数';
    }

    const fill = $('gauge-fill');
    fill.className = `gauge-fill ${lvClass}`;
    fill.style.width = `${Math.max(2, Math.min(100, usage.percent))}%`;

    const tot = snap.models?.totals || usage.totals || {};

    const hitPill = $('cache-hit-pill');
    if (tot.promptTokens > 0) {
      hitPill.className = `hit-pill ${hitClass(tot.cacheHitRate)}`;
      hitPill.textContent = `总命中 ${fmtPct(tot.cacheHitRate)}`;
    } else {
      hitPill.className = 'hit-pill hidden';
    }

    renderModelList(snap.models, snap.models?.activeModel);

    const totBox = $('session-total-box');
    const totLine = $('session-total-line');
    clear(totLine);
    if (tot.steps > 0) {
      totBox.classList.remove('hidden');
      $('session-total-meta').textContent = `${tot.steps} 次 · 归零 ${tot.cacheMisses || 0} 次`;
      const parts = [
        ['总输入 ', fmtK(tot.promptTokens)],
        ['缓存读 ', fmtK(tot.cacheReadTokens)],
        ['总输出 ', fmtK(tot.outputTokens)],
      ];
      if (tot.tps !== null && tot.tps !== undefined) parts.push(['均速 ', `${tot.tps.toFixed(1)}/s`]);
      for (const [kTxt, vTxt] of parts) {
        const sp = el('span');
        sp.appendChild(document.createTextNode(kTxt));
        sp.appendChild(el('b', '', vTxt));
        totLine.appendChild(sp);
      }
      const rw = tot.rewound;
      if (rw && rw.steps > 0) {
        const sp = el('span', 'rewound-note');
        sp.title = '编辑/重试旧消息后被丢弃分支上的模型调用：已实际消耗，但不属于当前对话，未计入上面的统计';
        sp.appendChild(document.createTextNode(`已回退分支 ${rw.steps} 次调用（输入 ${fmtK(rw.promptTokens)} / 输出 ${fmtK(rw.outputTokens)}）未计入`));
        totLine.appendChild(sp);
      }
    } else {
      totBox.classList.add('hidden');
    }

    $('guard-sync').textContent = guard.stale
      ? `⚠ Hook 未运行${guard.lagMs ? `（落后 ${Math.round(guard.lagMs / 60000)} 分钟）` : ''}`
      : guard.connected
        ? `Hook ${guard.lastEvent}${guard.updatedAt ? ` · ${fmtAgo(guard.updatedAt * 1000)}` : ''}`
          + (guard.verifying ? ' · 恢复包待确认' : guard.lastDelivery === 'ok' ? ' · 恢复包已送达' : guard.lastDelivery === 'failed' ? ' · ⚠ 恢复包未送达' : '')
        : 'Hook 尚未写入状态';

    const tsDot = $('ts-dot');
    const tsBtn = $('btn-open-ts');
    if (taskState.exists) {
      tsDot.className = 'status-dot ok';
      $('ts-meta').textContent = `${(taskState.size / 1024).toFixed(1)} KB · 更新于 ${fmtAgo(taskState.mtime)}`;
      tsBtn.disabled = !taskState.uri;
    } else {
      tsDot.className = 'status-dot missing';
      $('ts-meta').textContent = '尚未创建（多步任务建议先建立工作记忆）';
      tsBtn.disabled = true;
    }

    const ckpts = compaction.checkpoints || [];
    const ckptSec = $('ckpt-section');
    const ckptList = $('ckpt-list');
    while (ckptList.firstChild) ckptList.removeChild(ckptList.firstChild);
    if (ckpts.length > 0) {
      ckptSec.classList.remove('hidden');
      for (const ck of [...ckpts].reverse().slice(0, 3)) {
        const li = document.createElement('li');
        const left = document.createElement('span');
        left.textContent = `CHECKPOINT #${ck.step ?? '?'}`;
        const right = document.createElement('span');
        right.textContent = ck.createdAt ? new Date(ck.createdAt).toLocaleTimeString() : '';
        li.appendChild(left);
        li.appendChild(right);
        ckptList.appendChild(li);
      }
    } else {
      ckptSec.classList.add('hidden');
    }

    renderBreakdownSummary(snap.breakdown);
  }

  async function poll() {
    const conv = currentConv();
    if (!conv) {
      render({ found: false, conversationId: '' });
      return;
    }
    try {
      const qs = new URLSearchParams({ conversationId: conv, etag: state.etag });
      const data = await apiFetch(`/api/context?${qs.toString()}`);
      $('live-pill').classList.remove('stale');
      $('live-text').textContent = 'Live';
      if (data.unchanged) return;
      state.etag = data.etag || '';
      render(data);
    } catch {
      $('live-pill').classList.add('stale');
      $('live-text').textContent = 'Reconnecting';
    }
  }

  $('btn-back-overview').addEventListener('click', () => {
    state.selectedModelKey = null;
    if (state.snap) render(state.snap);
  });

  $('btn-open-ts').addEventListener('click', () => {
    const uri = state.snap?.taskState?.uri;
    if (!uri) return;
    window.sidecar?.ui?.toggleAuxPane({ tabType: 'fileView', uri });
  });

  $('btn-copy-compact').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText('/compact');
      showToast('已复制 /compact，请在对话输入框粘贴发送');
    } catch {
      const ta = document.createElement('textarea');
      ta.value = '/compact';
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
      showToast('已复制 /compact，请在对话输入框粘贴发送');
    }
  });

  $('btn-remind-ts').addEventListener('click', async () => {
    const conv = currentConv();
    if (!conv) return;
    const now = Date.now();
    const btn = $('btn-remind-ts');
    if (now > state.remindConfirmUntil) {
      state.remindConfirmUntil = now + 4000;
      btn.textContent = '再次点击确认发送';
      setTimeout(() => {
        if (Date.now() > state.remindConfirmUntil) btn.textContent = '提醒更新记忆';
      }, 4100);
      return;
    }
    state.remindConfirmUntil = 0;
    btn.textContent = '发送中…';
    btn.disabled = true;
    try {
      await apiFetch('/api/actions/remind-state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversationId: conv }),
      });
      showToast('已向当前对话发送更新 task_state.md 提醒');
    } catch (err) {
      showToast(`发送失败: ${err.message}`);
    } finally {
      btn.textContent = '提醒更新记忆';
      btn.disabled = false;
    }
  });

  poll();
  setInterval(poll, POLL_MS);
})();
