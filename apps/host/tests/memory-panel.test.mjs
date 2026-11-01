// The Memory panel (strata_server/web/memory-panel.js): one framework-free module that the server's web page imports
// and the app's Monitor screen wraps (src/components/MemoryPanel.tsx), so both draw the same bars from GET /v1/memory.
// The fixture is strata-q38's real GET_MEMORY_USE answer with three conversations on a Strix Halo (2026-10-31),
// passed through the server's StrataEngine.memory and memory_view with a 64 GiB BIOS share.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { drawMemoryPanel } from '../../../strata_server/web/memory-panel.js';

const read = path => readFile(new URL(path, import.meta.url), 'utf8');
const fixture = JSON.parse(await read('./fixtures/memory-strix-halo.json'));
/** Draws `view` into a stand-in for a page element that records the panel's event listeners; `fire` sends one of
 *  them an event. */
const mount = (view, onChange) => {
  const listeners = {}, element = { innerHTML: '', ownerDocument: { activeElement: null }, contains: () => false,
    addEventListener: (type, listener) => { listeners[type] = listener; }, setPointerCapture() {},
    querySelector: () => null };
  drawMemoryPanel(element, view, onChange);
  const fire = (type, event) => listeners[type]({ button: 0, pointerId: 1, preventDefault() {}, ...event });
  return { element, fire };
};
/** Gets an event target inside the divider of memory `id`, on a bar 1000 pixels wide. */
const onDivider = id => ({ closest: selector => selector === '[data-divider]'
  ? { dataset: { divider: id }, parentElement: { getBoundingClientRect: () => ({ width: 1000 }) } } : null });
/** Gets an event target that is one of the panel's controls, matching `selector`. */
const control = (selector, fields) => ({ matches: s => s === selector, ...fields });
const draw = view => mount(view).element.innerHTML;
const rows = html => html.split('<div class="mem-row">').slice(1);
const widths = row => [...row.matchAll(/class="mem-seg"[^>]*style="width:([\d.e-]+)%/g)].map(m => Number(m[1]));
const barWidth = row => Number(/class="mem-bar" style="width:([\d.]+)%/.exec(row)[1]);

test('a GPU whose memory is system RAM is drawn as its BIOS share, and what it borrows is a segment of the RAM', () => {
  const [gpu, ram] = rows(draw(fixture));
  assert.match(gpu, /AMD Radeon\(TM\) 8060S Graphics — BIOS share 64.0 GiB/);
  assert.match(ram, /System RAM 63.5 GiB \(GPU borrows 0.3 GiB\)/);
  assert.equal(barWidth(gpu), 100);                       // above 32 GiB: the full width
  // the engine's uses are scaled to what it holds in the share, so the segments fill the bar exactly
  assert.ok(Math.abs(widths(gpu).reduce((a, b) => a + b, 0) - 100) < 1e-9);
  assert.match(gpu, /Experts 15.54 GiB/);                 // the legend keeps the engine's own numbers
  assert.match(gpu, /data-use="pool_free"[^]*Free in the pool 53 MiB/);
  assert.match(ram, /data-use="borrowed"/);
});

test('a memory below 32 GiB is as wide as its share of 32 GiB; names are escaped', () => {
  const card = { id: 'gpu1', name: 'RTX <3060>', total: 16 * 2 ** 30, free: 2 ** 30, unified: false, paused_count: null,
    paused_budget: null, uses: { experts: 15 * 2 ** 30 } };
  const [row] = rows(draw({ available: true, memories: [card], settings: {} }));
  assert.equal(barWidth(row), 50);
  assert.match(row, /RTX &lt;3060&gt;/);
  assert.deepEqual(widths(row), [93.75, 6.25]);
});

test('a page pool says how much of it is allocated and how much is free in runs no expert fits', () => {
  const card = { id: 'gpu0', name: 'GPU', total: 2 ** 30, free: 0, unified: false, paused_count: null,
    paused_budget: null, uses: { experts: 2 ** 29 }, allocated_pct: 97.5, fragmented_pct: 0.38 };
  assert.match(rows(draw({ available: true, memories: [card], settings: {} }))[0],
    /mem-facts">1[.,]0 GiB · pool 97[.,]5% allocated, 0[.,]38% fragmented</);
  // engines without a pool (and memories other than the pool's) say nothing of it
  assert.doesNotMatch(draw({ available: true, memories: [{ ...card, allocated_pct: null }], settings: {} }),
    /allocated/);
});

test('the conversations the engine holds are listed with their state, tokens and memory', () => {
  const html = draw(fixture), table = /<table class="mem-conversations">(.*)<\/table>/.exec(html)?.[1] ?? '';
  assert.equal(table.match(/<tr>/g).length, 1 + fixture.conversations.length);
  assert.match(table, /<td>slot2<\/td><td>holding \(GPU\)<\/td><td class="mem-num">8.401 \/ 32.768<\/td>/);
  assert.match(table, /8.401 \/ 32.768<\/td><td class="mem-num">230 MiB<\/td>/);
  // engines before strata-q38 list no conversations
  assert.doesNotMatch(draw({ ...fixture, conversations: undefined }), /mem-conversations/);
});

test('the settings line names the settings and says when the reading is an old one', () => {
  assert.match(draw(fixture), new RegExp('<p class="mem-note">Context: 32.768 tokens · Conversations at once: 3 · ' +
    'Rows per step: 8 · Lookahead: 0 · Conversation tokens, minimum: 8.192</p>'));
  assert.match(draw({ ...fixture, busy: true }), /mem-note">[^<]* · a request has the engine: the last reading</);
});

test('the web page and the app load the one module instead of copies', async () => {
  assert.match(await read('../../../strata_server/web/app.js'), /import\("\.\/memory-panel\.js"\)/);
  assert.match(await read('../../../strata_server/web/index.html'), /href="web\/memory-panel\.css"/);
  const wrapper = await read('../src/components/MemoryPanel.tsx');
  assert.match(wrapper, /from '\.\.\/\.\.\/\.\.\/\.\.\/strata_server\/web\/memory-panel\.js'/);
  assert.match(wrapper, /import '\.\.\/\.\.\/\.\.\/\.\.\/strata_server\/web\/memory-panel\.css'/);
  assert.match(await read('../src/screens/Monitor.tsx'), /<MemoryPanel instanceId=\{selectedInstance\.id\}\/>/);
});

// M6: an engine that changes its settings while it runs (strata-q38's READY says `settings`).  The GPU's pool is what
// the fixture's engine holds in it (experts, conversations, free pages); it accepts 4 to 30 GiB.
const GIB = 2 ** 30, MIB = 2 ** 20;
const [gpu0, ram] = fixture.memories, pool = gpu0.uses.experts + gpu0.uses.conversations + gpu0.uses.pool_free;
const live = { ...fixture, can_change: true, config: true,
  memories: [{ ...gpu0, setting: pool, smallest: 4 * GIB, largest: 30 * GIB }, ram],
  settings: { ...fixture.settings, slots: 4, drafter: 1, tokens_max: 32768, 'memory_mib.gpu0': pool / MIB } };
/** Bytes one pixel of the 1000-pixel GPU bar stands for: the BIOS share, drawn scaled to what the engine holds in
 *  it. */
const perPx = gpu0.total / (gpu0.in_share / Object.values(gpu0.uses).reduce((a, b) => a + b)) / 1000;
/** Gets a stand-in for the caller's onChange that records its calls and answers `answer`. */
const recorder = (answer = { ok: true }) => {
  const calls = [];
  return { calls, onChange: async (set, save) => { calls.push([set, save]); return answer; } };
};
const settle = () => new Promise(resolve => setImmediate(resolve));
const undoButton = { closest: selector => selector === '[data-undo]' ? {} : null };

test('dragging a divider previews the new size and changes nothing until it is released', async () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  assert.match(element.innerHTML, /class="mem-divider" data-divider="gpu0" tabindex="0" role="slider"/);
  assert.match(element.innerHTML, /title="Drag to change the memory the engine&#39;s pool takes on this GPU/);
  assert.match(element.innerHTML, /<p class="mem-hint">Drag the dividers to change sizes<\/p>/);
  fire('pointerdown', { target: onDivider('gpu0'), clientX: 500 });
  fire('pointermove', { clientX: 600 });
  assert.equal(calls.length, 0);
  assert.match(element.innerHTML, /<p class="mem-preview">16[.,]08 GiB → 22[.,]\d\d GiB: release to apply/);
  assert.match(element.innerHTML, /class="mem-change"/);
  fire('pointerup', {});
  await settle();
  assert.deepEqual(calls, [[{ 'memory_mib.gpu0': Math.round((pool + 100 * perPx) / MIB) }, 'session']]);
  assert.match(element.innerHTML, /mem-notice" role="status">Changed for this session\. <button[^>]*data-undo>Undo/);
});

test('a change that got less than asked says what it lacked and still offers Undo', async () => {
  const { onChange } = recorder({ ok: true, view: { ...live, shortfall: "the RAM's pool: out of memory" } });
  const { element, fire } = mount(live, onChange);
  fire('keydown', { target: onDivider('gpu0'), key: 'ArrowRight' });
  fire('keydown', { target: onDivider('gpu0'), key: 'Enter' });
  await settle();
  assert.match(element.innerHTML, new RegExp('mem-notice mem-error" role="status">Changed for this session\\. ' +
    'But it got less than asked: the RAM&#39;s pool: out of memory\\. <button[^>]*data-undo>Undo'));
});

test('"Save as default" goes with the change, and Undo sends the size back the same way', async () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  assert.match(element.innerHTML, /<option value="session" selected>This session<\/option><option value="default">/);
  fire('change', { target: control('[data-save]', { value: 'default' }) });
  fire('keydown', { target: onDivider('gpu0'), key: 'ArrowRight', shiftKey: true });
  fire('keydown', { target: onDivider('gpu0'), key: 'Enter' });
  await settle();
  fire('click', { target: undoButton });
  await settle();
  assert.deepEqual(calls, [[{ 'memory_mib.gpu0': Math.round(pool / MIB) + 1024 }, 'default'],
    [{ 'memory_mib.gpu0': Math.round(pool / MIB) }, 'default']]);
  assert.doesNotMatch(element.innerHTML, /data-undo/);    // an Undo is not undone in turn
  // a server without a run config offers no default to save to
  assert.doesNotMatch(mount({ ...live, config: false }, onChange).element.innerHTML, /Save as default/);
});

test('a size outside what the engine accepts shows red with the reason, and releasing there changes nothing', () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  fire('pointerdown', { target: onDivider('gpu0'), clientX: 500 });
  fire('pointermove', { clientX: 250 });                 // about 16.9 GiB less: below the 4 GiB the engine needs
  assert.match(element.innerHTML, /class="mem-divider mem-refused"/);
  assert.match(element.innerHTML,
    /<p class="mem-preview mem-refused">16[.,]08 GiB → [^:]*: at least 4[.,]00 GiB is needed here now/);
  assert.match(element.innerHTML, /releasing here changes nothing/);
  fire('pointerup', {});
  assert.equal(calls.length, 0);
  assert.match(element.innerHTML, /mem-notice mem-error" role="status">Nothing changed: at least 4[.,]00 GiB/);
});

test('Escape cancels a drag, and dropping a divider where it started changes nothing', () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  fire('pointerdown', { target: onDivider('gpu0'), clientX: 500 });
  fire('pointermove', { clientX: 650 });
  fire('keydown', { target: onDivider('gpu0'), key: 'Escape' });
  assert.doesNotMatch(element.innerHTML, /mem-preview/);
  fire('pointerup', {});
  fire('pointerdown', { target: onDivider('gpu0'), clientX: 500 });
  fire('pointermove', { clientX: 650 });
  fire('pointermove', { clientX: 500 });
  fire('pointerup', {});
  assert.equal(calls.length, 0);
  assert.doesNotMatch(element.innerHTML, /mem-notice/);
  // a keyboard move the reader leaves by moving the focus away is dropped at the next reading
  fire('keydown', { target: onDivider('gpu0'), key: 'ArrowRight' });
  drawMemoryPanel(element, live, onChange);
  assert.doesNotMatch(element.innerHTML, /mem-preview/);
});

test('the server\'s refusal is shown as it said it', async () => {
  const why = 'the biggest active conversation needs 3.10 GiB; the GPU&#39;s pool would keep 2.00 GiB';
  const { onChange } = recorder({ ok: false, error: why.replace('&#39;', "'") });
  const { element, fire } = mount(live, onChange);
  fire('keydown', { target: onDivider('gpu0'), key: 'ArrowLeft' });
  fire('keydown', { target: onDivider('gpu0'), key: 'Enter' });
  assert.match(element.innerHTML, /mem-notice" role="status">Changing…/);
  await settle();
  assert.match(element.innerHTML, new RegExp(`mem-notice mem-error" role="status">${why}</p>`));
  assert.doesNotMatch(element.innerHTML, /data-undo/);
});

test('the controls change the other settings within their ranges', async () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  assert.match(element.innerHTML, /data-setting="concurrent" value="3" min="1" max="4"/);
  assert.match(element.innerHTML, /data-setting="rows" value="8" min="3" max="8"/);
  assert.match(element.innerHTML, /data-setting="lookahead" value="0" min="0" max="7"/);
  assert.match(element.innerHTML, /<input type="checkbox" data-setting="drafter" checked> Drafter/);
  const setting = (key, fields) => control('[data-setting]', { dataset: { setting: key }, ...fields });
  fire('change', { target: setting('concurrent', { value: '5' }) });
  assert.equal(calls.length, 0);
  assert.match(element.innerHTML,
    /mem-error" role="status">Conversations at once: choose a whole number from 1 to 4\./);
  fire('change', { target: setting('concurrent', { value: '2' }) });
  await settle();
  fire('change', { target: setting('drafter', { checked: false }) });
  await settle();
  assert.deepEqual(calls, [[{ concurrent: 2 }, 'session'], [{ drafter: false }, 'session']]);
  // the controls' settings are not listed again in the line under the panel
  assert.match(element.innerHTML, /<p class="mem-note">Context: 32.768 tokens · Batch slots: 4<\/p>/);
});

test('an engine that cannot change its settings while it runs shows them read-only', () => {
  const { calls, onChange } = recorder(), fixed = mount({ ...live, can_change: false }, onChange).element.innerHTML;
  for (const html of [fixed, draw(live)]) {             // draw(live): a caller that sends no changes
    assert.doesNotMatch(html, /data-divider|<input|<select|mem-hint/);
    assert.match(html, /Conversations at once: 3/);
  }
  assert.match(fixed, /Drafter: on · .* · these change when the model starts again<\/p>/);
  assert.equal(calls.length, 0);
});

test('the SSD\'s divider is the end of the room for paused conversations', async () => {
  const ssd = { id: 'ssd', name: null, total: null, free: null, unified: false, paused_count: 0,
    paused_budget: 8 * GIB, uses: { experts: 8 * GIB } };
  const { calls, onChange } = recorder(), { element, fire } = mount({ ...live, memories: [ssd] }, onChange);
  assert.match(element.innerHTML, /data-divider="ssd"[^>]*style="left:100%"/);
  fire('pointerdown', { target: onDivider('ssd'), clientX: 1000 });
  fire('pointermove', { clientX: 750 });                 // a quarter of the bar's 16 GiB less
  fire('pointerup', {});
  await settle();
  assert.deepEqual(calls, [[{ ssd_pause_mib: 4096 }, 'session']]);
});
