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
const draw = view => { const element = { innerHTML: '' }; drawMemoryPanel(element, view); return element.innerHTML; };
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
    'Rows per step: 8 · Lookahead: 0 · tokens_min: 8.192</p>'));
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
