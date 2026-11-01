// strata_server/web/memory-panel.js - the Memory panel (docs/MEMORY_MANAGER_PLAN.md, steps M0 and M0b): what the
// engine holds in each memory (each GPU, the RAM, the SSD), the conversations it holds and its settings, drawn from
// GET /v1/memory (strata_server/server.py memory_view).  Read-only for now; step M6 adds dividers between a bar's
// segments (`.mem-seg`) to drag.  No framework: the web page (app.js) imports this file as it is, and the desktop app
// (apps/host/src/components/MemoryPanel.tsx) wraps it.  Its styles are memory-panel.css beside it.
//
// A GPU whose memory is system RAM (Strix Halo) is drawn as the BIOS divides it: its bar is the BIOS share, with the
// engine's uses drawn scaled to what the engine holds in the share (`in_share`; the engine cannot know which
// allocation landed where; the legend keeps the engine's numbers), and what it borrows beyond the share is one segment
// of the RAM's bar.  A bar is as wide as its memory's share of FULL_BAR (one of unknown size fills it), so a card is
// seen beside the RAM without becoming a sliver.

/** Names shown for the uses a memory's bar is divided into (`uses` keys and the segments the panel adds). */
const USES = {weights: "Weights", experts: "Experts", conversations: "Conversations", pool_free: "Free in the pool",
              paused: "Paused conversations", drafter: "Drafter", scratch: "Scratch", engram: "Engram",
              borrowed: "GPU (borrowed from RAM)", other: "Other", free: "Free"};
/** Size in bytes (32 GiB) at which a memory's bar fills the width. */
const FULL_BAR = 32 * 1073741824;
/** Names shown for the engine's settings (keys of the view's `settings`); others show as the engine names them. */
const SETTINGS = {context: "Context", concurrent: "Conversations at once", rows: "Rows per step",
                  lookahead: "Lookahead"};
/** HTML entities for the characters esc replaces. @type {Record<string, string>} */
const ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"};
/** Names shown for where a conversation is held (its `place`). */
const PLACES = {vram: "GPU", ram: "RAM", ssd: "SSD"};

/**
 * Draws a GET /v1/memory answer whose `available` is true into `element`, replacing what it held: a bar per memory
 * with its legend, a table of the conversations the engine holds, and a line with the engine's settings that says
 * whether the reading is an old one.
 * @param {HTMLElement} element
 * @param {MemoryView} view
 */
export function drawMemoryPanel(element, view) {
  const note = [Object.entries(view.settings).map(([k, v]) =>
    `${SETTINGS[k] || k}: ${fmt(v)}${k === "context" ? " tokens" : ""}`).join(" · "),
    view.busy ? "a request has the engine: the last reading" : view.error || ""].filter(Boolean).join(" · ");
  element.innerHTML = `<div class="mem-bars">${view.memories.map(drawBar).join("")}</div>` +
    drawConversations(view.conversations || []) + (note ? `<p class="mem-note">${esc(note)}</p>` : "");
}

/**
 * The GET /v1/memory answer as the panel reads it (the fields memory_view and StrataEngine.memory describe).
 * @typedef {{available: boolean, reason?: string, busy?: boolean, error?: string | null,
 *   memories: MemoryUse[], settings: Record<string, number>, conversations?: HeldConversation[]}} MemoryView
 * @typedef {{id: string, name: string | null, total: number | null, free: number | null, unified: boolean,
 *   uses: Record<string, number>, paused_count: number | null, paused_budget: number | null, other?: number,
 *   in_share?: number, borrowed?: number | null, gpu_borrowed?: number, allocated_pct?: number | null,
 *   fragmented_pct?: number | null}} MemoryUse
 * @typedef {{id: string, place: string, state: string, count: number, capacity: number, bytes: number,
 *   used: number}} HeldConversation
 */

/** Gets the markup of one memory's row: its name and size, its bar and the bar's legend.
 * @param {MemoryUse} x */
function drawBar(x) {
  const named = Object.values(x.uses).reduce((a, b) => a + b, 0);
  const scale = x.in_share == null || !named ? 1 : Math.min(1, x.in_share / named);
  /** @type {{use: string, bytes: number, drawn?: number}[]} */
  const parts = Object.entries(x.uses).map(([use, bytes]) => ({use, bytes, drawn: bytes * scale}));
  if (x.gpu_borrowed)
    parts.push({use: "borrowed", bytes: x.gpu_borrowed});
  if (x.other)
    parts.push({use: "other", bytes: x.other});
  if (x.free != null)
    parts.push({use: "free", bytes: x.free});
  // a memory of unknown size (the SSD) is drawn against what it holds plus the room left in its paused budget
  const used = parts.reduce((a, p) => a + (p.drawn ?? p.bytes), 0);
  const total = x.total || Math.max(used, used - (x.uses.paused || 0) + (x.paused_budget || 0));
  const name = x.id === "ram" ? "System RAM" : x.id === "ssd" ? "SSD" : x.name || x.id.toUpperCase();
  const label = "borrowed" in x ? `${name} — BIOS share ${gb(x.total)} GiB` :
    x.gpu_borrowed ? `${name} ${gb(x.total)} GiB (GPU borrows ${gb(x.gpu_borrowed)} GiB)` : name;
  const facts = [x.total && label === name ? `${gb(x.total)} GiB` : "", x.paused_count == null ? "" :
    `${fmt(x.paused_count)} paused` + (x.paused_budget ? ` (budget ${gb(x.paused_budget, 2)} GiB)` : ""),
    x.allocated_pct == null ? "" :
    `pool ${fmt(x.allocated_pct, 1)}% allocated, ${fmt(x.fragmented_pct, 2)}% fragmented`]
    .filter(Boolean).join(" · ");
  const title = (/** @type {{use: string}} */ p) => esc(USES[p.use] || p.use);
  const seg = (/** @type {{use: string, bytes: number, drawn?: number}} */ p) =>
    `<span class="mem-seg" data-use="${esc(p.use)}" title="${title(p)}: ${size(p.bytes)}" ` +
    `style="width:${total ? (100 * (p.drawn ?? p.bytes)) / total : 0}%"></span>`;
  const key = (/** @type {{use: string, bytes: number}} */ p) =>
    `<span><i class="mem-key" data-use="${esc(p.use)}"></i>${title(p)} ${size(p.bytes)}</span>`;
  return `<div class="mem-row"><div class="mem-head"><span>${esc(label)}</span><span class="mem-facts">${esc(facts)}` +
    `</span></div><div class="mem-bar" style="width:${x.total ? 100 * Math.min(x.total / FULL_BAR, 1) : 100}%">` +
    `${parts.map(seg).join("")}</div><div class="mem-legend">${parts.map(key).join("")}</div></div>`;
}

/** Gets the markup of the table of conversations the engine holds; none for an empty list or an engine that does
 *  not list them (all but strata-q38).
 * @param {HeldConversation[]} list */
function drawConversations(list) {
  if (!list.length)
    return "";
  const row = (/** @type {HeldConversation} */ c) => `<tr><td>${esc(c.id)}</td>` +
    `<td>${esc(c.state)} (${esc(PLACES[c.place] || c.place)})</td>` +
    `<td class="mem-num">${fmt(c.count)} / ${fmt(c.capacity)}</td><td class="mem-num">${size(c.bytes)}</td></tr>`;
  return `<table class="mem-conversations"><thead><tr><th>Conversation</th><th>State</th>` +
    `<th class="mem-num">Tokens</th><th class="mem-num">Memory</th></tr></thead>` +
    `<tbody>${list.map(row).join("")}</tbody></table>`;
}

/** Escapes text for HTML. @param {unknown} s */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ESCAPES[c] ?? c);
}

/** Formats a number with `d` decimals in the reader's locale; "–" for none. @param {number | null | undefined} n */
function fmt(n, d = 0) {
  return n == null || Number.isNaN(n) ? "–" :
    Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
}

/** Formats bytes as binary GB (GiB, as Windows shows them) without the unit. @param {number | null} b */
function gb(b, d = 1) {
  return b == null ? "–" : fmt(b / 1073741824, d);
}

/** Formats bytes with their unit: MiB below 1 GiB (a conversation, the scratch), else GiB. @param {number} b */
function size(b) {
  return b < 1073741824 ? `${fmt(b / 1048576)} MiB` : `${gb(b, 2)} GiB`;
}
