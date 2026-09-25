// Cutout: a free background remover that runs in the browser.
// All image processing happens on the visitor's device; nothing is uploaded.

const MAX_LONG = 1920; // Full HD cap: long side ≤ 1920px …
const MAX_SHORT = 1080; // … and short side ≤ 1080px
const UNDO_LIMIT = 15;
const SWATCHES = ['#ffffff', '#000000', '#f3efe6', '#e7e4ff', '#ffd9e2', '#ffeaa6', '#cdf3d9', '#c3e4ff', '#5b4cf5', '#ff5a5f', '#1f9d63'];

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const el = {
  home: $('#home'),
  editor: $('#editor'),
  fileInput: $('#fileInput'),
  uploadBtn: $('#uploadBtn'),
  dropzone: $('#dropzone'),
  dropOverlay: $('#dropOverlay'),
  view: $('#view'),
  wrap: $('#canvasWrap'),
  status: $('#status'),
  statusText: $('#statusText'),
  statusSub: $('#statusSub'),
  progress: $('#progress'),
  progressBar: $('#progressBar'),
  compareHandle: $('#compareHandle'),
  brushCursor: $('#brushCursor'),
  brushSize: $('#brushSize'),
  brushSizeWrap: $('#brushSizeWrap'),
  undoBtn: $('#undoBtn'),
  compareBtn: $('#compareBtn'),
  tray: $('#tray'),
  addMoreBtn: $('#addMoreBtn'),
  downloadBtn: $('#downloadBtn'),
  downloadAllBtn: $('#downloadAllBtn'),
  dlSize: $('#dlSize'),
  swatches: $('#swatches'),
  bgImageBtn: $('#bgImageBtn'),
  bgImageInput: $('#bgImageInput'),
  blurAmount: $('#blurAmount'),
  deleteBtn: $('#deleteBtn'),
  newBtn: $('#newBtn'),
  toast: $('#toast'),
};
const vctx = el.view.getContext('2d');

const state = {
  items: [],
  currentId: null,
  tool: 'none', // 'none' | 'erase' | 'restore'
  format: 'png',
  compare: false,
  split: 0.5,
  model: { status: 'idle', loaded: 0, total: 0, device: null, key: null, gpu: null },
  quality: loadPref(), // 'auto' | 'best' | 'light'
  busyId: null,
};
let nextId = 1;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

const current = () => state.items.find((i) => i.id === state.currentId) || null;

let toastTimer;
function toast(msg, ms = 3500) {
  el.toast.textContent = msg;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.toast.hidden = true), ms);
}

const mb = (bytes) => (bytes / 1048576).toFixed(0);

function fitSize(w, h) {
  const long = Math.max(w, h);
  const short = Math.min(w, h);
  const scale = Math.min(1, MAX_LONG / long, MAX_SHORT / short);
  return { w: Math.max(1, Math.round(w * scale)), h: Math.max(1, Math.round(h * scale)) };
}

async function decode(blob) {
  try {
    return await createImageBitmap(blob, { imageOrientation: 'from-image' });
  } catch {
    // Fallback for browsers that can't decode some formats via createImageBitmap
    const url = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

function baseName(name) {
  return (name || 'image').replace(/\.[^.]+$/, '').replace(/[^\w\-]+/g, '-').slice(0, 60) || 'image';
}

/* ------------------------------------------------------------------ */
/* AI worker                                                           */
/* ------------------------------------------------------------------ */

function loadPref() {
  try {
    return localStorage.getItem('cutout-quality') || 'auto';
  } catch {
    return 'auto';
  }
}
function savePref(v) {
  try {
    localStorage.setItem('cutout-quality', v);
  } catch {}
}

let worker = null;

// (Re)start the AI worker. A fresh worker is used whenever the model changes, so a
// failed GPU session can never affect the next attempt.
function startWorker(preference) {
  worker?.terminate();
  const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker = w;
  w.onmessage = (e) => w === worker && onWorkerMessage(e); // ignore a replaced worker
  worker.onerror = (e) => {
    console.error(e);
    toast('Your browser blocked the AI engine. Try the latest Chrome, Edge, Firefox or Safari.', 7000);
  };
  worker.postMessage({ type: 'config', preference });
  Object.assign(state.model, { status: 'idle', key: null, loaded: 0, total: 0 });
  // Anything that was mid-way through processing goes back in the queue
  const busy = state.items.find((i) => i.id === state.busyId);
  if (busy) {
    busy.status = 'queued';
    updateThumb(busy);
  }
  state.busyId = null;
}

function resetItem(it) {
  it.status = 'queued';
  it.mask = null;
  it.undo = [];
  it.cutDirty = true;
  updateThumb(it);
}

function setQuality(q) {
  const m = state.model;
  if (m.status === 'ready' && m.key === q) return;
  state.quality = q;
  savePref(q);
  startWorker(q);
  const it = current(); // redo the current image with the new model
  if (it) resetItem(it);
  if (state.items.length) warmup();
  syncUI();
  updateStatus();
  render();
  pump();
}

function warmup() {
  if (state.model.status !== 'idle') return;
  state.model.status = 'loading';
  worker.postMessage({ type: 'warmup' });
}

function onWorkerMessage({ data }) {
  const m = state.model;
  switch (data.type) {
    case 'gpu-failed': {
      // The best model didn't work on this device's graphics card: use the light one.
      console.warn('Best quality failed on this device, switching to Fast:', data.message);
      toast("Best quality doesn't work on this device. Switched to Fast.", 5000);
      state.quality = 'light';
      startWorker('light');
      warmup();
      syncUI();
      pump();
      break;
    }
    case 'loading':
      m.status = 'loading';
      m.device = data.device;
      m.key = data.model;
      m.loaded = 0;
      m.total = 0;
      break;
    case 'progress':
      m.loaded = data.loaded;
      m.total = data.total;
      break;
    case 'ready':
      m.status = 'ready';
      m.device = data.device;
      m.key = data.model;
      m.gpu = data.gpu;
      syncUI();
      break;
    case 'result': {
      const it = state.items.find((i) => i.id === data.id);
      if (state.busyId === data.id) state.busyId = null;
      if (it) {
        it.mask = maskToCanvas(data.mask, it.w, it.h);
        it.status = 'done';
        it.cutDirty = true;
        updateThumb(it);
        if (it.id === state.currentId) render();
      }
      syncUI();
      pump();
      break;
    }
    case 'error': {
      const it = state.items.find((i) => i.id === data.id);
      if (state.busyId === data.id) state.busyId = null;
      if (it) {
        it.status = 'error';
        it.error = data.message;
        updateThumb(it);
      }
      console.error(data.message);
      toast("Sorry, that image couldn't be processed.");
      pump();
      break;
    }
    case 'fatal':
      m.status = 'idle';
      console.error(data.message);
      toast("The AI model couldn't load. Check your connection and try again.", 6000);
      break;
  }
  updateStatus();
}

startWorker(state.quality);

function maskToCanvas(mask, w, h) {
  const c = makeCanvas(w, h);
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(w, h);
  const ch = Math.round(mask.length / (w * h)) || 1;
  for (let i = 0, j = 0; i < w * h; i++, j += ch) img.data[i * 4 + 3] = mask[j];
  ctx.putImageData(img, 0, 0);
  return c;
}

function pump() {
  if (state.busyId) return;
  const next = state.items.find((i) => i.status === 'queued');
  if (!next) return;
  state.busyId = next.id;
  next.status = 'processing';
  updateThumb(next);
  const data = next.src.getContext('2d').getImageData(0, 0, next.w, next.h);
  worker.postMessage({ type: 'run', id: next.id, width: next.w, height: next.h, buffer: data.data.buffer }, [data.data.buffer]);
  updateStatus();
}

/* ------------------------------------------------------------------ */
/* Adding images                                                       */
/* ------------------------------------------------------------------ */

async function addFiles(entries) {
  const list = entries.filter((e) => e.blob && e.blob.type.startsWith('image/'));
  if (!list.length) {
    toast('Please choose an image file (PNG, JPG or WebP).');
    return;
  }
  warmup();
  showEditor();
  let first = null;
  for (const { blob, name } of list) {
    try {
      const it = await createItem(blob, name);
      if (!first) {
        first = it;
        select(it.id);
      }
      pump();
    } catch (err) {
      console.error(err);
      toast(`Couldn't open ${name || 'that file'}.`);
    }
  }
  if (!state.items.length) goHome();
}

async function createItem(blob, name) {
  const bmp = await decode(blob);
  const srcW = bmp.width || bmp.naturalWidth;
  const srcH = bmp.height || bmp.naturalHeight;
  const { w, h } = fitSize(srcW, srcH);
  const src = makeCanvas(w, h);
  const sctx = src.getContext('2d');
  sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(bmp, 0, 0, w, h);
  bmp.close?.();

  const it = {
    id: nextId++,
    name: baseName(name),
    src,
    w,
    h,
    mask: null,
    cut: null,
    cutDirty: true,
    blurCache: null,
    status: 'queued',
    bg: { type: 'transparent', color: '#ffffff', image: null, blur: 14 },
    undo: [],
    thumb: null,
  };

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'thumb';
  btn.title = name || 'Image';
  btn.innerHTML = '<img alt="" />';
  btn.addEventListener('click', () => select(it.id));
  el.tray.insertBefore(btn, el.addMoreBtn);
  it.thumb = btn;

  state.items.push(it);
  updateThumb(it);
  return it;
}

function updateThumb(it) {
  if (!it.thumb) return;
  const size = 128;
  const s = Math.min(size / it.w, size / it.h);
  const c = makeCanvas(Math.round(it.w * s), Math.round(it.h * s));
  const ctx = c.getContext('2d');
  if (it.status === 'done') {
    paintBackground(ctx, it, c.width, c.height);
    ctx.drawImage(getCut(it), 0, 0, c.width, c.height);
  } else {
    ctx.drawImage(it.src, 0, 0, c.width, c.height);
  }
  it.thumb.querySelector('img').src = c.toDataURL('image/png');
  it.thumb.classList.toggle('busy', it.status === 'queued' || it.status === 'processing');
  it.thumb.classList.toggle('failed', it.status === 'error');
  it.thumb.classList.toggle('active', it.id === state.currentId);
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */

function getCut(it) {
  if (!it.cut) it.cut = makeCanvas(it.w, it.h);
  if (it.cutDirty) {
    const ctx = it.cut.getContext('2d');
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, it.w, it.h);
    ctx.drawImage(it.mask, 0, 0);
    ctx.globalCompositeOperation = 'source-in';
    ctx.drawImage(it.src, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    it.cutDirty = false;
  }
  return it.cut;
}

function getBlurred(it) {
  const amount = it.bg.blur;
  if (it.blurCache?.amount === amount) return it.blurCache.canvas;
  const c = makeCanvas(it.w, it.h);
  const ctx = c.getContext('2d');
  const radius = (amount / 1000) * Math.max(it.w, it.h);
  if ('filter' in ctx) {
    // Draw slightly oversized so blurred edges don't fade to transparent
    const pad = radius * 2;
    ctx.filter = `blur(${radius}px)`;
    ctx.drawImage(it.src, -pad, -pad, it.w + pad * 2, it.h + pad * 2);
    ctx.filter = 'none';
  } else {
    // Fallback: downscale + upscale
    const f = Math.max(2, radius / 2);
    const small = makeCanvas(Math.max(1, Math.round(it.w / f)), Math.max(1, Math.round(it.h / f)));
    small.getContext('2d').drawImage(it.src, 0, 0, small.width, small.height);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(small, 0, 0, it.w, it.h);
  }
  it.blurCache = { amount, canvas: c };
  return c;
}

function paintBackground(ctx, it, w, h, { forceOpaque = false } = {}) {
  const bg = it.bg;
  if (bg.type === 'color') {
    ctx.fillStyle = bg.color;
    ctx.fillRect(0, 0, w, h);
  } else if (bg.type === 'image' && bg.image) {
    const iw = bg.image.width;
    const ih = bg.image.height;
    const s = Math.max(w / iw, h / ih);
    ctx.drawImage(bg.image, (w - iw * s) / 2, (h - ih * s) / 2, iw * s, ih * s);
  } else if (bg.type === 'blur') {
    ctx.drawImage(getBlurred(it), 0, 0, w, h);
  } else if (forceOpaque) {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
  }
}

let rafPending = false;
function render() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    draw();
  });
}

function draw() {
  const it = current();
  if (!it) return;
  if (el.view.width !== it.w || el.view.height !== it.h) {
    el.view.width = it.w;
    el.view.height = it.h;
  }
  fitView(it);
  vctx.clearRect(0, 0, it.w, it.h);
  if (it.status !== 'done') {
    vctx.drawImage(it.src, 0, 0);
    el.compareHandle.hidden = true;
    return;
  }
  paintBackground(vctx, it, it.w, it.h);
  vctx.drawImage(getCut(it), 0, 0);

  if (state.compare) {
    const x = Math.round(it.w * state.split);
    vctx.save();
    vctx.beginPath();
    vctx.rect(0, 0, x, it.h);
    vctx.clip();
    vctx.clearRect(0, 0, x, it.h);
    vctx.drawImage(it.src, 0, 0);
    vctx.restore();
    el.compareHandle.hidden = false;
    el.compareHandle.style.left = `${state.split * 100}%`;
  } else {
    el.compareHandle.hidden = true;
  }
}

// Scale the preview to fill the stage (up or down) while keeping the aspect ratio
function fitView(it) {
  const stage = el.wrap.parentElement;
  const cs = getComputedStyle(stage);
  const availW = stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const boxH = cs.maxHeight !== 'none' ? parseFloat(cs.maxHeight) : stage.clientHeight;
  const availH = boxH - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  if (availW <= 0 || availH <= 0) return;
  const s = Math.min(availW / it.w, availH / it.h);
  el.view.style.width = `${Math.floor(it.w * s)}px`;
  el.view.style.height = `${Math.floor(it.h * s)}px`;
}
window.addEventListener('resize', render);

/* ------------------------------------------------------------------ */
/* Status overlay & UI sync                                            */
/* ------------------------------------------------------------------ */

function updateStatus() {
  const it = current();
  if (!it || it.status === 'done') {
    el.status.hidden = true;
    el.wrap.classList.remove('scanning');
    return;
  }
  el.status.hidden = false;
  const spinner = el.status.querySelector('.spinner');

  if (it.status === 'error') {
    spinner.hidden = true;
    el.wrap.classList.remove('scanning');
    el.statusText.textContent = "Couldn't remove the background";
    el.statusSub.textContent = 'Try another image or reload the page.';
    el.progress.hidden = true;
    return;
  }

  spinner.hidden = false;
  el.wrap.classList.add('scanning');
  const m = state.model;
  if (m.status !== 'ready' && m.total > 0 && m.loaded < m.total) {
    el.statusText.textContent = 'Getting the AI ready…';
    el.progress.hidden = false;
    el.progressBar.style.width = `${Math.min(100, (m.loaded / m.total) * 100).toFixed(1)}%`;
    el.statusSub.textContent = `${mb(m.loaded)} of ${mb(m.total)} MB · saved on your device for next time`;
  } else if (m.status !== 'ready') {
    el.statusText.textContent = 'Getting the AI ready…';
    el.progress.hidden = true;
    el.statusSub.textContent = 'This only takes long the first time.';
  } else if (it.status === 'queued') {
    el.statusText.textContent = 'Waiting in line…';
    el.progress.hidden = true;
    el.statusSub.textContent = '';
  } else {
    el.statusText.textContent = 'Removing background…';
    el.progress.hidden = true;
    el.statusSub.textContent =
      m.device === 'webgpu'
        ? 'Using your graphics card'
        : m.key === 'best'
          ? 'Best quality without a graphics card can take up to a minute.'
          : '';
  }
}

function syncUI() {
  const it = current();
  const done = it?.status === 'done';

  el.downloadBtn.disabled = !done;
  el.undoBtn.disabled = !done || !it.undo.length;
  el.compareBtn.disabled = !done;
  $$('.tool').forEach((b) => {
    b.disabled = !done && b.dataset.tool !== 'none';
    b.classList.toggle('active', b.dataset.tool === state.tool);
  });
  el.brushSizeWrap.hidden = state.tool === 'none';
  el.wrap.classList.toggle('painting', state.tool !== 'none' && done);
  el.compareBtn.setAttribute('aria-pressed', String(state.compare));

  const doneCount = state.items.filter((i) => i.status === 'done').length;
  el.downloadAllBtn.hidden = doneCount < 2;
  el.downloadAllBtn.textContent = `Download all (${doneCount})`;

  if (it) {
    el.dlSize.textContent = `${it.w} × ${it.h}${it.w >= 1920 || it.h >= 1080 ? ' · Full HD' : ''}`;
    $$('.bg-tab').forEach((t) => t.classList.toggle('active', t.dataset.bg === it.bg.type));
    $$('.bg-pane').forEach((p) => (p.hidden = p.dataset.pane !== it.bg.type));
    $$('.swatch').forEach((s) => s.classList.toggle('active', it.bg.type === 'color' && s.dataset.color === it.bg.color));
    el.blurAmount.value = it.bg.blur;
    el.bgImageBtn.textContent = it.bg.image ? 'Change background photo' : 'Choose a background photo';
  }
  $$('.fmt').forEach((b) => b.classList.toggle('active', b.dataset.fmt === state.format));

  const m = state.model;
  const shown = m.key || (state.quality === 'auto' ? null : state.quality);
  $$('.q-tab').forEach((b) => {
    b.classList.toggle('active', b.dataset.quality === shown);
    b.disabled = m.status === 'loading';
  });
  const hint = $('#qualityHint');
  if (!shown) hint.textContent = 'Picking the best option for your device…';
  else if (shown === 'best')
    hint.textContent =
      m.gpu === false
        ? 'Sharpest cutouts. No graphics card was found so each image may take up to a minute.'
        : 'Sharpest cutouts using your graphics card. One-time 220 MB download.';
  else hint.textContent = 'Smaller 46 MB download that works on any device. Edges are less precise.';
  state.items.forEach((i) => i.thumb?.classList.toggle('active', i.id === state.currentId));
}

function select(id) {
  state.currentId = id;
  syncUI();
  updateStatus();
  render();
}

function showEditor() {
  el.home.hidden = true;
  el.editor.hidden = false;
  window.scrollTo(0, 0);
}

function goHome() {
  for (const it of state.items) it.thumb?.remove();
  state.items = [];
  state.currentId = null;
  state.compare = false;
  setTool('none');
  el.editor.hidden = true;
  el.home.hidden = false;
}

function removeItem(id) {
  const idx = state.items.findIndex((i) => i.id === id);
  if (idx < 0) return;
  const [it] = state.items.splice(idx, 1);
  it.thumb?.remove();
  if (!state.items.length) return goHome();
  select(state.items[Math.min(idx, state.items.length - 1)].id);
}

/* ------------------------------------------------------------------ */
/* Erase / Restore brush and compare slider                            */
/* ------------------------------------------------------------------ */

function setTool(tool) {
  state.tool = tool;
  if (tool !== 'none') state.compare = false;
  el.brushCursor.hidden = true;
  syncUI();
  render();
}

function toCanvasPoint(e, it) {
  const r = el.view.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.width) * it.w,
    y: ((e.clientY - r.top) / r.height) * it.h,
    scale: it.w / r.width,
  };
}

function moveCursor(e) {
  const r = el.wrap.getBoundingClientRect();
  const size = Number(el.brushSize.value);
  Object.assign(el.brushCursor.style, {
    left: `${e.clientX - r.left}px`,
    top: `${e.clientY - r.top}px`,
    width: `${size}px`,
    height: `${size}px`,
  });
}

let stroke = null;
let dragging = false;

el.wrap.addEventListener('pointerdown', (e) => {
  const it = current();
  if (!it || it.status !== 'done') return;

  if (state.tool === 'none') {
    if (!state.compare) return;
    dragging = true;
    el.wrap.setPointerCapture(e.pointerId);
    setSplit(e);
    return;
  }

  e.preventDefault();
  el.wrap.setPointerCapture(e.pointerId);
  const mctx = it.mask.getContext('2d');
  it.undo.push(mctx.getImageData(0, 0, it.w, it.h));
  if (it.undo.length > UNDO_LIMIT) it.undo.shift();

  const p = toCanvasPoint(e, it);
  const width = Number(el.brushSize.value) * p.scale;
  mctx.globalCompositeOperation = state.tool === 'erase' ? 'destination-out' : 'source-over';
  mctx.fillStyle = mctx.strokeStyle = '#000';
  mctx.lineWidth = width;
  mctx.lineCap = 'round';
  mctx.lineJoin = 'round';
  mctx.beginPath();
  mctx.arc(p.x, p.y, width / 2, 0, Math.PI * 2);
  mctx.fill();
  stroke = { it, mctx, last: p };
  it.cutDirty = true;
  render();
  syncUI();
});

el.wrap.addEventListener('pointermove', (e) => {
  if (state.tool !== 'none' && current()?.status === 'done') {
    el.brushCursor.hidden = false;
    moveCursor(e);
  }
  if (dragging) return setSplit(e);
  if (!stroke) return;
  const p = toCanvasPoint(e, stroke.it);
  const { mctx, last } = stroke;
  mctx.beginPath();
  mctx.moveTo(last.x, last.y);
  mctx.lineTo(p.x, p.y);
  mctx.stroke();
  stroke.last = p;
  stroke.it.cutDirty = true;
  render();
});

function endPointer() {
  dragging = false;
  if (stroke) {
    stroke.mctx.globalCompositeOperation = 'source-over';
    updateThumb(stroke.it);
    stroke = null;
  }
}
el.wrap.addEventListener('pointerup', endPointer);
el.wrap.addEventListener('pointercancel', endPointer);
el.wrap.addEventListener('pointerleave', () => (el.brushCursor.hidden = true));

function setSplit(e) {
  const r = el.view.getBoundingClientRect();
  state.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  render();
}

function undo() {
  const it = current();
  if (!it?.undo.length) return;
  it.mask.getContext('2d').putImageData(it.undo.pop(), 0, 0);
  it.cutDirty = true;
  updateThumb(it);
  syncUI();
  render();
}

/* ------------------------------------------------------------------ */
/* Download                                                            */
/* ------------------------------------------------------------------ */

async function exportItem(it) {
  const c = makeCanvas(it.w, it.h);
  const ctx = c.getContext('2d');
  const jpg = state.format === 'jpg';
  paintBackground(ctx, it, it.w, it.h, { forceOpaque: jpg });
  ctx.drawImage(getCut(it), 0, 0);
  const blob = await new Promise((res) => c.toBlob(res, jpg ? 'image/jpeg' : 'image/png', 0.92));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${it.name}-cutout.${jpg ? 'jpg' : 'png'}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

/* ------------------------------------------------------------------ */
/* Event wiring                                                        */
/* ------------------------------------------------------------------ */

// Upload
el.uploadBtn.addEventListener('click', () => {
  warmup(); // start fetching the model while the user picks a file
  el.fileInput.click();
});
el.addMoreBtn.addEventListener('click', () => el.fileInput.click());
el.fileInput.addEventListener('change', () => {
  addFiles([...el.fileInput.files].map((f) => ({ blob: f, name: f.name })));
  el.fileInput.value = '';
});
el.dropzone.addEventListener('pointerenter', warmup, { once: true });

// Samples
$$('.sample').forEach((btn) =>
  btn.addEventListener('click', async () => {
    warmup();
    try {
      const res = await fetch(btn.dataset.src);
      const blob = await res.blob();
      addFiles([{ blob, name: 'sample' }]);
    } catch {
      toast("Couldn't load the sample image.");
    }
  }),
);

// Drag & drop anywhere
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  el.dropOverlay.hidden = false;
  el.dropzone.classList.add('drag');
});
window.addEventListener('dragover', (e) => hasFiles(e) && e.preventDefault());
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) {
    el.dropOverlay.hidden = true;
    el.dropzone.classList.remove('drag');
  }
});
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  el.dropOverlay.hidden = true;
  el.dropzone.classList.remove('drag');
  addFiles([...e.dataTransfer.files].map((f) => ({ blob: f, name: f.name })));
});

// Paste
window.addEventListener('paste', (e) => {
  const files = [...(e.clipboardData?.items || [])]
    .filter((i) => i.kind === 'file' && i.type.startsWith('image/'))
    .map((i) => ({ blob: i.getAsFile(), name: 'pasted' }));
  if (files.length) addFiles(files);
});

// Quality
$$('.q-tab').forEach((b) => b.addEventListener('click', () => setQuality(b.dataset.quality)));

// Tools
$$('.tool').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
el.brushSize.addEventListener('input', () => {
  const r = el.wrap.getBoundingClientRect();
  el.brushCursor.hidden = false;
  moveCursor({ clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 });
});
el.brushSize.addEventListener('change', () => (el.brushCursor.hidden = true));
el.undoBtn.addEventListener('click', undo);
el.compareBtn.addEventListener('click', () => {
  state.compare = !state.compare;
  if (state.compare) {
    state.tool = 'none';
    state.split = 0.5;
  }
  syncUI();
  render();
});

window.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !el.editor.hidden) {
    e.preventDefault();
    undo();
  }
});

// Background
el.swatches.innerHTML =
  SWATCHES.map((c) => `<button type="button" class="swatch" data-color="${c}" style="background:${c}" aria-label="Colour ${c}"></button>`).join('') +
  '<label class="swatch custom" title="Pick any colour"><input type="color" id="customColor" value="#5b4cf5" aria-label="Pick any colour" /></label>';

function setBg(patch) {
  const it = current();
  if (!it) return;
  Object.assign(it.bg, patch);
  syncUI();
  render();
  if (it.status === 'done') updateThumb(it);
}

$$('.bg-tab').forEach((t) =>
  t.addEventListener('click', () => {
    const type = t.dataset.bg;
    setBg({ type });
    if (type === 'image' && !current()?.bg.image) el.bgImageInput.click();
  }),
);
el.swatches.addEventListener('click', (e) => {
  const s = e.target.closest('.swatch[data-color]');
  if (s) setBg({ type: 'color', color: s.dataset.color });
});
$('#customColor').addEventListener('input', (e) => setBg({ type: 'color', color: e.target.value }));
el.bgImageBtn.addEventListener('click', () => el.bgImageInput.click());
el.bgImageInput.addEventListener('change', async () => {
  const f = el.bgImageInput.files[0];
  el.bgImageInput.value = '';
  if (!f) return;
  try {
    setBg({ type: 'image', image: await decode(f) });
  } catch {
    toast("Couldn't open that background image.");
  }
});
el.blurAmount.addEventListener('input', () => setBg({ type: 'blur', blur: Number(el.blurAmount.value) }));

// Download
$$('.fmt').forEach((b) =>
  b.addEventListener('click', () => {
    state.format = b.dataset.fmt;
    syncUI();
  }),
);
el.downloadBtn.addEventListener('click', () => {
  const it = current();
  if (it?.status === 'done') exportItem(it);
});
el.downloadAllBtn.addEventListener('click', async () => {
  for (const it of state.items.filter((i) => i.status === 'done')) {
    await exportItem(it);
    await new Promise((r) => setTimeout(r, 350));
  }
});

// Manage
el.deleteBtn.addEventListener('click', () => {
  const it = current();
  if (it) removeItem(it.id);
});
el.newBtn.addEventListener('click', goHome);
