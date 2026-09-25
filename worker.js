// Runs the background-removal AI off the main thread so the page never freezes.
import {
  AutoModel,
  AutoProcessor,
  RawImage,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js';

// Both models are free for commercial use. Each downloads once, then the browser caches it.
const MODELS = {
  // BEN2 (MIT): best quality: clean edges, handles illustrations, objects and hair.
  // Graph optimisation is kept at 'basic': the fully optimised graph fuses a LayerNorm
  // kernel that fails to compile on some GPUs (e.g. Apple Silicon in Chrome).
  best: { id: 'onnx-community/BEN2-ONNX', dtype: 'fp16', options: { graphOptimizationLevel: 'basic' } },
  // IS-Net general (Apache-2.0): lighter fallback for devices without a usable GPU.
  light: { id: 'Ko033/isnet-general-use-onnx', dtype: 'q8' },
};

env.allowLocalModels = false;

let model = null;
let processor = null;
let active = null; // key of the loaded model
let loading = null;
let preference = 'auto'; // 'auto' | 'best' | 'light'

async function hasFastGPU() {
  try {
    if (!self.navigator?.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter && adapter.features.has('shader-f16');
  } catch {
    return false;
  }
}

function onProgress(p) {
  if (p.status === 'progress' && p.file?.endsWith('.onnx')) {
    postMessage({ type: 'progress', loaded: p.loaded, total: p.total });
  }
}

async function pickModel() {
  const gpu = await hasFastGPU();
  if (preference === 'light') return { key: 'light', device: 'wasm' };
  if (preference === 'best') return { key: 'best', device: gpu ? 'webgpu' : 'wasm' };
  return gpu ? { key: 'best', device: 'webgpu' } : { key: 'light', device: 'wasm' };
}

async function loadModel(choice) {
  const cfg = MODELS[choice.key];
  postMessage({ type: 'loading', model: choice.key, device: choice.device });
  const m = await AutoModel.from_pretrained(cfg.id, {
    dtype: cfg.dtype,
    device: choice.device,
    session_options: cfg.options,
    progress_callback: onProgress,
  });
  const p = await AutoProcessor.from_pretrained(cfg.id);
  await model?.dispose?.();
  model = m;
  processor = p;
  active = choice.key;
  postMessage({ type: 'ready', model: choice.key, device: choice.device, gpu: await hasFastGPU() });
}

function ensureLoaded() {
  if (!loading) {
    loading = (async () => {
      const choice = await pickModel();
      try {
        await loadModel(choice);
      } catch (err) {
        if (choice.key === 'light') throw err;
        // A failed GPU session can leave the runtime in a bad state, so ask the page
        // to start a fresh worker with the light model instead of retrying here.
        throw Object.assign(err, { gpuFailed: true });
      }
    })();
    loading.catch(() => (loading = null));
  }
  return loading;
}

async function predict(image) {
  const { pixel_values } = await processor(image);
  const session = Object.values(model.sessions)[0];
  const outputs = await model({ [session.inputNames[0]]: pixel_values });
  let t = outputs[session.outputNames[0]];
  if (t.dims.length === 4) t = t[0]; // [1, 1, H, W] -> [1, H, W]
  // Some models return logits instead of 0..1 values
  const d = t.data;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < d.length; i += 257) {
    if (Number.isNaN(d[i])) throw new Error('Model returned an invalid mask');
    if (d[i] < min) min = d[i];
    if (d[i] > max) max = d[i];
  }
  if (min < -0.01 || max > 1.01) t = t.sigmoid();
  return t;
}

self.onmessage = async ({ data }) => {
  if (data.type === 'config') {
    if (data.preference !== preference) {
      preference = data.preference;
      loading = null; // reload with the new choice on the next image
    }
    return;
  }

  if (data.type === 'warmup') {
    try {
      await ensureLoaded();
    } catch (err) {
      const message = String(err?.message || err);
      postMessage(err?.gpuFailed ? { type: 'gpu-failed', message } : { type: 'fatal', message });
    }
    return;
  }

  if (data.type !== 'run') return;
  const { id, width, height, buffer } = data;
  try {
    await ensureLoaded();
    const image = new RawImage(new Uint8ClampedArray(buffer), width, height, 4).rgb();
    const started = performance.now();

    let alpha;
    try {
      alpha = await predict(image);
    } catch (err) {
      if (active === 'light') throw err;
      throw Object.assign(err, { gpuFailed: true });
    }

    const mask = await RawImage.fromTensor(alpha.mul(255).clamp(0, 255).to('uint8')).resize(width, height);
    const out = new Uint8ClampedArray(mask.data);
    postMessage(
      { type: 'result', id, mask: out, ms: Math.round(performance.now() - started), model: active },
      [out.buffer],
    );
  } catch (err) {
    const message = String(err?.message || err);
    postMessage(err?.gpuFailed ? { type: 'gpu-failed', id, message } : { type: 'error', id, message });
  }
};
