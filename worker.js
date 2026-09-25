// Runs the background-removal model off the main thread so the page never freezes.
import {
  AutoModel,
  AutoProcessor,
  RawImage,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js';

// ormbg — Apache-2.0 licensed, free for commercial use. Downloaded once, then cached by the browser.
const MODEL_ID = 'onnx-community/ormbg-ONNX';

env.allowLocalModels = false;

let model = null;
let processor = null;
let device = null;
let loading = null;

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

async function loadModel(forceCPU = false) {
  const useGPU = !forceCPU && (await hasFastGPU());
  const options = useGPU
    ? { device: 'webgpu', dtype: 'fp16' } // fast path on modern GPUs (~88 MB)
    : { device: 'wasm', dtype: 'q8' }; // works everywhere (~44 MB)
  postMessage({ type: 'loading', device: options.device });
  model = await AutoModel.from_pretrained(MODEL_ID, { ...options, progress_callback: onProgress });
  processor ??= await AutoProcessor.from_pretrained(MODEL_ID);
  device = options.device;
  postMessage({ type: 'ready', device });
}

function ensureLoaded() {
  if (!loading) {
    loading = loadModel().catch(async (err) => {
      console.warn('GPU load failed, falling back to CPU', err);
      await loadModel(true);
    });
  }
  return loading;
}

async function predict(image) {
  const { pixel_values } = await processor(image);
  const { alphas } = await model({ pixel_values });
  return alphas;
}

self.onmessage = async ({ data }) => {
  if (data.type === 'warmup') {
    try {
      await ensureLoaded();
    } catch (err) {
      postMessage({ type: 'fatal', message: String(err?.message || err) });
    }
    return;
  }

  if (data.type !== 'run') return;
  const { id, width, height, buffer } = data;
  try {
    await ensureLoaded();
    const image = new RawImage(new Uint8ClampedArray(buffer), width, height, 4).rgb();
    const started = performance.now();

    let alphas;
    try {
      alphas = await predict(image);
    } catch (err) {
      if (device !== 'webgpu') throw err;
      console.warn('GPU inference failed, retrying on CPU', err);
      loading = loadModel(true);
      await loading;
      alphas = await predict(image);
    }

    const mask = await RawImage.fromTensor(alphas[0].mul(255).to('uint8')).resize(width, height);
    const out = new Uint8ClampedArray(mask.data);
    postMessage(
      { type: 'result', id, mask: out, ms: Math.round(performance.now() - started), device },
      [out.buffer],
    );
  } catch (err) {
    postMessage({ type: 'error', id, message: String(err?.message || err) });
  }
};
