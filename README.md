# Cutout — free background remover

Remove the background from any photo, 100% free. The AI runs **in the visitor's browser**, so images never leave their device and there are no server costs.

## Features
- Upload, drag & drop, or paste (Ctrl+V) one or many images
- Automatic background removal with two AI models:
  - **Best**: BEN2 (MIT), sharp edges, handles illustrations, objects and hair. Used automatically when the device has a capable graphics card (WebGPU).
  - **Fast**: IS-Net general (Apache-2.0), 46 MB, works on any device.
- Replace the background with a colour, a photo, or a blur of the original
- Erase / Restore brushes with undo for touch-ups
- Before/after compare slider
- Download PNG (transparent) or JPG, up to Full HD (1920 × 1080)

## How it works
- Plain HTML/CSS/JS — no build step. Just static files.
- [Transformers.js](https://huggingface.co/docs/transformers.js) (loaded from the jsDelivr CDN) runs the model in a Web Worker.
- Models download from Hugging Face on first use and are cached by the browser:
  - `onnx-community/BEN2-ONNX` (fp16, ~220 MB), run on WebGPU
  - `Ko033/isnet-general-use-onnx` (8-bit, ~46 MB), run on the CPU with WebAssembly
- Visitors can switch between Best and Fast in the editor. The choice is remembered.

## Files
| File | What it does |
|---|---|
| `index.html` | Page layout (home + editor) |
| `styles.css` | All styling, light & dark mode |
| `app.js` | Uploading, editing, backgrounds, downloads |
| `worker.js` | Loads the AI model and removes backgrounds |

## Run locally
```bash
python3 -m http.server 8080
# open http://localhost:8080
```

## Deploy
Any static host works: GitHub Pages, Cloudflare Pages, Netlify, Vercel. Upload the folder as-is.

## Licenses
- Models: [BEN2](https://huggingface.co/PramaLLC/BEN2) (MIT), [IS-Net](https://github.com/xuebinqin/DIS) (Apache-2.0)
- Transformers.js — Apache-2.0
- Sample images are loaded from Pexels and Hugging Face at runtime.
