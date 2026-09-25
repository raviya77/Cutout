# Cutout — free background remover

Remove the background from any photo, 100% free. The AI runs **in the visitor's browser**, so images never leave their device and there are no server costs.

## Features
- Upload, drag & drop, or paste (Ctrl+V) one or many images
- Automatic background removal (ormbg model, Apache-2.0 license)
- Replace the background with a colour, a photo, or a blur of the original
- Erase / Restore brushes with undo for touch-ups
- Before/after compare slider
- Download PNG (transparent) or JPG, up to Full HD (1920 × 1080)

## How it works
- Plain HTML/CSS/JS — no build step. Just static files.
- [Transformers.js](https://huggingface.co/docs/transformers.js) (loaded from the jsDelivr CDN) runs the model in a Web Worker.
- The model (`onnx-community/ormbg-ONNX`) downloads from Hugging Face on first use (~44 MB, or ~88 MB on WebGPU) and is cached by the browser.
- Uses WebGPU when available, otherwise WebAssembly on the CPU.

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
- Model: [ormbg](https://huggingface.co/schirrmacher/ormbg) — Apache-2.0
- Transformers.js — Apache-2.0
- Sample images are loaded from Pexels and Hugging Face at runtime.
