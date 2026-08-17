# KGAU Biomedical Disease Prediction Demo

Local thesis demo for **symptom-to-disease prediction** using KGAU-trained embeddings on a Hetionet subset.

This is **not** a medical diagnosis tool. The app loads checkpoints only; training stays in the parent repository.

## Layout

```
usecase/
├── database/hetionet_subset/   # offline KG + metadata
├── model/                      # checkpoints + registry.json
├── backend/                    # FastAPI inference API
└── frontend/                   # React + Tailwind dashboard
```

Inference uses `entities.json` **list order** as the embedding row index. Do not use `entity2id.json` for lookup.

## Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Node.js 18+** and npm (frontend only)
- **PyTorch** already installed in the same environment as the backend
- A cloned copy of this repository so the backend can import `models/` and `utils/` from the repo root

## 1. Create a virtual environment (recommended)

From the **repository root** (`KGDirectAU/`):

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

## 2. Install Python dependencies

PyTorch is **not** listed in `usecase/backend/requirements.txt`. Install it first, then the API packages.

```bash
# CPU-only example
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or CUDA 12.6 (pick the index that matches your driver)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -r usecase/backend/requirements.txt
```

If you already use this repo for training, installing from the root `requirements.txt` plus the usecase backend file is enough.

## 3. Start the backend

The API loads **ComplEx-AU** on startup and precomputes t-SNE / UMAP once. The first launch can take a minute.

```bash
cd usecase/backend
python run.py
```

| | URL |
| --- | --- |
| API | http://127.0.0.1:8000 |
| Health check | http://127.0.0.1:8000/api/health |
| Interactive docs | http://127.0.0.1:8000/docs |

Leave this terminal running.

## 4. Start the frontend

Open a **second** terminal (activate the same venv is not required for Node):

```bash
cd usecase/frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173**.

Vite proxies `/api` to `http://127.0.0.1:8000`, so both processes must be running during development.

## Optional: serve the UI from FastAPI

Build the frontend, then restart the backend. FastAPI serves `frontend/dist` on port 8000.

```bash
cd usecase/frontend
npm install
npm run build

cd ../backend
python run.py
```

Then open **http://127.0.0.1:8000** only. You do not need `npm run dev`.

## Using the demo

1. Add up to **5 symptoms**.
2. Click **Predict**.
3. Inspect ranked diseases, the 2D embedding plot (t-SNE or UMAP), and disease metadata.
4. Switch language (English / Vietnamese) from the header.

Prediction pipeline for each selected symptom:

1. Missing-head link prediction: `(Presents, symptom) → diseases`
2. Keep the top 10 diseases per symptom (hard cutoff; ties do not add extra diseases)
3. Merge lists: frequency, min-max-normalized average/max score
4. Sort by frequency, then average score; return at most 50 diseases
5. Plot those candidates plus the input symptoms in a precomputed 2D projection

Similar diseases use `resemble_diseases.json` (graph lookup, no scoring).

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Frontend loads but predict fails | Backend is not running, or it is not on port 8000 |
| `Unknown model` / model unavailable | Default checkpoint is `usecase/model/ComplEx-AU_best.mdl`. Other registry entries are marked unavailable until their files exist. |
| Slow first request / long startup | Normal: embeddings load and t-SNE/UMAP are computed once |
| Import errors for `models` or `utils` | Run `python run.py` from `usecase/backend` inside the cloned repo, not from a copied backend folder |
| Port already in use | Stop the other process on 8000 (API) or 5173 (Vite) |
