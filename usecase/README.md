# KGAU Biomedical Disease Prediction Demo

Local thesis demo for symptom-to-disease prediction with KGAU-trained embeddings on Hetionet. This is **not** a medical diagnosis tool.

The app loads checkpoints only. Training stays in the parent repository.

## Layout

```
usecase/
├── database/hetionet_subset/   # offline KG + metadata
├── model/                      # checkpoints + registry.json
├── backend/                    # FastAPI inference API
└── frontend/                   # React + Tailwind dashboard
```

Inference uses `entities.json` **list order** as the embedding row index. Do not use `entity2id.json` for lookup.

## Run

Install backend dependencies from the repo root (PyTorch should already be installed):

```bash
pip install -r usecase/backend/requirements.txt
```

Start the API (loads ComplEx-AU and precomputes t-SNE/UMAP once):

```bash
cd usecase/backend
python run.py
```

The API is at `http://127.0.0.1:8000`. Docs: `http://127.0.0.1:8000/docs`.

Start the UI:

```bash
cd usecase/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to the backend.

Optional: `npm run build` then restart the backend; FastAPI will serve `frontend/dist` at port 8000.

## Prediction

For each selected symptom (max 5):

1. Missing-head link prediction: `(Presents, symptom) → diseases` via `predict_head_rt_`
2. Keep the top 10 diseases per symptom (hard cutoff; ties do not add extra diseases)
3. Merge lists: frequency, min-max-normalized average/max score
4. Sort by frequency, then average score; return at most 50 diseases
5. Plot those candidates plus the input symptoms in a precomputed 2D projection

Similar diseases use `resemble_diseases.json` (graph lookup, no scoring).
