from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.inference.engine import InferenceEngine
from app.inference.registry import find_model, load_registry, model_cards
from app.kg.catalog import load_catalog
from app.paths import FRONTEND_DIST
from app.schemas import (
    DiseaseDetailOut,
    ModelListOut,
    PredictRequest,
    PredictResponse,
    SymptomOut,
)

catalog = load_catalog()
default_model_id, registered_models = load_registry()
engines: dict[str, InferenceEngine] = {}
current_model_id = default_model_id


def parse_lang(value: str | None) -> str:
    return "vi" if (value or "").strip().lower() in {"vi", "vn", "vietnamese"} else "en"


def get_engine(model_id: str | None = None) -> InferenceEngine:
    global current_model_id
    target = model_id or current_model_id or default_model_id
    try:
        registered = find_model(registered_models, target)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {target}") from exc
    if not registered.available:
        raise HTTPException(status_code=409, detail=f"Model {target} is not available yet.")
    if target not in engines:
        engines[target] = InferenceEngine(catalog, registered)
    current_model_id = target
    return engines[target]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_engine(default_model_id)
    yield


app = FastAPI(title="KGAU Biomedical Disease Prediction", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/symptoms", response_model=list[SymptomOut])
def list_symptoms(q: str | None = None, lang: str = "en"):
    lang = parse_lang(lang)
    query = (q or "").strip().lower()
    rows = []
    for item in catalog.symptoms:
        name = item.localized_name(lang)
        other = item.localized_name("en" if lang == "vi" else "vi")
        if not query or query in name.lower() or query in other.lower() or query in item.id.lower():
            rows.append(SymptomOut(id=item.id, name=name))
    rows.sort(key=lambda item: item.name.lower())
    return rows


@app.get("/api/models", response_model=ModelListOut)
def list_models():
    return ModelListOut(
        default_model_id=default_model_id,
        current_model_id=current_model_id,
        models=model_cards(registered_models),
    )


@app.post("/api/models/{model_id}/load", response_model=ModelListOut)
def load_model(model_id: str):
    get_engine(model_id)
    return list_models()


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    engine = get_engine(payload.model_id)
    lang = parse_lang(payload.lang)
    try:
        ranked, points, metrics = engine.retrieve(payload.symptom_ids, payload.projection, lang)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    focused = engine.disease_detail(ranked[0].id, ranked, payload.symptom_ids, lang) if ranked else None
    return PredictResponse(
        model=engine.card,
        projection=payload.projection,
        selected_symptoms=engine.symptom_out(payload.symptom_ids, lang),
        candidates=engine.candidates_out(ranked, lang),
        points=points,
        focused=focused,
        symptom_metrics=metrics,
    )


@app.get("/api/diseases/{disease_id}", response_model=DiseaseDetailOut)
def disease_detail(disease_id: str, symptoms: str = "", lang: str = "en"):
    lang = parse_lang(lang)
    entity = catalog.id_to_entity.get(disease_id)
    if entity is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown entity: {disease_id}" if lang != "vi" else f"Không tìm thấy thực thể: {disease_id}",
        )
    selected_ids = [item for item in symptoms.split(",") if item]
    engine = get_engine()
    ranked = None
    if entity.kind == "disease" and selected_ids:
        ranked, _, _ = engine.retrieve(selected_ids, lang=lang)
    return engine.disease_detail(disease_id, ranked, selected_ids, lang)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
