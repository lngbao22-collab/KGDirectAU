from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SymptomOut(BaseModel):
    id: str
    name: str


class ModelCardOut(BaseModel):
    id: str
    label: str
    available: bool
    framework: str = "KGAU"
    backbone: str = ""
    embedding_dim: int | None = None
    training_strategy: str = ""
    negative_sampling: str = ""
    best_epoch: int | None = None
    checkpoint: str = ""
    tuni: float | None = None
    gamma_q: float | None = None
    gamma_t: float | None = None
    gamma_ent: float | None = None
    head_eval_mode: str = "rt_forward"


class ModelListOut(BaseModel):
    default_model_id: str
    current_model_id: str
    models: list[ModelCardOut]


class PredictRequest(BaseModel):
    symptom_ids: list[str] = Field(min_length=1, max_length=5)
    model_id: str | None = None
    projection: Literal["umap", "tsne"] = "tsne"
    lang: Literal["en", "vi"] = "en"


class MatchedSymptomOut(BaseModel):
    id: str
    name: str
    score: float


class CandidateOut(BaseModel):
    rank: int
    id: str
    name: str
    frequency: int
    avg_similarity: float
    max_similarity: float
    matched_symptoms: list[MatchedSymptomOut]


class ScatterPointOut(BaseModel):
    id: str
    name: str
    kind: Literal["symptom", "disease"]
    x: float
    y: float
    frequency: int | None = None
    avg_similarity: float | None = None
    rank: int | None = None
    symptom_ids: list[str] = []
    ground_truth_ids: list[str] = []


class SimilarDiseaseOut(BaseModel):
    id: str
    name: str


class DiseaseDetailOut(BaseModel):
    id: str
    name: str
    kind: Literal["disease", "symptom"] = "disease"
    description: str
    wiki_url: str
    hetionet_id: str
    matched_symptoms: list[MatchedSymptomOut] = []
    matched_count: int = 0
    selected_count: int = 0
    similar_diseases: list[SimilarDiseaseOut] = []


class PredictResponse(BaseModel):
    model: ModelCardOut
    projection: Literal["umap", "tsne"]
    selected_symptoms: list[SymptomOut]
    candidates: list[CandidateOut]
    points: list[ScatterPointOut]
    focused: DiseaseDetailOut | None = None
