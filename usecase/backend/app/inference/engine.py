from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from app.inference.projector import project_entities
from app.inference.registry import RegisteredModel, _as_card
from app.kg.catalog import PRESENTS_RELATION, Catalog
from app.paths import REPO_ROOT
from app.schemas import (
    CandidateOut,
    DiseaseDetailOut,
    MatchedSymptomOut,
    ScatterPointOut,
    SimilarDiseaseOut,
    SymptomOut,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.complex import ComplExModel, ComplExScorer
from models.embedders.lookup_embedder import (
    ComplExEntityEmbedder,
    ComplExRelationEmbedder,
    LookupEmbedder,
)
from utils.checkpoint import load_checkpoint

TOP_K_PER_SYMPTOM = 10
MAX_CANDIDATES = 50


def _minmax(values: np.ndarray) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.ones_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _top_k_indices(scores: np.ndarray, diseases: list, k: int) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        return finite
    finite_scores = scores[finite]
    tie_break = np.array([diseases[int(index)].id for index in finite])
    order = np.lexsort((tie_break, -finite_scores))
    return finite[order[:k]]


def _build_complex(state: dict) -> ComplExModel:
    ent_shape = tuple(state["ent_embedder.ent_re.embedding.weight"].shape)
    rel_shape = tuple(state["rel_embedder.rel_re.embedding.weight"].shape)
    n_ent, ent_dim = int(ent_shape[0]), int(ent_shape[1])
    n_rel, rel_dim = int(rel_shape[0]), int(rel_shape[1])
    ent = ComplExEntityEmbedder(
        LookupEmbedder(n_ent, ent_dim, args=None, role="entity"),
        LookupEmbedder(n_ent, ent_dim, args=None, role="entity"),
    )
    rel = ComplExRelationEmbedder(
        LookupEmbedder(n_rel, rel_dim, args=None, role="relation"),
        LookupEmbedder(n_rel, rel_dim, args=None, role="relation"),
    )
    model = ComplExModel(ent, rel, scorers=[ComplExScorer(None)], args=None)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@dataclass
class RankedDisease:
    id: str
    frequency: int
    avg_similarity: float
    max_similarity: float
    matched: list[tuple[str, float]]


class InferenceEngine:
    def __init__(self, catalog: Catalog, registered: RegisteredModel):
        self.catalog = catalog
        self.registered = registered
        checkpoint = load_checkpoint(str(registered.checkpoint_path), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        self.model = _build_complex(state).to("cpu")
        with torch.no_grad():
            self.entity_vectors = self.model.embed_all_entities().cpu()
        if self.entity_vectors.size(0) != catalog.n_entities:
            raise ValueError(
                f"Checkpoint entity table has {self.entity_vectors.size(0)} rows, "
                f"but entities.json has {catalog.n_entities}. Use entities.json order."
            )
        vectors = self.entity_vectors.numpy()
        self.projections = {
            "umap": project_entities(vectors, "umap"),
            "tsne": project_entities(vectors, "tsne"),
        }
        self.presents_idx = int(catalog.relation_to_idx[PRESENTS_RELATION])
        self.disease_index_tensor = torch.tensor(catalog.disease_indices, dtype=torch.long)
        self.card = _as_card(registered)

    def _score_symptom(self, symptom_index: int) -> np.ndarray:
        relation = torch.tensor([self.presents_idx], dtype=torch.long)
        tail = torch.tensor([symptom_index], dtype=torch.long)
        with torch.no_grad():
            scores = self.model.predict_head_rt_(
                relation,
                tail,
                all_h_embs=self.entity_vectors,
            )[0].cpu().numpy()
        masked = np.full(scores.shape, -np.inf, dtype=np.float32)
        disease_idx = np.asarray(self.catalog.disease_indices, dtype=np.int64)
        masked[disease_idx] = scores[disease_idx]
        return masked

    def retrieve(
        self,
        symptom_ids: list[str],
        projection: str = "tsne",
        lang: str = "en",
    ) -> tuple[list[RankedDisease], list[ScatterPointOut]]:
        if not symptom_ids:
            raise ValueError("Select at least one symptom." if lang != "vi" else "Chọn ít nhất một triệu chứng.")
        if len(symptom_ids) > 5:
            raise ValueError("Select at most five symptoms." if lang != "vi" else "Chọn tối đa năm triệu chứng.")

        per_disease: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for symptom_id in symptom_ids:
            symptom = self.catalog.get(symptom_id)
            if symptom.kind != "symptom":
                raise ValueError(
                    f"{symptom_id} is not a symptom in the demo vocabulary."
                    if lang != "vi"
                    else f"{symptom_id} không phải triệu chứng trong từ vựng demo."
                )
            raw = self._score_symptom(symptom.index)
            disease_raw = raw[self.catalog.disease_indices]
            finite = np.isfinite(disease_raw)
            scaled = np.zeros_like(disease_raw, dtype=np.float32)
            if finite.any():
                scaled[finite] = _minmax(disease_raw[finite])
            ranked_local = _top_k_indices(disease_raw, self.catalog.diseases, TOP_K_PER_SYMPTOM)
            for local_rank in ranked_local:
                disease = self.catalog.diseases[int(local_rank)]
                per_disease[disease.id].append((symptom_id, float(scaled[local_rank])))

        ranked: list[RankedDisease] = []
        for disease_id, matches in per_disease.items():
            scores = [score for _, score in matches]
            ranked.append(
                RankedDisease(
                    id=disease_id,
                    frequency=len(matches),
                    avg_similarity=float(np.mean(scores)),
                    max_similarity=float(np.max(scores)),
                    matched=sorted(matches, key=lambda item: -item[1]),
                )
            )
        ranked.sort(key=lambda item: (-item.frequency, -item.avg_similarity, item.id))
        ranked = ranked[:MAX_CANDIDATES]

        coords = self.projections["tsne" if projection == "tsne" else "umap"]
        candidate_ids = {item.id for item in ranked}
        points: list[ScatterPointOut] = []
        for symptom_id in symptom_ids:
            symptom = self.catalog.get(symptom_id)
            truth_ids = [
                disease_id
                for disease_id in self.catalog.symptom_to_diseases.get(symptom_id, [])
                if disease_id in candidate_ids
            ]
            points.append(
                ScatterPointOut(
                    id=symptom.id,
                    name=symptom.localized_name(lang),
                    kind="symptom",
                    x=float(coords[symptom.index, 0]),
                    y=float(coords[symptom.index, 1]),
                    symptom_ids=[symptom.id],
                    ground_truth_ids=truth_ids,
                )
            )
        for rank, item in enumerate(ranked, start=1):
            disease = self.catalog.get(item.id)
            matched_ids = {symptom_id for symptom_id, _ in item.matched}
            points.append(
                ScatterPointOut(
                    id=disease.id,
                    name=disease.localized_name(lang),
                    kind="disease",
                    x=float(coords[disease.index, 0]),
                    y=float(coords[disease.index, 1]),
                    frequency=item.frequency,
                    avg_similarity=item.avg_similarity,
                    rank=rank,
                    symptom_ids=[symptom_id for symptom_id in symptom_ids if symptom_id in matched_ids],
                )
            )
        return ranked, points

    def candidates_out(self, ranked: list[RankedDisease], lang: str = "en") -> list[CandidateOut]:
        rows: list[CandidateOut] = []
        for rank, item in enumerate(ranked, start=1):
            disease = self.catalog.get(item.id)
            rows.append(
                CandidateOut(
                    rank=rank,
                    id=disease.id,
                    name=disease.localized_name(lang),
                    frequency=item.frequency,
                    avg_similarity=round(item.avg_similarity, 4),
                    max_similarity=round(item.max_similarity, 4),
                    matched_symptoms=[
                        MatchedSymptomOut(
                            id=symptom_id,
                            name=self.catalog.get(symptom_id).localized_name(lang),
                            score=round(score, 4),
                        )
                        for symptom_id, score in item.matched
                    ],
                )
            )
        return rows

    def disease_detail(
        self,
        disease_id: str,
        ranked: list[RankedDisease] | None,
        selected_ids: list[str],
        lang: str = "en",
    ) -> DiseaseDetailOut:
        disease = self.catalog.get(disease_id)
        if disease.kind == "symptom":
            return DiseaseDetailOut(
                id=disease.id,
                name=disease.localized_name(lang),
                kind="symptom",
                description=disease.localized_description(lang),
                wiki_url=disease.wiki_url,
                hetionet_id=disease.id,
            )
        matched: list[MatchedSymptomOut] = []
        if ranked:
            for item in ranked:
                if item.id == disease_id:
                    matched = [
                        MatchedSymptomOut(
                            id=symptom_id,
                            name=self.catalog.get(symptom_id).localized_name(lang),
                            score=round(score, 4),
                        )
                        for symptom_id, score in item.matched
                    ]
                    break
        similar: list[SimilarDiseaseOut] = []
        for neighbor_id in self.catalog.resembles.get(disease_id, []):
            neighbor = self.catalog.id_to_entity.get(neighbor_id)
            if neighbor is None:
                continue
            similar.append(SimilarDiseaseOut(id=neighbor.id, name=neighbor.localized_name(lang)))
        return DiseaseDetailOut(
            id=disease.id,
            name=disease.localized_name(lang),
            kind="disease",
            description=disease.localized_description(lang),
            wiki_url=disease.wiki_url,
            hetionet_id=disease.id,
            matched_symptoms=matched,
            matched_count=len(matched),
            selected_count=len(selected_ids),
            similar_diseases=similar,
        )

    def symptom_out(self, symptom_ids: list[str], lang: str = "en") -> list[SymptomOut]:
        return [
            SymptomOut(id=item.id, name=item.localized_name(lang))
            for item in (self.catalog.get(symptom_id) for symptom_id in symptom_ids)
        ]
