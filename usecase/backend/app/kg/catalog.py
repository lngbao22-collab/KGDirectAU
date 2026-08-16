from __future__ import annotations

import csv
import json

from collections import defaultdict

from app.paths import (
    DISEASE_METADATA,
    DISEASE_METADATA_VN,
    ENTITIES_JSON,
    GROUND_TRUTH,
    RESEMBLE_JSON,
    RELATION2ID_JSON,
    SYMPTOM_METADATA,
    SYMPTOM_METADATA_VN,
)

PRESENTS_RELATION = "Disease:presents:Symptom"
RESEMBLES_RELATION = "Disease:resembles:Disease"


class EntityRecord:
    __slots__ = ("id", "index", "kind", "name", "description", "wiki_url", "name_vi", "description_vi")

    def __init__(
        self,
        id: str,
        index: int,
        kind: str,
        name: str,
        description: str,
        wiki_url: str,
        name_vi: str = "",
        description_vi: str = "",
    ):
        self.id = id
        self.index = index
        self.kind = kind
        self.name = name
        self.description = description
        self.wiki_url = wiki_url
        self.name_vi = name_vi or name
        self.description_vi = description_vi or description

    def localized_name(self, lang: str = "en") -> str:
        return self.name_vi if lang == "vi" else self.name

    def localized_description(self, lang: str = "en") -> str:
        return self.description_vi if lang == "vi" else self.description


class Catalog:
    def __init__(
        self,
        entities: list[EntityRecord],
        relation_to_idx: dict[str, int],
        resembles: dict[str, list[str]],
        symptom_to_diseases: dict[str, list[str]] | None = None,
    ):
        self.entities = entities
        self.id_to_entity = {entity.id: entity for entity in entities}
        self.diseases = [entity for entity in entities if entity.kind == "disease"]
        self.symptoms = [entity for entity in entities if entity.kind == "symptom"]
        self.disease_indices = [entity.index for entity in self.diseases]
        self.relation_to_idx = relation_to_idx
        self.resembles = resembles
        self.symptom_to_diseases = symptom_to_diseases or {}
        self.presents_edges, self.disease_to_symptoms, self.resembles_edges = _index_edges(
            self.symptom_to_diseases,
            self.resembles,
            set(self.id_to_entity),
        )

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    def get(self, entity_id: str) -> EntityRecord:
        return self.id_to_entity[entity_id]


def _entity_kind(entity_id: str) -> str:
    if entity_id.startswith("DOID:"):
        return "disease"
    return "symptom"


def _load_metadata(path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entity_id = (row.get("id") or "").strip()
            if entity_id:
                rows[entity_id] = row
    return rows


def _display_name(entity_id: str, meta: dict[str, str] | None) -> str:
    if not meta:
        return entity_id
    name = (meta.get("name") or "").strip()
    if not name:
        return entity_id
    if entity_id.startswith("DOID:"):
        return name[:1].upper() + name[1:]
    return name


def _index_edges(
    symptom_to_diseases: dict[str, list[str]],
    resembles: dict[str, list[str]],
    known_ids: set[str],
) -> tuple[list[tuple[str, str]], dict[str, list[str]], list[tuple[str, str]]]:
    presents: list[tuple[str, str]] = []
    disease_to_symptoms: dict[str, list[str]] = defaultdict(list)
    for symptom_id, disease_ids in symptom_to_diseases.items():
        if symptom_id not in known_ids:
            continue
        for disease_id in disease_ids:
            if disease_id not in known_ids:
                continue
            presents.append((disease_id, symptom_id))
            disease_to_symptoms[disease_id].append(symptom_id)
    resemble_pairs: set[tuple[str, str]] = set()
    for source_id, target_ids in resembles.items():
        if source_id not in known_ids:
            continue
        for target_id in target_ids:
            if target_id not in known_ids or target_id == source_id:
                continue
            pair = (source_id, target_id) if source_id < target_id else (target_id, source_id)
            resemble_pairs.add(pair)
    return (
        presents,
        {key: list(dict.fromkeys(value)) for key, value in disease_to_symptoms.items()},
        sorted(resemble_pairs),
    )


def _is_positive_label(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        return int(float(text)) == 1
    except ValueError:
        return False


def _load_ground_truth(path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            # head, relation, tail, label — keep only positive (label=1) edges
            if len(parts) < 4 or not _is_positive_label(parts[3]):
                continue
            disease_id, symptom_id = parts[0].strip(), parts[2].strip()
            if disease_id and symptom_id:
                mapping[symptom_id].append(disease_id)
    return {key: list(dict.fromkeys(value)) for key, value in mapping.items()}


def load_catalog() -> Catalog:
    raw_entities = json.loads(ENTITIES_JSON.read_text(encoding="utf-8"))
    relation_to_idx = {
        str(key): int(value)
        for key, value in json.loads(RELATION2ID_JSON.read_text(encoding="utf-8")).items()
    }
    disease_meta = _load_metadata(DISEASE_METADATA)
    symptom_meta = _load_metadata(SYMPTOM_METADATA)
    disease_meta_vi = _load_metadata(DISEASE_METADATA_VN)
    symptom_meta_vi = _load_metadata(SYMPTOM_METADATA_VN)
    raw_resembles = json.loads(RESEMBLE_JSON.read_text(encoding="utf-8")) if RESEMBLE_JSON.exists() else {}
    resembles = {str(key): [str(item) for item in value] for key, value in raw_resembles.items()}

    entities: list[EntityRecord] = []
    for index, item in enumerate(raw_entities):
        entity_id = item["entity_id"]
        kind = _entity_kind(entity_id)
        meta = disease_meta.get(entity_id) if kind == "disease" else symptom_meta.get(entity_id)
        meta_vi = disease_meta_vi.get(entity_id) if kind == "disease" else symptom_meta_vi.get(entity_id)
        name = _display_name(entity_id, meta)
        description = ((meta or {}).get("description") or "Description not found.").strip()
        entities.append(
            EntityRecord(
                id=entity_id,
                index=index,
                kind=kind,
                name=name,
                description=description,
                wiki_url=((meta or {}).get("wiki_url") or "").strip(),
                name_vi=_display_name(entity_id, meta_vi) if meta_vi else name,
                description_vi=((meta_vi or {}).get("description") or description).strip(),
            )
        )
    known_ids = {entity.id for entity in entities}
    symptom_to_diseases = {
        symptom_id: [disease_id for disease_id in disease_ids if disease_id in known_ids]
        for symptom_id, disease_ids in _load_ground_truth(GROUND_TRUTH).items()
    }
    return Catalog(entities, relation_to_idx, resembles, symptom_to_diseases)
