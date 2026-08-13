"""Filter Hetionet to Disease:presents:Symptom triples only."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Dict, Iterable, List, Sequence, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PRESENT_RELATION = "Disease:presents:Symptom"
RESEMBLE_RELATION = "Disease:resembles:Disease"
KEEP_RELATIONS = {PRESENT_RELATION}

SPLIT_FILES = (
    "train.txt",
    "valid.txt",
    "test.txt",
    "valid_w_label.txt",
    "test_w_label.txt",
)

# Entity vocabulary is built from positive KG splits only.
ENTITY_SOURCE_SPLITS = ("train.txt", "valid.txt", "test.txt")


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as reader:
        return reader.readlines()


def _parse_triple_line(line: str, path: str) -> Tuple[str, str, str, str]:
    fields = line.strip().split("\t")
    if len(fields) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 tab-separated fields in {path}: {line.strip()}")
    head, relation, tail = fields[:3]
    label = fields[3] if len(fields) == 4 else ""
    return head, relation, tail, label


def _load_filtered_split(path: str) -> List[Tuple[str, str, str, str]]:
    triples: List[Tuple[str, str, str, str]] = []
    for line in _read_lines(path):
        if not line.strip():
            continue
        head, relation, tail, label = _parse_triple_line(line, path)
        if relation in KEEP_RELATIONS:
            triples.append((head, relation, tail, label))
    return triples


def _collect_typed_entities(
    splits: Dict[str, Sequence[Tuple[str, str, str, str]]],
) -> Tuple[List[str], List[str], List[str]]:
    """Collect disease/symptom entities from positive KG triples only."""

    diseases: Set[str] = set()
    symptoms: Set[str] = set()

    for split_name in ENTITY_SOURCE_SPLITS:
        for head, relation, tail, _ in splits[split_name]:
            if relation == PRESENT_RELATION:
                diseases.add(head)
                symptoms.add(tail)
            elif relation == RESEMBLE_RELATION:
                diseases.add(head)
                diseases.add(tail)

    disease_list = sorted(diseases)
    symptom_list = sorted(symptoms)
    entities = sorted(diseases | symptoms)
    return entities, disease_list, symptom_list


def _count_relations(splits: Dict[str, Sequence[Tuple[str, str, str, str]]]) -> Tuple[int, int]:
    present = 0
    resemble = 0
    for split_name in ENTITY_SOURCE_SPLITS:
        for _, relation, _, _ in splits[split_name]:
            if relation == PRESENT_RELATION:
                present += 1
            elif relation == RESEMBLE_RELATION:
                resemble += 1
    return present, resemble


def _write_dict(path: str, values: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as writer:
        for idx, value in enumerate(values):
            writer.write(f"{idx}\t{value}\n")


def _write_split(path: str, triples: Sequence[Tuple[str, str, str, str]], *, with_label: bool) -> None:
    with open(path, "w", encoding="utf-8") as writer:
        for head, relation, tail, label in triples:
            if with_label:
                writer.write(f"{head}\t{relation}\t{tail}\t{label}\n")
            else:
                writer.write(f"{head}\t{relation}\t{tail}\n")


def _write_statistics(
    path: str,
    *,
    num_entities: int,
    num_diseases: int,
    num_symptoms: int,
    num_relations: int,
    num_triples: int,
    train_size: int,
    valid_size: int,
    valid_w_label_size: int,
    test_size: int,
    test_w_label_size: int,
    present_count: int,
    resemble_count: int,
) -> None:
    fieldnames = [
        "Tập dữ liệu",
        "Số thực thể",
        "Số thực thể bệnh",
        "Số thực thể triệu chứng",
        "Số quan hệ",
        "Số bộ ba",
        "Tập huấn luyện",
        "Tập xác thực",
        "Tập xác thực có nhãn",
        "Tập kiểm thử",
        "Tập kiểm thử có nhãn",
        "Số bộ ba present",
        "Số bộ ba resemble",
    ]
    row = {
        "Tập dữ liệu": "hetionet_subset",
        "Số thực thể": num_entities,
        "Số thực thể bệnh": num_diseases,
        "Số thực thể triệu chứng": num_symptoms,
        "Số quan hệ": num_relations,
        "Số bộ ba": num_triples,
        "Tập huấn luyện": train_size,
        "Tập xác thực": valid_size,
        "Tập xác thực có nhãn": valid_w_label_size,
        "Tập kiểm thử": test_size,
        "Tập kiểm thử có nhãn": test_w_label_size,
        "Số bộ ba present": present_count,
        "Số bộ ba resemble": resemble_count,
    }
    with open(path, "w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerow(row)


def preprocess_hetionet(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    filtered_splits: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for split_name in SPLIT_FILES:
        split_path = os.path.join(input_dir, split_name)
        if not os.path.exists(split_path):
            raise FileNotFoundError(split_path)
        filtered = _load_filtered_split(split_path)
        filtered_splits[split_name] = filtered
        logger.info("Kept %d filtered triples for %s", len(filtered), split_name)

    entities, diseases, symptoms = _collect_typed_entities(filtered_splits)
    if set(diseases) & set(symptoms):
        raise ValueError("Disease and symptom entity sets should be disjoint")
    if len(entities) != len(diseases) + len(symptoms):
        raise ValueError("Total entities must equal disease + symptom counts")

    relations = sorted(KEEP_RELATIONS)
    present_count, resemble_count = _count_relations(filtered_splits)

    entities_path = os.path.join(output_dir, "entities.dict")
    relations_path = os.path.join(output_dir, "relations.dict")
    _write_dict(entities_path, entities)
    _write_dict(relations_path, relations)
    logger.info("Wrote %d entities to %s", len(entities), entities_path)
    logger.info("Wrote %d disease / %d symptom entities", len(diseases), len(symptoms))
    logger.info("Wrote %d relations to %s", len(relations), relations_path)

    for split_name, triples in filtered_splits.items():
        out_path = os.path.join(output_dir, split_name)
        _write_split(out_path, triples, with_label=split_name.endswith("_w_label.txt"))
        logger.info("Wrote %d triples to %s", len(triples), out_path)

    train_size = len(filtered_splits["train.txt"])
    valid_size = len(filtered_splits["valid.txt"])
    valid_w_label_size = len(filtered_splits["valid_w_label.txt"])
    test_size = len(filtered_splits["test.txt"])
    test_w_label_size = len(filtered_splits["test_w_label.txt"])
    stats_path = os.path.join(output_dir, "statistics.csv")
    _write_statistics(
        stats_path,
        num_entities=len(entities),
        num_diseases=len(diseases),
        num_symptoms=len(symptoms),
        num_relations=len(relations),
        num_triples=train_size + valid_size + test_size,
        train_size=train_size,
        valid_size=valid_size,
        valid_w_label_size=valid_w_label_size,
        test_size=test_size,
        test_w_label_size=test_w_label_size,
        present_count=present_count,
        resemble_count=resemble_count,
    )
    logger.info("Wrote statistics to %s", stats_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create hetionet_subset keeping only Disease:presents:Symptom triples."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join("data", "hetionet"),
        type=str,
        help="directory containing raw Hetionet split files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "hetionet_subset"),
        type=str,
        help="directory where the filtered subset will be written",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preprocess_hetionet(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
