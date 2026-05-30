"""Singleton-like hub for loading and caching KG data structures."""

import os
import glob
from transformers import AutoTokenizer

from configs.config import args
from utils.logger import logger

train_triplet_dict = None
all_triplet_dict = None
link_graph = None
entity_dict = None
relation_id_map = None
tokenizer: AutoTokenizer = None


def _resolve_preprocessed_dir() -> str:
    """Resolve the directory that contains preprocessed JSON artifacts when available."""

    candidate_dirs = [
        os.path.dirname(args.valid_path),
        os.path.dirname(args.test_path),
        os.path.dirname(args.train_path),
    ]
    for candidate_dir in candidate_dirs:
        if not candidate_dir:
            continue
        candidate_path = os.path.join(candidate_dir, 'train.txt.json')
        if os.path.exists(candidate_path):
            return candidate_dir
    for candidate_dir in candidate_dirs:
        if candidate_dir:
            return candidate_dir
    return os.getcwd()


def _init_entity_dict() -> None:
    """Initialize the entity dictionary if it hasn't been loaded yet."""

    global entity_dict
    if not entity_dict:
        from data.dataset import EntityDict
        entity_dict_dir = os.path.dirname(args.valid_path) or os.path.dirname(args.train_path) or os.getcwd()
        entity_dict = EntityDict(entity_dict_dir=entity_dict_dir)


def _init_relation_id_map():
    """Initialize the relation id map if it hasn't been loaded yet."""

    global relation_id_map
    if relation_id_map is not None:
        return

    candidate_dir = _resolve_preprocessed_dir()
    relation_path = os.path.join(candidate_dir, 'relation2id.json')
    if os.path.exists(relation_path):
        import json
        with open(relation_path, 'r', encoding='utf-8') as reader:
            relation_id_map = json.load(reader)
            return

    from data.dataset import TripletDict
    fallback_paths = []
    for source_path in [args.train_path, args.valid_path, args.test_path]:
        if source_path:
            fallback_paths.append(source_path)
    if fallback_paths:
        triplets = TripletDict(path_list=[fallback_paths[0]])
        relation_id_map = {relation: idx for idx, relation in enumerate(sorted(triplets.relations))}
        return

    relation_id_map = {}


def _init_train_triplet_dict() -> None:
    """Initialize the training triplet dictionary if it hasn't been loaded yet."""

    global train_triplet_dict
    if not train_triplet_dict:
        from data.dataset import TripletDict
        data_dir = _resolve_preprocessed_dir()
        train_path = os.path.join(data_dir, 'train.txt.json')
        if not os.path.exists(train_path):
            train_path = args.train_path
        train_triplet_dict = TripletDict(path_list=[train_path])


def _init_all_triplet_dict() -> None:
    """Initialize the all triplet dictionary if it hasn't been loaded yet."""

    global all_triplet_dict
    if not all_triplet_dict:
        from data.dataset import TripletDict
        path_pattern = '{}/*.txt.json'.format(_resolve_preprocessed_dir())
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))


def _init_link_graph() -> None:
    """Initialize the link graph if it hasn't been loaded yet."""

    global link_graph
    if not link_graph:
        from data.dataset import LinkGraph
        data_dir = _resolve_preprocessed_dir()
        train_path = os.path.join(data_dir, 'train.txt.json')
        if not os.path.exists(train_path):
            train_path = args.train_path
        link_graph = LinkGraph(train_path=train_path)


def get_entity_dict() -> 'EntityDict':
    """Get the entity dictionary, initializing it if necessary."""

    _init_entity_dict()
    return entity_dict


def get_relation_id_map() -> dict:
    """Get the relation-to-id mapping, initializing it if necessary."""

    _init_relation_id_map()
    return relation_id_map


def get_train_triplet_dict() -> 'TripletDict':
    """Get the training triplet dictionary, initializing it if necessary."""

    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict() -> 'TripletDict':
    """Get the all triplet dictionary, initializing it if necessary."""

    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph() -> 'LinkGraph':
    """Get the link graph, initializing it if necessary."""

    _init_link_graph()
    return link_graph


def build_tokenizer(args) -> None:
    """Build the tokenizer from the specified pretrained model, caching it for future use."""

    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_encoder)
        logger.info('Build tokenizer from {}'.format(args.bert_encoder))


def get_tokenizer() -> AutoTokenizer:
    """Get the tokenizer, initializing it if necessary."""

    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer
