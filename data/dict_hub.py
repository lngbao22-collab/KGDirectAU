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


def _resolve_entity_dict_dir() -> str:
	"""Resolve the directory containing ``entities.json`` for the configured dataset."""

	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	candidate_dirs: list[str] = []

	def _add_dir(path: str | None) -> None:
		if not path:
			return
		abs_path = path
		if not os.path.isabs(abs_path):
			for base in (os.getcwd(), repo_root):
				candidate = os.path.join(base, path)
				if os.path.exists(candidate):
					abs_path = os.path.abspath(candidate)
					break
			else:
				abs_path = os.path.abspath(os.path.join(os.getcwd(), path))
		if os.path.isfile(abs_path):
			candidate_dirs.append(os.path.dirname(abs_path))
			parent = os.path.dirname(os.path.dirname(abs_path))
			if parent:
				candidate_dirs.append(parent)
		elif os.path.isdir(abs_path):
			candidate_dirs.append(abs_path)

	for split_path in (
		getattr(args, 'train_path', None),
		getattr(args, 'valid_path', None),
		getattr(args, 'test_path', None),
	):
		_add_dir(split_path)

	dataset = str(getattr(args, 'dataset', '') or '').strip()
	if dataset:
		for layout in (
			os.path.join('data', dataset, 'preprocessed'),
			os.path.join('data', dataset.upper(), 'preprocessed'),
			os.path.join('data', dataset),
			os.path.join('data', dataset.upper()),
		):
			for base in (os.getcwd(), repo_root):
				candidate_dirs.append(os.path.join(base, layout))

	seen: set[str] = set()
	unique_dirs: list[str] = []
	for candidate_dir in candidate_dirs:
		norm = os.path.normpath(candidate_dir)
		if not norm or norm in seen:
			continue
		seen.add(norm)
		unique_dirs.append(norm)

	for candidate_dir in unique_dirs:
		if os.path.exists(os.path.join(candidate_dir, 'entities.json')):
			return candidate_dir

	for candidate_dir in unique_dirs:
		for name in ('train.txt.json', 'valid.txt.json', 'test.txt.json', 'train.txt', 'valid.txt'):
			if os.path.exists(os.path.join(candidate_dir, name)):
				return candidate_dir

	return unique_dirs[0] if unique_dirs else os.getcwd()


def _init_entity_dict() -> None:
	"""Initialize the entity dictionary if it hasn't been loaded yet."""

	global entity_dict
	if not entity_dict:
		from data.dataset import EntityDict
		entity_dict = EntityDict(entity_dict_dir=_resolve_entity_dict_dir())


def _init_relation_id_map():
    """Initialize the relation id map if it hasn't been loaded yet."""

    global relation_id_map
    if relation_id_map is not None:
        return

    from base.embeddings import load_relation_to_idx

    relation_id_map = load_relation_to_idx(args)


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


def init_dataloader_worker(_worker_id: int = 0) -> None:
    """Pre-load read-only caches in DataLoader worker processes (spawn-safe)."""

    _init_entity_dict()
    _init_train_triplet_dict()
    if getattr(args, 'use_link_graph', False):
        _init_link_graph()


def warmup_data_structures() -> None:
    """Eagerly load shared data structures in the main training process."""

    _init_entity_dict()
    _init_train_triplet_dict()
    if getattr(args, 'use_link_graph', False):
        _init_link_graph()


def build_tokenizer(args) -> None:
    """Build the tokenizer from the specified pretrained model, caching it for future use."""

    global tokenizer
    if tokenizer is None:
        encoder = str(getattr(args, 'bert_encoder', '') or '').strip()
        if not encoder:
            raise RuntimeError(
                'bert_encoder is not configured; text tokenization is only required for SimKGC-style models.'
            )
        tokenizer = AutoTokenizer.from_pretrained(encoder)
        logger.info('Build tokenizer from %s', encoder)


def get_tokenizer() -> AutoTokenizer:
    """Get the tokenizer, initializing it if necessary."""

    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer
