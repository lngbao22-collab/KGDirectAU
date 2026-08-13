"""Singleton-like hub for loading and caching KG data structures."""

import os
import glob

from configs.config import args

train_triplet_dict = None
all_triplet_dict = None
entity_dict = None
relation_id_map = None


def _split_parent_dirs() -> list[str]:
    """Parent directories of configured train/valid/test paths (deduplicated, order preserved)."""

    dirs: list[str] = []
    seen: set[str] = set()
    for source_path in [args.valid_path, args.test_path, args.train_path]:
        if not source_path:
            continue
        candidate_dir = os.path.dirname(source_path)
        if not candidate_dir or candidate_dir in seen:
            continue
        seen.add(candidate_dir)
        dirs.append(candidate_dir)
    return dirs


def _resolve_preprocessed_dir() -> str:
    """Resolve the directory that contains preprocessed JSON artifacts when available."""

    candidate_dirs = _split_parent_dirs()
    for candidate_dir in candidate_dirs:
        candidate_path = os.path.join(candidate_dir, 'train.txt.json')
        if os.path.exists(candidate_path):
            return candidate_dir
    for candidate_dir in candidate_dirs:
        return candidate_dir
    return os.getcwd()


def _resolve_entity_dict_dir() -> str:
    """Prefer a directory that already contains ``entities.json`` (preprocessed or dataset root)."""

    dataset = getattr(args, 'dataset', None) or ''
    candidate_dirs: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        if not path or path in seen:
            return
        seen.add(path)
        candidate_dirs.append(path)

    for parent in _split_parent_dirs():
        _add(parent)
        _add(os.path.join(parent, 'preprocessed'))
        # When splits resolve to raw ``data/<ds>/train.txt``, also check sibling preprocessed/.
        _add(os.path.join(os.path.dirname(parent), 'preprocessed'))
        _add(os.path.dirname(parent))

    if dataset:
        _add(os.path.join('data', dataset, 'preprocessed'))
        _add(os.path.join('data', dataset))

    for candidate_dir in candidate_dirs:
        if os.path.exists(os.path.join(candidate_dir, 'entities.json')):
            return candidate_dir

    # Fall back to the best split/preprocessed dir so EntityDict can synthesize from splits.
    return _resolve_preprocessed_dir()


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

    from utils.relations import load_relation_to_idx

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


def init_dataloader_worker(_worker_id: int = 0) -> None:
    """Pre-load read-only caches in DataLoader worker processes (spawn-safe)."""

    _init_entity_dict()
    _init_train_triplet_dict()


def warmup_data_structures() -> None:
    """Eagerly load shared data structures in the main training process."""

    _init_entity_dict()
    _init_train_triplet_dict()
