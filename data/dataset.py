"""Triplet and entity dictionaries for ComplEx / ComplEx-AU."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

from utils.logger import logger

from data.dict_hub import get_entity_dict


def _get_entity_dict() -> EntityDict:
	"""Get the entity dictionary, which provides mapping from entity IDs to their descriptions."""

	return get_entity_dict()


def reverse_triplet(obj) -> dict:
	"""Given a triplet object, return a new triplet object with head and tail reversed, and relation modified to indicate inversion."""

	return {
		'head_id': obj['tail_id'],
		'head': obj['tail'],
		'relation': 'inverse {}'.format(obj['relation']),
		'tail_id': obj['head_id'],
		'tail': obj['head'],
	}


@dataclass
class EntityExample:
	"""Data class representing an entity example, including its ID, name, and description."""

	entity_id: str
	entity: str
	entity_desc: str = ''


class TripletDict:
	"""Data structure for storing triplets and providing neighbor information for entities."""

	def __init__(self, path_list: List[str]):
		self.path_list = path_list
		self.relations = set()
		self.hr2tails = {}
		self.rt2heads = {}
		self.triplet_cnt = 0

		for path in self.path_list:
			self._load(path)

	def _load(self, path: str) -> None:
		"""Load triplets from a given path and populate the internal data structures for neighbor retrieval."""

		examples = []
		if path.endswith('.json'):
			examples = json.load(open(path, 'r', encoding='utf-8'))
		elif path.endswith('.txt'):
			with open(path, 'r', encoding='utf-8') as reader:
				for line in reader:
					fields = line.strip().split('\t')
					if len(fields) not in (3, 4):
						continue
					head_id, relation, tail_id = fields[:3]
					examples.append({'head_id': head_id, 'relation': relation, 'tail_id': tail_id})
		else:
			raise ValueError(f'Unsupported format: {path}')

		for ex in examples:
			rt_key = (ex['relation'], ex['tail_id'])
			if rt_key not in self.rt2heads:
				self.rt2heads[rt_key] = set()
			self.rt2heads[rt_key].add(ex['head_id'])

		reversed_examples = [
			{
				'head_id': ex['tail_id'],
				'relation': 'inverse {}'.format(ex['relation']),
				'tail_id': ex['head_id'],
			}
			for ex in examples
		]
		examples += reversed_examples
		for ex in examples:
			self.relations.add(ex['relation'])
			key = (ex['head_id'], ex['relation'])
			if key not in self.hr2tails:
				self.hr2tails[key] = set()
			self.hr2tails[key].add(ex['tail_id'])
		self.triplet_cnt += len(examples)

	def get_neighbors(self, h: str, r: str) -> set:
		"""Given a head entity ID and a relation, return the set of tail entity IDs that are connected to the head via the relation."""

		return self.hr2tails.get((h, r), set())

	def get_heads(self, r: str, t: str) -> set:
		"""Given a relation and tail entity ID, return known head entities for filtered head prediction."""

		return self.rt2heads.get((r, t), set())


class EntityDict:
	"""Data structure for storing entity information and providing mapping from entity IDs to their descriptions."""

	def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
		path = os.path.join(entity_dict_dir, 'entities.json')
		from configs.config import args as current_args

		if os.path.exists(path):
			self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]
			source = path
		else:
			# Index KGE can proceed from split entity IDs when preprocess was not run yet.
			self.entity_exs = []
			source = f'splits beside {entity_dict_dir} (missing entities.json)'
			dataset_name = getattr(current_args, 'dataset', None) or '<dataset>'
			logger.warning(
				'entities.json not found under %s; building entity vocabulary from train/valid/test. '
				'Run `python data/preprocess.py --dataset %s` to generate the full preprocessed layout.',
				entity_dict_dir,
				dataset_name,
			)
		self._ensure_entity_coverage(entity_dict_dir)

		if inductive_test_path:
			examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
			valid_entity_ids = set()
			for ex in examples:
				valid_entity_ids.add(ex['head_id'])
				valid_entity_ids.add(ex['tail_id'])
			self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

		if not self.entity_exs:
			raise FileNotFoundError(
				f'No entities found (looked for {path}). '
				'Generate preprocessed data, e.g. '
				'`python data/preprocess.py --dataset hetionet_subset`, '
				'or ensure train/valid/test paths exist.'
			)

		self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
		self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
		logger.info('Load {} entities from {}'.format(len(self.id2entity), source))

	def _ensure_entity_coverage(self, entity_dict_dir: str) -> None:
		"""Backfill entities that appear in raw split files but are missing from entities.json."""

		from configs.config import args as current_args

		known_entity_ids = {ex.entity_id for ex in self.entity_exs}
		missing_entity_ids = set()
		for split_path in [getattr(current_args, 'train_path', ''), getattr(current_args, 'valid_path', ''), getattr(current_args, 'test_path', '')]:
			if not split_path or not os.path.exists(split_path):
				continue
			if split_path.endswith('.json'):
				with open(split_path, 'r', encoding='utf-8') as reader:
					for obj in json.load(reader):
						missing_entity_ids.add(obj['head_id'])
						missing_entity_ids.add(obj['tail_id'])
			else:
				with open(split_path, 'r', encoding='utf-8') as reader:
					for line in reader:
						fields = line.strip().split('\t')
						if len(fields) not in (3, 4):
							continue
						missing_entity_ids.add(fields[0])
						missing_entity_ids.add(fields[2])

		missing_entity_ids.difference_update(known_entity_ids)
		if not missing_entity_ids:
			return

		definition_candidates = [
			os.path.join(entity_dict_dir, '..', 'wordnet-mlj12-definitions.txt'),
			os.path.join(entity_dict_dir, 'wordnet-mlj12-definitions.txt'),
		]
		entity_text_map = {}
		for candidate in definition_candidates:
			if not os.path.exists(candidate):
				continue
			with open(candidate, 'r', encoding='utf-8') as reader:
				for line in reader:
					fields = line.strip().split('\t')
					if len(fields) != 3:
						continue
					entity_id, word, _ = fields
					entity_text_map[entity_id] = word.replace('__', ' ')
				break

		for entity_id in sorted(missing_entity_ids):
			self.entity_exs.append(EntityExample(
				entity_id=entity_id,
				entity=entity_text_map.get(entity_id, entity_id),
				entity_desc='',
			))

	def entity_to_idx(self, entity_id: str) -> int:
		"""Given an entity ID, return its corresponding index in the entity list."""

		return self.entity2idx[entity_id]

	def get_entity_by_id(self, entity_id: str) -> EntityExample:
		"""Given an entity ID, return the corresponding EntityExample object containing its description."""

		return self.id2entity[entity_id]

	def get_entity_by_idx(self, idx: int) -> EntityExample:
		"""Given an index, return the corresponding EntityExample object."""

		return self.entity_exs[idx]

	def __len__(self) -> int:
		"""Return the total number of entities in the dictionary."""

		return len(self.entity_exs)


class Example:
	"""Triplet example with entity-name lookups for ComplEx / ComplEx-AU."""

	def __init__(self, head_id, relation, tail_id, label=None, **kwargs):
		self.head_id = head_id
		self.tail_id = tail_id
		self.relation = relation
		self.label = int(label) if label is not None else None

	@property
	def head_desc(self) -> str:
		"""Return the description of the head entity, or an empty string if the head ID is not provided."""

		if not self.head_id:
			return ''
		return _get_entity_dict().get_entity_by_id(self.head_id).entity_desc

	@property
	def tail_desc(self) -> str:
		"""Return the description of the tail entity, or an empty string if the tail ID is not provided."""

		if not self.tail_id:
			return ''
		return _get_entity_dict().get_entity_by_id(self.tail_id).entity_desc

	@property
	def head(self) -> str:
		"""Return the name of the head entity, or an empty string if the head ID is not provided."""

		if not self.head_id:
			return ''
		return _get_entity_dict().get_entity_by_id(self.head_id).entity

	@property
	def tail(self) -> str:
		"""Return the name of the tail entity, or an empty string if the tail ID is not provided."""

		if not self.tail_id:
			return ''
		return _get_entity_dict().get_entity_by_id(self.tail_id).entity


def load_data(path: str, add_forward_triplet: bool = True, add_backward_triplet: bool = True) -> List[Example]:
	"""Load examples from a given path, which can be in JSON or TXT format, and return a list of Example objects. The function also supports adding forward and backward triplets based on the specified flags."""

	examples = []
	if path.endswith('.json'):
		data = json.load(open(path, 'r', encoding='utf-8'))
		logger.info('Load {} examples from {}'.format(len(data), path))
		for i, obj in enumerate(data):
			if add_forward_triplet:
				examples.append(Example(**obj))
			if add_backward_triplet:
				examples.append(Example(**reverse_triplet(obj)))
			data[i] = None
	elif path.endswith('.txt'):
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				fs = line.strip().split('\t')
				if len(fs) == 4:
					head_id, relation, tail_id, label = fs
					if str(label) == '1':
						if add_forward_triplet:
							examples.append(Example(head_id=head_id, relation=relation, tail_id=tail_id, label=label))
						if add_backward_triplet:
							examples.append(Example(**reverse_triplet({'head_id': head_id, 'head': '', 'relation': relation, 'tail_id': tail_id, 'tail': ''})))
					elif not (add_forward_triplet or add_backward_triplet):
						examples.append(Example(head_id=head_id, relation=relation, tail_id=tail_id, label=label))
				elif len(fs) == 3:
					head_id, relation, tail_id = fs
					if add_forward_triplet:
						examples.append(Example(head_id=head_id, relation=relation, tail_id=tail_id, label='1'))
					if add_backward_triplet:
						examples.append(Example(**reverse_triplet({'head_id': head_id, 'head': '', 'relation': relation, 'tail_id': tail_id, 'tail': ''})))
	else:
		raise ValueError(f'Unsupported format: {path}')
	return examples
