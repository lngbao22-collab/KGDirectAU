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
tokenizer: AutoTokenizer = None


def _init_entity_dict():
    """Initialize the entity dictionary if it hasn't been loaded yet."""

    global entity_dict
    if not entity_dict:
        from data.dataset import EntityDict
        entity_dict_dir = os.path.dirname(args.valid_path) or os.path.dirname(args.train_path) or os.getcwd()
        entity_dict = EntityDict(entity_dict_dir=entity_dict_dir)


def _init_train_triplet_dict():
    """Initialize the training triplet dictionary if it hasn't been loaded yet."""

    global train_triplet_dict
    if not train_triplet_dict:
        from data.dataset import TripletDict
        train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict():
    """Initialize the all triplet dictionary if it hasn't been loaded yet."""

    global all_triplet_dict
    if not all_triplet_dict:
        from data.dataset import TripletDict
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))


def _init_link_graph():
    """Initialize the link graph if it hasn't been loaded yet."""

    global link_graph
    if not link_graph:
        from data.dataset import LinkGraph
        link_graph = LinkGraph(train_path=args.train_path)


def get_entity_dict():
    """Get the entity dictionary, initializing it if necessary."""

    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    """Get the training triplet dictionary, initializing it if necessary."""

    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    """Get the all triplet dictionary, initializing it if necessary."""

    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph():
    """Get the link graph, initializing it if necessary."""

    _init_link_graph()
    return link_graph


def build_tokenizer(args):
    """Build the tokenizer from the specified pretrained model, caching it for future use."""

    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder)
        logger.info('Build tokenizer from {}'.format(args.encoder))


def get_tokenizer():    
    """Get the tokenizer, initializing it if necessary."""

    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer
