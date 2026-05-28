"""Dynamic import helpers to load modules/attributes from config-specified paths.

This module was previously named `loader.py` and provides small helpers used
by `main.py` and strategy modules to import modules and attributes by either
dot-path or filesystem path.
"""

from importlib import import_module
import importlib.util
import os
import sys
from types import ModuleType
from typing import Any


def _normalize_path(path: str) -> str:
	"""Normalize a module path by converting file paths to dotted module paths and stripping leading './'."""

	# Accept module paths or file paths
	if path.endswith('.py'):
		# convert file path to module-like dotted path by removing .py and replacing separators
		path = path[:-3]
		path = path.replace('/', '.').replace('\\', '.')
	# strip leading ./ or .\
	if path.startswith('./') or path.startswith('.\\'):
		path = path[2:]
	return path


def import_module_from_path(path: str) -> ModuleType:
	"""Import a module given either a dotted module path or a filesystem path ending with .py."""
	
	if not path:
		raise ValueError('Empty module path')
	if os.path.exists(path) and path.endswith('.py'):
		# load by file path
		spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
		module = importlib.util.module_from_spec(spec)
		sys.modules[spec.name] = module
		spec.loader.exec_module(module)
		return module
	# else treat as dotted module
	mod_path = _normalize_path(path)
	return import_module(mod_path)


def load_attr_from_path(path: str, attr: str) -> Any:
	"""Load an attribute from a module specified by path.

	Path can be a dotted module string or a .py file path. Attr is the symbol name inside.
	"""
	module = import_module_from_path(path)
	if not hasattr(module, attr):
		raise AttributeError(f"Module {path} has no attribute {attr}")
	return getattr(module, attr)
