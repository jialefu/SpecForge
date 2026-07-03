import importlib

_LAZY_MODULES = {
    "dataset_processors": "specforge.data.dataset_processors",
    "dataset_recipes": "specforge.data.dataset_recipes",
    "parse": "specforge.data.parse",
    "preprocessing": "specforge.data.preprocessing",
    "template": "specforge.data.template",
    "utils": "specforge.data.utils",
}

_LAZY_ATTRS = {
    "ChatTemplate": "specforge.data.template",
    "DatasetRecipe": "specforge.data.dataset_recipes",
    "DatasetRecipeRegistry": "specforge.data.dataset_recipes",
    "DATASET_RECIPE_REGISTRY": "specforge.data.dataset_recipes",
    "build_eagle3_dataset": "specforge.data.preprocessing",
    "build_offline_eagle3_dataset": "specforge.data.preprocessing",
    "generate_vocab_mapping_file": "specforge.data.preprocessing",
    "preprocess_conversations": "specforge.data.preprocessing",
    "prepare_dp_dataloaders": "specforge.data.utils",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted([*_LAZY_MODULES, *_LAZY_ATTRS])
