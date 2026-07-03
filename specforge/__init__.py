import importlib

_LAZY_MODULES = {
    "core": "specforge.core",
    "modeling": "specforge.modeling",
}

_LAZY_ATTRS = {
    "OnlineDFlashModel": "specforge.core",
    "OnlineDominoModel": "specforge.core",
    "OnlineEagle3Model": "specforge.core",
    "OnlinePEagleModel": "specforge.core",
    "QwenVLOnlineEagle3Model": "specforge.core",
    "AutoDraftModelConfig": "specforge.modeling",
    "AutoEagle3DraftModel": "specforge.modeling",
    "CustomEagle3TargetModel": "specforge.modeling",
    "HFEagle3TargetModel": "specforge.modeling",
    "LlamaForCausalLMEagle3": "specforge.modeling",
    "PEagleDraftModel": "specforge.modeling",
    "SGLangEagle3TargetModel": "specforge.modeling",
    "get_eagle3_target_model": "specforge.modeling",
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
