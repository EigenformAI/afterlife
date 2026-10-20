from importlib import import_module

__all__ = [
    "resolve_training_export_format",
    "train_sft",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    qlora = import_module(".qlora", __name__)
    return getattr(qlora, name)
