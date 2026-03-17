from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, is_dataclass
from typing import Any


class BaseOutput(OrderedDict):
    """
    Minimal multimodal output base class modeled after diffusers' `BaseOutput`.

    Currently supported:
    - dataclass subclasses populate dict keys from non-None fields in `__post_init__`
    - string indexing like a dict, e.g. `output["prev_sample"]`
    - integer/slice indexing through `to_tuple()`
    - synchronized attribute and dict-style updates for existing keys
    - conversion to tuple via `to_tuple()`

    Not supported yet:
    - framework-specific pytree registration
    - any torch-specific distributed helpers
    """

    def __post_init__(self) -> None:
        class_fields = fields(self)
        if not class_fields:
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                value = getattr(self, field.name)
                if value is not None:
                    self[field.name] = value

    def __delitem__(self, *args, **kwargs):
        raise TypeError(f"You cannot use `__delitem__` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise TypeError(f"You cannot use `setdefault` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise TypeError(f"You cannot use `pop` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise TypeError(f"You cannot use `update` on a {self.__class__.__name__} instance.")

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return dict(self.items())[key]
        return self.to_tuple()[key]

    def __setattr__(self, name: Any, value: Any) -> None:
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable_obj, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable_obj, args, *remaining

    def to_tuple(self) -> tuple[Any, ...]:
        return tuple(self[k] for k in self.keys())