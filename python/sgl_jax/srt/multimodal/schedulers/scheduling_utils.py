import importlib
import os
from dataclasses import dataclass

import jax

from sgl_jax.srt.multimodal.configuration_utils import ConfigMixin

SCHEDULER_CONFIG_NAME = "scheduler_config.json"


@dataclass
class SchedulerOutput:
    prev_sample: jax.Array


class SchedulerMixin(ConfigMixin):
    """
    Minimal scheduler mixin built on top of the local `ConfigMixin`.

    Currently supported:
    - scheduler default config filename via `scheduler_config.json`
    - `save_pretrained` delegating to local `save_config`
    - `from_pretrained` loading from a local directory or config path
    - listing locally compatible scheduler classes via `compatibles`

    Not supported yet:
    - Hugging Face Hub loading/saving
    - hub auth, cache, revision, proxy, or offline-mode arguments
    - diffusers-wide compatible scheduler discovery across packages
    - `PushToHubMixin` behavior from the full HF implementation
    """

    config_name = SCHEDULER_CONFIG_NAME
    _compatibles: list[str] = []
    has_compatibles = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
        subfolder: str | None = None,
        return_unused_kwargs: bool = False,
        **kwargs,
    ):
        if pretrained_model_name_or_path is None:
            raise ValueError("`pretrained_model_name_or_path` must be provided.")

        config_path = pretrained_model_name_or_path
        if subfolder is not None:
            config_path = os.path.join(str(pretrained_model_name_or_path), subfolder)

        config = cls.load_config(config_path)
        model = cls.from_config(config, **kwargs)
        if return_unused_kwargs:
            return model, {}
        return model

    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs):
        del kwargs
        self.save_config(save_directory=save_directory)

    @property
    def compatibles(self):
        return self._get_compatibles()

    @classmethod
    def _get_compatibles(cls):
        module = importlib.import_module(cls.__module__)
        compatible_classes = []
        for class_name in list(dict.fromkeys([cls.__name__, *cls._compatibles])):
            if hasattr(module, class_name):
                compatible_classes.append(getattr(module, class_name))
        return compatible_classes
