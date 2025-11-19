import logging
from enum import IntEnum, auto

import jax

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput

logger = logging.getLogger(__name__)


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    STANDALONE = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE or self == SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_standalone(self):
        return self == SpeculativeAlgorithm.STANDALONE

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "STANDALONE": SpeculativeAlgorithm.STANDALONE,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]


def detect_nan(logits_output: LogitsProcessorOutput):
    logits = logits_output.next_token_logits
    if jax.numpy.any(jax.numpy.isnan(logits)):
        logger.error("Detected errors during sampling! NaN in the logits.")
        raise ValueError("Detected errors during sampling! NaN in the logits.")
