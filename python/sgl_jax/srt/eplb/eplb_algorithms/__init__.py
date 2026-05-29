from enum import Enum, auto

import numpy as np

from sgl_jax.srt.eplb.eplb_algorithms import deepseek


class EplbAlgorithm(Enum):
    deepseek = auto()
    deepseek_hierarchical = auto()


def rebalance_experts(
    tokens_per_expert: np.ndarray,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: int | None,
    num_nodes: int,
    algorithm: EplbAlgorithm,
):
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        return deepseek.rebalance_experts(
            weight=tokens_per_expert,
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
        )

    raise NotImplementedError


def compute_algorithm(
    raw_algorithm: str,
    num_groups: int | None,
    num_nodes: int,
) -> EplbAlgorithm:
    if raw_algorithm != "auto":
        return EplbAlgorithm[raw_algorithm]

    # TODO test on real scenarios and know which ones perform better
    if (num_groups is not None) and (num_groups % num_nodes == 0):
        return EplbAlgorithm.deepseek_hierarchical
    else:
        return EplbAlgorithm.deepseek
