import json
import logging
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb import eplb_algorithms
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)

# Global variables set during server initialization
_GLOBAL_SERVER_ARGS = None
_GLOBAL_EXPERT_LOCATION_METADATA = None


def get_global_server_args():
    return _GLOBAL_SERVER_ARGS


def set_global_server_args(args):
    global _GLOBAL_SERVER_ARGS
    _GLOBAL_SERVER_ARGS = args


def get_global_expert_location_metadata():
    return _GLOBAL_EXPERT_LOCATION_METADATA


def set_global_expert_location_metadata(metadata):
    global _GLOBAL_EXPERT_LOCATION_METADATA
    _GLOBAL_EXPERT_LOCATION_METADATA = metadata


@register_pytree_node_class
class ExpertLocationMetadata:
    """
    Stores global expert mapping metadata.
    """

    def __init__(
        self,
        ep_dispatch_algorithm: Literal["static", "dynamic", "fake"] = "static",
        logical_to_rank_dispatch_physical_map: np.ndarray | None = None,
        logical_to_all_physical_map: np.ndarray | None = None,
        logical_to_all_physical_map_num_valid: np.ndarray | None = None,
        physical_to_logical_map: np.ndarray | None = None,
        num_physical_experts: int = 0,
    ):
        self.ep_dispatch_algorithm = ep_dispatch_algorithm
        self.logical_to_rank_dispatch_physical_map = device_array(
            logical_to_rank_dispatch_physical_map, sharding=(P(None))
        )
        self.logical_to_all_physical_map = device_array(
            logical_to_all_physical_map, sharding=(P(None))
        )
        self.logical_to_all_physical_map_num_valid = device_array(
            logical_to_all_physical_map_num_valid, sharding=(P(None))
        )
        self.physical_to_logical_map = device_array(physical_to_logical_map, sharding=(P(None)))
        self.num_physical_experts = num_physical_experts

    def tree_flatten(self):
        children = (
            self.logical_to_rank_dispatch_physical_map,
            self.logical_to_all_physical_map,
            self.logical_to_all_physical_map_num_valid,
            self.physical_to_logical_map,
        )
        aux_data = {
            "ep_dispatch_algorithm": self.ep_dispatch_algorithm,
            "num_physical_experts": self.num_physical_experts,
        }

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.logical_to_rank_dispatch_physical_map = children[0]
        obj.logical_to_all_physical_map = children[1]
        obj.logical_to_all_physical_map_num_valid = children[2]
        obj.physical_to_logical_map = children[3]
        obj.ep_dispatch_algorithm = aux_data["ep_dispatch_algorithm"]
        obj.num_physical_experts = aux_data["num_physical_experts"]
        return obj

    @staticmethod
    def init_trivial(server_args: ServerArgs, model_config: ModelConfig):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        num_physical_experts = common["num_physical_experts"]
        num_layers = common["num_layers"]
        num_logical_experts = common["num_logical_experts"]

        physical_to_logical_map = (
            np.tile(np.arange(0, num_physical_experts), (num_layers, 1)) % num_logical_experts
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        model_config: ModelConfig,
        physical_to_logical_map,
    ):
        if not isinstance(physical_to_logical_map, np.ndarray):
            physical_to_logical_map = np.array(physical_to_logical_map)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        num_logical_experts = common["num_logical_experts"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            server_args=server_args,
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=num_logical_experts,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def init_by_eplb(server_args: ServerArgs, model_config: ModelConfig, logical_count):
        if not isinstance(logical_count, np.ndarray):
            logical_count = np.array(logical_count)
        if len(logical_count.shape) == 1:
            logical_count = logical_count[None, :]

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        num_physical_experts = common["num_physical_experts"]
        num_groups = common["num_groups"]
        num_nodes = server_args.nnodes

        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=getattr(server_args, "eplb_algorithm", "auto"),
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        num_logical_experts = getattr(model_config.hf_config, "num_experts", 0)
        num_layers = getattr(model_config.hf_config, "num_hidden_layers", 0)
        num_groups = getattr(model_config.hf_config, "num_expert_group", 1)

        if num_logical_experts == 0 or num_layers == 0:
            return None

        ep_size = server_args.ep_size
        num_physical_experts = num_logical_experts + server_args.ep_num_redundant_experts

        return {
            "num_logical_experts": num_logical_experts,
            "num_layers": num_layers,
            "num_groups": num_groups,
            "ep_size": ep_size,
            "num_physical_experts": num_physical_experts,
        }

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        physical_to_logical_map: np.ndarray,
        logical_to_all_physical_map: np.ndarray,
    ):
        # from jax.experimental import multihost_utils

        # def sync_shape_and_data(data, name):
        #     is_root = jax.process_index() == 0
        #     ndim = np.array(data.ndim if is_root else 0, dtype=np.int32)
        #     actual_ndim = int(multihost_utils.broadcast_one_to_all(ndim, is_source=is_root))

        #     if is_root:
        #         local_shape = np.array(data.shape, dtype=np.int32)
        #     else:
        #         local_shape = np.zeros((actual_ndim,), dtype=np.int32)
        #     global_shape = multihost_utils.broadcast_one_to_all(local_shape, is_source=is_root)
        #     global_shape = tuple(map(int, global_shape))

        #     if not is_root:
        #         data = np.zeros(global_shape, dtype=data.dtype)

        #     return multihost_utils.broadcast_one_to_all(data, is_source=is_root)

        # p2l_synced = sync_shape_and_data(physical_to_logical_map, "p2l")
        # l2p_synced = sync_shape_and_data(logical_to_all_physical_map, "l2p")

        # physical_to_logical_map = np.array(p2l_synced)
        # logical_to_all_physical_map = np.array(l2p_synced)

        logical_to_all_physical_map_num_valid = np.sum(logical_to_all_physical_map != -1, axis=2)
        logical_to_rank_dispatch_physical_map = logical_to_all_physical_map[:, :, 0]

        return ExpertLocationMetadata(
            ep_dispatch_algorithm=server_args.ep_dispatch_algorithm or "static",
            logical_to_rank_dispatch_physical_map=logical_to_rank_dispatch_physical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            physical_to_logical_map=physical_to_logical_map,
            num_physical_experts=physical_to_logical_map.shape[1],
        )


def compute_initial_expert_location_metadata(
    server_args: ServerArgs,
    model_config: ModelConfig,
) -> ExpertLocationMetadata | None:
    data = server_args.init_expert_location
    logger.info("Computing initial expert location metadata from: %s", data)

    if data == "trivial":
        return ExpertLocationMetadata.init_trivial(server_args, model_config)

    if data.endswith(".npy"):
        try:
            loaded = np.load(data, allow_pickle=True)
            # If it's a dictionary saved as an object array
            if loaded.dtype == object and loaded.ndim == 0:
                data_dict = loaded.item()
            else:
                # If it's a regular array, assume it's the physical_to_logical_map
                data_dict = {"physical_to_logical_map": loaded}
            logger.info("Loaded .npy data keys: %s", list(data_dict.keys()))
        except Exception as e:
            logger.error("Failed to load .npy file %s: %s", data, e)
            return None
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
        logger.info("Loaded .json data keys: %s", list(data_dict.keys()))
    else:
        try:
            data_dict = json.loads(data)
            logger.info("Parsed JSON string data keys: %s", list(data_dict.keys()))
        except json.JSONDecodeError:
            logger.error("Failed to parse init_expert_location as JSON string.")
            return None

    if "physical_to_logical_map" in data_dict:
        mapping = data_dict["physical_to_logical_map"]
        logger.info(
            "init_expert_location via init_by_mapping. Mapping shape: %s",
            getattr(mapping, "shape", "unknown"),
        )
        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=mapping,
        )
    elif "logical_count" in data_dict:
        counts = data_dict["logical_count"]
        logger.info(
            "init_expert_location via init_by_eplb. Counts shape: %s",
            getattr(counts, "shape", "unknown"),
        )
        return ExpertLocationMetadata.init_by_eplb(server_args, model_config, logical_count=counts)
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format. Keys found: {list(data_dict.keys())}"
        )


def init_expert_location_metadata(server_args, model_config):
    """
    Initializes the global expert mapping.
    """
    metadata = compute_initial_expert_location_metadata(server_args, model_config)
    if metadata is not None:
        set_global_expert_location_metadata(metadata)


def _compute_logical_to_all_physical_map(
    server_args: ServerArgs,
    physical_to_logical_map: np.ndarray,
    num_logical_experts: int,
):
    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]

    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = int(physical_to_logical_map[layer_id, physical_expert_id])
            logical_to_all_physical_map[layer_id][logical_expert_id].append(physical_expert_id)

    logical_to_all_physical_map = _pad_nested_array(logical_to_all_physical_map, pad_value=-1)

    return np.array(logical_to_all_physical_map)


def _pad_nested_array(arr, pad_value):
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [[inner + [pad_value] * (max_len - len(inner)) for inner in outer] for outer in arr]
    return padded


def topk_ids_logical_to_physical(
    topk_ids: jax.Array,
    info: ExpertLocationMetadata | None,
    layer_id: int = 0,
) -> jax.Array:
    """
    Maps logical expert IDs to physical expert IDs.
    Because 'info' contains Static arrays, JAX will use the concrete values during trace.
    """
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info, layer_id)
    if info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info, layer_id)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")


def _topk_ids_logical_to_physical_static(
    topk_ids: jax.Array,
    info: ExpertLocationMetadata,
    layer_id: int = 0,
) -> jax.Array:
    return info.logical_to_rank_dispatch_physical_map[layer_id, topk_ids]


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: jax.Array,
    info: ExpertLocationMetadata,
    layer_id: int = 0,
) -> jax.Array:
    num_valid = info.logical_to_all_physical_map_num_valid.at[layer_id, topk_ids].get(
        out_sharding=P("data", None)
    )
    selected_expert_replicas = info.logical_to_all_physical_map.at[layer_id, topk_ids].get(
        out_sharding=P("data", None)
    )

    rng_key = jax.random.key(0)
    random_offsets = jax.random.randint(
        rng_key, shape=topk_ids.shape, minval=0, maxval=65536, dtype=jnp.int32
    )
    chosen_dispatch_index = random_offsets % num_valid

    physical_ids = jnp.take_along_axis(
        selected_expert_replicas, chosen_dispatch_index[..., None], axis=-1
    ).squeeze(-1)

    return physical_ids
