from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

MARKER = "SGL_BENCH"


@dataclasses.dataclass
class MoEBenchmarkCase:
    name: str
    num_tokens: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    seed: int = 0
    activation: str = "silu"
    renormalize_topk_logits: bool = True
    num_expert_group: int = 0
    topk_group: int = 0
    routed_scaling_factor: float | None = None
    # If None, auto-pick based on available devices.
    ep_size: int | None = None
    tp_size: int | None = None


# Bailing MoE defaults (matches the observed precompile shapes).
BAILING_BASE = dict(
    num_experts=256,
    top_k=8,
    hidden_size=8192,
    intermediate_size=2048,
    activation="silu",
    renormalize_topk_logits=True,
    num_expert_group=8,
    topk_group=4,
    ep_size=None,
)

BAILING_MINI = dict(
    num_experts=256,
    top_k=8,
    hidden_size=2048,
    intermediate_size=512,
    activation="silu",
    renormalize_topk_logits=True,
    num_expert_group=8,
    topk_group=4,
    ep_size=None,
)

# Bailing MoE defaults (matches the observed precompile shapes).
BAILING_MINI_BASE = dict(
    num_experts=256,
    top_k=8,
    hidden_size=2048,
    intermediate_size=512,
    activation="silu",
    renormalize_topk_logits=True,
    num_expert_group=8,
    topk_group=4,
    # Let benchmarks pick ep_size based on available devices by default.
    ep_size=None,
)

_NUM_TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

BASE_CASES: Iterable[MoEBenchmarkCase] = tuple(
    MoEBenchmarkCase(
        name=f"bailing_nt{n}_ne256_tk8_h8192_i2048",
        num_tokens=n,
        **BAILING_BASE,
    )
    for n in _NUM_TOKENS
)

MINI_CASES: Iterable[MoEBenchmarkCase] = tuple(
    MoEBenchmarkCase(
        name=f"bailing_mini_nt{n}_ne256_tk8_h2048_i512",
        num_tokens=n,
        **BAILING_MINI,
    )
    for n in _NUM_TOKENS
)

GROUP_GEMM_CASES: Iterable[MoEBenchmarkCase] = BASE_CASES + MINI_CASES


def generate_router_logits(
    num_tokens: int,
    num_experts: int,
    scenario: str,
    num_experts_per_tok: int = 2,
    imbalance_factor: float = 3.0,
) -> jax.Array:
    """Synthetic router logits with configurable balance; keep generation cheap."""
    if scenario == "random":
        base = jnp.reshape(
            jnp.arange(num_tokens * num_experts, dtype=jnp.float32),
            (num_tokens, num_experts),
        )
        return base * 0.001

    if scenario == "balanced":
        logits = -10.0 * jnp.ones((num_tokens, num_experts), dtype=jnp.float32)
        token_ids = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
        cols = (
            token_ids * num_experts_per_tok + jnp.arange(num_experts_per_tok, dtype=jnp.int32)
        ) % num_experts
        logits = logits.at[jnp.arange(num_tokens)[:, None], cols].set(10.0)
        return logits

    if scenario == "imbalanced":
        temperature = num_experts / (imbalance_factor * 2)
        expert_base_logits = jnp.arange(num_experts, dtype=jnp.float32)
        expert_base_logits = 10.0 * jnp.exp(-expert_base_logits / temperature)
        logits = jnp.tile(expert_base_logits, (num_tokens, 1))
        return logits

    raise ValueError(f"Unknown scenario '{scenario}'. Use random|balanced|imbalanced.")


def build_group_sizes(
    router_logits: jax.Array, top_k: int, num_experts: int
) -> Tuple[jax.Array, jax.Array]:
    token_ids = np.arange(router_logits.shape[0], dtype=np.int32)
    topk_ids_np = np.empty((router_logits.shape[0], top_k), dtype=np.int32)
    for i in range(top_k):
        topk_ids_np[:, i] = (token_ids * top_k + i) % num_experts
    group_sizes = np.bincount(topk_ids_np.reshape(-1), minlength=num_experts).astype(np.int32)
    return jnp.asarray(group_sizes), jnp.asarray(topk_ids_np, dtype=jnp.int32)


def build_grouped_lhs(
    group_sizes: jax.Array, hidden_size: int, dtype: jnp.dtype, seed: int
) -> jax.Array:
    total = int(np.asarray(group_sizes, dtype=np.int32).sum())
    return jnp.empty((total, hidden_size), dtype=dtype)


def prepare_gmm_inputs(
    case: MoEBenchmarkCase,
    scenario: str,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Dict[str, jax.Array]:
    router_logits = generate_router_logits(
        case.num_tokens,
        case.num_experts,
        scenario,
        num_experts_per_tok=case.top_k,
        imbalance_factor=case.routed_scaling_factor or 3.0,
    ).astype(dtype)
    group_sizes, topk_ids = build_group_sizes(router_logits, case.top_k, case.num_experts)
    lhs = build_grouped_lhs(group_sizes, case.hidden_size, dtype, seed=case.seed + 1)
    rhs = jnp.empty((case.num_experts, case.hidden_size, case.intermediate_size), dtype=dtype)
    return {
        "router_logits": router_logits,
        "group_sizes": group_sizes,
        "topk_ids": topk_ids,
        "gmm_lhs": lhs,
        "gmm_rhs": rhs,
    }


def prepare_fused_moe_inputs(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype = jnp.bfloat16,
    mesh: jax.sharding.Mesh | None = None,
    *,
    ep_axis_name: str = "tensor",
    include_weights: bool = True,
    include_shared_expert: bool = False,
    se_intermediate_size: int | None = None,
) -> Dict[str, jax.Array]:
    if mesh is None:
        tokens = jnp.empty((case.num_tokens, case.hidden_size), dtype=dtype)
        out: dict[str, jax.Array] = {"tokens": tokens}
        if include_weights:
            out["w1"] = jnp.empty(
                (case.num_experts, case.hidden_size, case.intermediate_size), dtype=dtype
            )
            out["w3"] = jnp.empty(
                (case.num_experts, case.hidden_size, case.intermediate_size), dtype=dtype
            )
            out["w2"] = jnp.empty(
                (case.num_experts, case.intermediate_size, case.hidden_size),
                dtype=dtype,
            )
            if include_shared_expert:
                if se_intermediate_size is None:
                    se_intermediate_size = case.intermediate_size
                out["w1_shared"] = jnp.empty((case.hidden_size, se_intermediate_size), dtype=dtype)
                out["w3_shared"] = jnp.empty((case.hidden_size, se_intermediate_size), dtype=dtype)
                out["w2_shared"] = jnp.empty((se_intermediate_size, case.hidden_size), dtype=dtype)

        # For fused_moe benchmarks, routing can be made deterministic inside the
        # kernel via `balanced_topk`, so router logits are unnecessary here.
        router_logits = jnp.zeros((case.num_tokens, case.num_experts), dtype=dtype)
        out["router_logits"] = router_logits
        return out

    ep_size = mesh.shape[ep_axis_name]
    if case.num_tokens % ep_size != 0:
        raise ValueError(
            f"Expected {case.num_tokens=} to be divisible by {ep_size=} for {ep_axis_name=}."
        )
    if case.num_experts % ep_size != 0:
        raise ValueError(
            f"Expected {case.num_experts=} to be divisible by {ep_size=} for {ep_axis_name=}."
        )

    tokens_sharding = NamedSharding(mesh, P(ep_axis_name, None))
    logits_sharding = NamedSharding(mesh, P(ep_axis_name, None))
    w1_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))
    w2_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))
    w3_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))

    se_w1_sharding = NamedSharding(mesh, P())
    se_w2_sharding = NamedSharding(mesh, P())
    se_w3_sharding = NamedSharding(mesh, P())

    # Avoid `jax.device_put(host_array, NamedSharding(...))` for large weights:
    # on multi-host runs it may trigger a cross-host equality check (allgather)
    # of the entire unsharded array and OOM device memory.
    tokens = jax.jit(
        lambda: jnp.zeros((case.num_tokens, case.hidden_size), dtype=dtype),
        out_shardings=tokens_sharding,
    )()
    out: dict[str, jax.Array] = {"tokens": tokens}
    if include_weights:
        out["w1"] = jax.jit(
            lambda: jnp.zeros(
                (case.num_experts, case.hidden_size, case.intermediate_size),
                dtype=dtype,
            ),
            out_shardings=w1_sharding,
        )()
        out["w3"] = jax.jit(
            lambda: jnp.zeros(
                (case.num_experts, case.hidden_size, case.intermediate_size),
                dtype=dtype,
            ),
            out_shardings=w3_sharding,
        )()
        out["w2"] = jax.jit(
            lambda: jnp.zeros(
                (case.num_experts, case.intermediate_size, case.hidden_size),
                dtype=dtype,
            ),
            out_shardings=w2_sharding,
        )()

        if include_shared_expert:
            if se_intermediate_size is None:
                se_intermediate_size = case.intermediate_size
            out["w1_shared"] = jax.jit(
                lambda: jnp.zeros(
                    (case.hidden_size, se_intermediate_size),
                    dtype=dtype,
                ),
                out_shardings=se_w1_sharding,
            )()
            out["w3_shared"] = jax.jit(
                lambda: jnp.zeros(
                    (case.hidden_size, se_intermediate_size),
                    dtype=dtype,
                ),
                out_shardings=se_w3_sharding,
            )()
            out["w2_shared"] = jax.jit(
                lambda: jnp.zeros(
                    (se_intermediate_size, case.hidden_size),
                    dtype=dtype,
                ),
                out_shardings=se_w2_sharding,
            )()

    # For fused_moe benchmarks, routing can be made deterministic inside the
    # kernel via `balanced_topk`, so router logits are unnecessary here.
    router_logits = jax.jit(
        lambda: jnp.zeros((case.num_tokens, case.num_experts), dtype=dtype),
        out_shardings=logits_sharding,
    )()
    out["router_logits"] = router_logits
    return out


def format_load_info(group_sizes: jax.Array) -> str:
    sizes = jnp.asarray(group_sizes)
    total = int(sizes.sum())
    avg = float(jnp.mean(sizes))
    return f"dispatch={total}, avg_per_expert={avg:.1f}, " f"min={sizes.min()}, max={sizes.max()}"


def select_cases(cases: Iterable[MoEBenchmarkCase] | None = None) -> Iterable[MoEBenchmarkCase]:
    num_devices = len(jax.devices())
    raw_cases: Iterable[MoEBenchmarkCase] = GROUP_GEMM_CASES if cases is None else cases

    def choose_parallelism(case: MoEBenchmarkCase) -> tuple[int, int]:
        """Pick (ep_size, tp_size) for benchmarks.

        If `case.ep_size` is None, try EP sizes starting from device_count.
        Always return (ep_size, tp_size) such that ep_size * tp_size == device_count.
        """
        if case.ep_size is None:
            target_ep = num_devices
        else:
            target_ep = case.ep_size
        target_ep = min(target_ep, case.num_experts, num_devices)

        for ep in range(target_ep, 0, -1):
            if num_devices % ep != 0:
                continue
            if case.num_tokens % ep != 0:
                continue
            if case.num_experts % ep != 0:
                continue
            return ep, num_devices // ep
        return 1, num_devices

    cases = []
    for case in raw_cases:
        ep_size, tp_size = choose_parallelism(case)
        cases.append(
            MoEBenchmarkCase(
                name=case.name,
                num_tokens=case.num_tokens,
                num_experts=case.num_experts,
                top_k=case.top_k,
                hidden_size=case.hidden_size,
                intermediate_size=case.intermediate_size,
                activation=case.activation,
                renormalize_topk_logits=case.renormalize_topk_logits,
                num_expert_group=case.num_expert_group,
                topk_group=case.topk_group,
                routed_scaling_factor=case.routed_scaling_factor,
                ep_size=ep_size,
                tp_size=tp_size,
            )
        )
    return cases


def build_mesh(ep_size: int = 1, tp_size: int = 1):
    if ep_size <= 0 or tp_size <= 0:
        raise ValueError(f"Expected {ep_size=} and {tp_size=} to be > 0.")
    devices = jax.devices()[: ep_size * tp_size]
    return create_device_mesh(
        ici_parallelism=[tp_size, ep_size],
        dcn_parallelism=[1, 1],
        devices=devices,
        mesh_axes=("data", "tensor"),
    )


class MoEImbalanceSimulator:
    @staticmethod
    def generate_counts(num_tokens, top_k, num_experts, mode, **kwargs):
        np.random.seed(42)
        rng = np.random.default_rng(42)
        total_picks = num_tokens * top_k

        all_indices = np.arange(num_experts)
        rng.shuffle(all_indices)
        probs = np.zeros(num_experts)

        if mode == "balanced":
            probs = np.full(num_experts, 1.0 / num_experts)

        elif mode == "dirichlet":
            alpha = kwargs.get("alpha", 1.0)
            probs = rng.dirichlet([alpha] * num_experts)

        elif mode == "zipf":
            s = kwargs.get("zipf_s", 1.1)
            ranks = np.arange(1, num_experts + 1)
            weights = 1 / (ranks**s)
            rng.shuffle(weights)
            probs = weights / weights.sum()

        elif mode in ["hotspot", "sparse_hotspot"]:
            hotspot_ratio = kwargs.get("hotspot_ratio", 0.5)
            hotspot_count = kwargs.get("hotspot_count", 1)
            alpha_base = kwargs.get("non_hotspot_alpha", 100.0)

            zero_spot_count = kwargs.get("zero_expert_count", 0) if mode == "sparse_hotspot" else 0

            # 边界检查
            hotspot_count = max(1, min(hotspot_count, num_experts - zero_spot_count - 1))

            # 分层挑选索引
            zero_indices = all_indices[:zero_spot_count]
            hot_indices = all_indices[zero_spot_count : zero_spot_count + hotspot_count]
            base_indices = all_indices[zero_spot_count + hotspot_count :]

            # 1. 零负载组
            probs[zero_indices] = 0.0
            # 2. 热点组 (平分热点配额)
            probs[hot_indices] = hotspot_ratio / hotspot_count
            # 3. 基础活跃组 (按 Dirichlet 分布划分剩余配额)
            if len(base_indices) > 0:
                base_dist = rng.dirichlet([alpha_base] * len(base_indices))
                probs[base_indices] = base_dist * (1 - hotspot_ratio)
            else:
                # 如果没有基础组，热点组拿走全部
                probs[hot_indices] = 1.0 / hotspot_count

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 将概率转换为具体的 token 数量
        print(f"{probs=}")
        counts = np.floor(probs * total_picks).astype(int)
        # 补齐因舍入导致的差值
        diff = total_picks - counts.sum()
        if diff > 0:
            indices = np.random.choice(num_experts, diff, p=probs)
            for idx in indices:
                counts[idx] += 1
        return counts

    @staticmethod
    def create_logits_from_counts(num_tokens, num_experts, top_k, counts):
        """
        根据目标 count 构造 router_logits。
        构造逻辑：为每个 token 分配 top_k 个专家，确保全局总数符合 counts。
        """
        # 创建专家池，按 counts 数量填充专家 ID
        expert_pool = []
        for e_id, c in enumerate(counts):
            expert_pool.extend([e_id] * c)
        np.random.shuffle(expert_pool)

        # 填充到 (num_tokens, top_k)
        expert_pool = expert_pool[: num_tokens * top_k]
        assignments = np.array(expert_pool).reshape(num_tokens, top_k)

        # 构造 logits: 被选中的专家给高分 (10.0), 未选中的给低分 (-10.0)
        logits = np.full((num_tokens, num_experts), -10.0, dtype=np.float32)
        row_indices = np.arange(num_tokens)[:, None]
        logits[row_indices, assignments] = 10.0

        return jnp.array(logits)
