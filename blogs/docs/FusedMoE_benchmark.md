# FusedMoE benchmark

## experts simulator 四种模式

### 模式 A：Dirichlet 集中度模式（全局受控）

这是最通用的统计学方法。使用 **Dirichlet 分布** 可以非常优雅地控制整体的“公平性”。

*   **Factor** $\alpha$**(Concentration):**

    *   $\alpha\rarr无穷$: 专家分布趋于绝对平均。

    *   $\alpha=0$: 专家负载随机分布。

    *   $\alpha\rarr0$: 负载极度不均衡，绝大部分 Token 集中在极个别专家身上。

*   **优点:** 只需要一个参数就能控制从“完美平均”到“极度倾斜”的平滑过渡。


### 模式 B：Zipf's Law 模式（长尾分布）

模拟自然语言中词频分布的规律。在 MoE 中，某些专家（例如处理常见虚词的专家）通常比处理生僻词的专家负载重。

*   **Factor** $s$ **(Skewness):**

    *   $s=0$ : 均匀分布。

    *   $s > 1$ : 典型的长尾分布，排名第 1 的专家负载远高于后面的。

*   **场景:** 用于模拟模型训练中后期，专家路由逐渐收敛后的自然不均衡状态。


### 模式 C：Hotspot（热点/刺突）模式

针对性模拟“瓶颈”。你可以指定 k 个专家作为热点，并给予它们不成比例的负载。

*   **Factor**$k$ **(Hotspot Count):** 有多少个专家是超载的。

*   **Factor** $R$ **(Load Ratio):** 这 $k$ 个专家占据了总 Token 数的百分之多少。

*   **计算逻辑:**

    1.  指定 $k$ 个专家分配 $N\times R$ 个 Token。

    2.  剩下 $E - k$ 个专家瓜分 $N\times (1-R)$ $ 个 Token。


*   **场景:** 专门测试当某一两个 Device 上的 All-to-All 接收缓冲区溢出或处理超时时的系统表现。


### 模式 D: Sparse Hotspot （稀疏热点）模式

在实际的大规模分布式推理中，这种“部分专家完全闲置”的情况经常发生（例如某些长文本序列中只有特定的知识领域被激活）

*   **Factor**$z$**(Cold Expert Count):** 无 token 路由的 expert 数量。

*   **Factor**$k$ **(Hotspot Count):** k 个专家是极热。

*   **Factor** $E - k - z$ **(Cool Count):** 剩余专家少量token

*   **Factor** $R$ **(Load Ratio):** 这 $k$ 个专家占据了总 Token 数的百分之多少。剩余的token由cool expert评分

*   **场景:** 用于模拟模型训练中后期，专家路由逐渐收敛后的自然不均衡状态。


## 测试结果

### NumTokens=4096

| 分布(sparse\_hotspot ) | all enable(baseline) | all disabled(baseline) | enable a2a | enable dynamic\_ffn1 | enable dynamic\_ffn2 | enable weight\_load | enable a2a\_s\_tile\_read | enable a2a\_s\_acc\_tile\_write | enable shared\_expert |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \--hotspot-ratio 1<br>\--hotspot-count 8 | 11.675 | 0.255 | 3.035 | 4.8 | 3.999 | 0.539 | 1.729 | 1.934 | 0.307 |  |  |  |
| \--hotspot-ratio 1p- <br>\--hotspot-count 16 | 7.135 | 0.255 | 2.76 | 2.982 | 2.504 | 0.54 | 1.149 | 1.313 | 0.308 |  |  |  |
| \--hotspot-ratio 1 <br>\--hotspot-count 32 | 4.982 | 0.255 | 1.861 | 2.072 | 1.753 | 0.54 | 0.859 | 1.025 | 0.308 |  |  |  |
| \--hotspot-ratio 1 <br>\--hotspot-count 64 | 3.161 | 0.254 | 1.536 | 1.165 | 1.006 | 0.538 | 0.566 | 0.688 | 0.308 |  |  |  |
| \--hotspot-ratio 1 <br>\--hotspot-count 128 | 2.239 | 0.255 | 1.219 | 0.712 | 0.632 | 0.538 | 0.419 | 0.501 | 0.308 |  |  |  |
| \--hotspot-ratio 1 <br>\--hotspot-count 256 | 2.086 | 0.254 | 1.137 | 0.711 | 0.632 | 0.539 | 0.419 | 0.5 | 0.307 |  |  |  |

### NumTokens=128

| 分布(sparse\_hotspot ) | all enable(baseline) | all disabled(baseline) | enable a2a | enable dynamic\_ffn1 | enable dynamic\_ffn2 | enable weight\_load | enable a2a\_s\_tile\_read | enable a2a\_s\_acc\_tile\_write | enable shared\_expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \--hotspot-ratio 1<br>\--hotspot-count 8 | 2.997 | 0.225 | 0.297 | 1.776 | 1.101 | 0.473 | 0.71 | 0.485 | 0.262 |
| \--hotspot-ratio 1 <br>\--hotspot-count 16 | 2.014 | 0.225 | 0.296 | 1.155 | 0.751 | 0.473 | 0.518 | 0.393 | 0.26 |
| \--hotspot-ratio 1 <br>\--hotspot-count 32 | 1.52 | 0.225 | 0.266 | 0.844 | 0.574 | 0.473 | 0.419 | 0.336 | 0.26 |
| \--hotspot-ratio 1 <br>\--hotspot-count 64 | 1.024 | 0.226 | 0.262 | 0.535 | 0.401 | 0.476 | 0.325 | 0.299 | 0.261 |
| \--hotspot-ratio 1 <br>\--hotspot-count 128 | 0.784 | 0.225 | 0.256 | 0.38 | 0.314 | 0.473 | 0.276 | 0.263 | 0.261 |
| \--hotspot-ratio 1 <br>\--hotspot-count 256 | 0.724 | 0.225 | 0.261 | 0.327 | 0.282 | 0.473 | 0.273 | 0.264 | 0.26 |

### 优化 SharedExpert 切分 hidden\_size + 重排 fetch weight/tokens 位置

| 分布(sparse\_hotspot ) | all enable(baseline) | all disabled(baseline) | enable a2a | enable dynamic\_ffn1 | enable dynamic\_ffn2 | enable weight\_load | enable a2a\_s\_tile\_read | enable a2a\_s\_acc\_tile\_write | enable shared\_expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \--hotspot-ratio 1<br>\--hotspot-count 8 | 2.771 | 0.225 | 0.297 | 1.69 | 1.002 | 0.488 | 0.495 | 0.456 | 0.283 |
| \--hotspot-ratio 1 <br>\--hotspot-count 16 | 1.855 | 0.225 | 0.296 | 1.104 | 0.691 | 0.489 | 0.386 | 0.371 | 0.282 |
| \--hotspot-ratio 1 <br>\--hotspot-count 32 | 1.38 | 0.225 | 0.266 | 0.809 | 0.534 | 0.488 | 0.331 | 0.325 | 0.282 |
| \--hotspot-ratio 1 <br>\--hotspot-count 64 | 0.926 | 0.225 | 0.261 | 0.518 | 0.38 | 0.489 | 0.279 | 0.289 | 0.282 |
| \--hotspot-ratio 1 <br>\--hotspot-count 128 | 0.712 | 0.225 | 0.256 | 0.372 | 0.303 | 0.488 | 0.253 | 0.258 | 0.282 |
| \--hotspot-ratio 1 <br>\--hotspot-count 256 | 0.648 | 0.225 | 0.263 | 0.322 | 0.275 | 0.489 | 0.252 | 0.259 | 0.282 |

### 优化 shared expert 预取 + 优化 acc tile 预取

| 分布(sparse\_hotspot ) | all enable(baseline) | all disabled(baseline) | enable a2a | enable dynamic\_ffn1 | enable dynamic\_ffn2 | enable weight\_load | enable a2a\_s\_tile\_read | enable a2a\_s\_acc\_tile\_write | enable shared\_expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \--hotspot-ratio 1<br>\--hotspot-count 8 | 2.759 | 0.223 | 0.294 | 1.686 | 0.998 | 0.486 | 0.491 | 0.452 | 0.271 |
| \--hotspot-ratio 1 <br>\--hotspot-count 16 | 1.842 | 0.223 | 0.294 | 1.099 | 0.686 | 0.486 | 0.382 | 0.367 | 0.272 |
| \--hotspot-ratio 1 <br>\--hotspot-count 32 | 1.369 | 0.223 | 0.263 | 0.804 | 0.529 | 0.487 | 0.327 | 0.321 | 0.272 |
| \--hotspot-ratio 1 <br>\--hotspot-count 64 | 0.914 | 0.223 | 0.259 | 0.514 | 0.376 | 0.486 | 0.275 | 0.286 | 0.272 |
| \--hotspot-ratio 1 <br>\--hotspot-count 128 | 0.702 | 0.223 | 0.252 | 0.369 | 0.301 | 0.489 | 0.250 | 0.256 | 0.271 |
| \--hotspot-ratio 1 <br>\--hotspot-count 256 | 0.633 | 0.223 | 0.259 | 0.320 | 0.273 | 0.487 | 0.249 | 0.256 | 0.271 |

### 优化 FFN2 计算时对 FFN1 的权重预取 + 优化 shared Expert  计算插入 wait\_a2a\_scatter\_recv 间隙

| 分布(sparse\_hotspot ) | all enable(baseline) | all disabled(baseline) | enable a2a | enable dynamic\_ffn1 | enable dynamic\_ffn2 | enable weight\_load | enable a2a\_s\_tile\_read | enable a2a\_s\_acc\_tile\_write | enable shared\_expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \--hotspot-ratio 1<br>\--hotspot-count 8 | 2.756 | 0.223 | 0.295 | 1.689 | 1.004 | 0.467 | 0.479 | 0.353 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 16 | 1.830 | 0.223 | 0.294 | 1.105 | 0.691 | 0.466 | 0.375 | 0.306 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 32 | 1.350 | 0.223 | 0.263 | 0.809 | 0.533 | 0.467 | 0.322 | 0.276 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 64 | 0.895 | 0.224 | 0.257 | 0.516 | 0.378 | 0.466 | 0.273 | 0.258 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 128 | 0.677 | 0.223 | 0.254 | 0.370 | 0.302 | 0.465 | 0.249 | 0.242 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 256 | 0.615 | 0.223 | 0.259 | 0.320 | 0.273 | 0.466 | 0.249 | 0.243 | 0.288 |

### 优化 activation\_fn 的重复计算 -> 无收益

### 优化 合并 w1/w3 计算 -> 无收益

| 分布(sparse\_hotspot ) | all enable(baseline) | all disabled(baseline) | enable all\_reduce metadata | enable topk | enable sync\_barrier | enable a2a | enable dynamic\_ffn1 | enable dynamic\_ffn2 | enable weight\_load | enable a2a\_s\_tile\_read | enable a2a\_s\_acc\_tile\_write | enable shared\_expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \--hotspot-ratio 1<br>\--hotspot-count 8 | 2.756 | 0.223 | 0.053 | 0.006 | 0.039 | 0.295 | 1.689 | 1.004 | 0.467 | 0.479 | 0.353 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 16 | 1.830 | 0.223 | 0.053 | 0.006 | 0.039 | 0.294 | 1.105 | 0.691 | 0.466 | 0.375 | 0.306 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 32 | 1.350 | 0.223 | 0.053 | 0.006 | 0.039 | 0.263 | 0.809 | 0.533 | 0.467 | 0.322 | 0.276 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 64 | 0.895 | 0.224 | 0.053 | 0.006 | 0.039 | 0.257 | 0.516 | 0.378 | 0.466 | 0.273 | 0.258 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 128 | 0.677 | 0.223 | 0.053 | 0.006 | 0.039 | 0.254 | 0.370 | 0.302 | 0.465 | 0.249 | 0.242 | 0.288 |
| \--hotspot-ratio 1 <br>\--hotspot-count 256 | 0.615 | 0.223 | 0.053 | 0.006 | 0.039 | 0.259 | 0.320 | 0.273 | 0.466 | 0.249 | 0.243 | 0.288 |

# 理论计算

测试配置

*   tokens=128, experts=256, top\_k=8, hidden=8192, intermediate=2048, ep\_size=32

*   bt=4, bf=2048, bd1=2048, bd2=2048, btc=4, bfc=2048, bd1c=2048, bd2c=2048, bse256


## FFN1

FFN1 的 GEMM 计数 = 8(expert)\*2(bf\_id)_2(bd1\_id)_\[2(p)\*2(w1+w3)\] = 128 次

_每次 GEMM shape：(4×2048) @ (2048×1024)， FLOPs（按 2MAC）= 2\*_4\*_2048\*_1024 = 16,777,216 总 FLOPs：128\*16,777,216 = 2.147 GFLOPs

耗时 0.320 - 0.223 = 0.1 ms，对应的算力 21.46 TFLOPS

matmul 测试

```shell
# python3 ben.py --m 4 --k 2048 --n 1024
Device: TPU_0(process=0,(0,0,0,0))
matmul: <4, 2048> x <2048, 1024>
dtype: <class 'jax.numpy.bfloat16'>

=== Benchmarking =====
  -> Warming up...
  -> Warmed up in (120847.70 us)
  -> N=5000   Time=3560.05 us
  -> Warming up...
  -> Warmed up in (122228.60 us)
  -> N=10000  Time=6865.99 us
  -> Warming up...
  -> Warmed up in (126551.70 us)
  -> N=20000  Time=13432.61 us

[Analysis Result for matmul]
  R-squared: 1.0000
  1. Base Overhead (Intercept): 276.73 us
  2. Per-Iter Latency (Slope):  657.9554 ns
  3. Measured Throughput:       25499.02 Ops/ns
                                (25.50 TFLOPS equivalent)
```

## FFN2

FFN2 的 GEMM 计数（每芯片）：8\*2\*2\*2(p) = 64 次

每次 GEMM shape：(4×1024) @ (1024×2048)，FLOPs 同样 16,777,216

总 FLOPs：1.074 GFLOPs 

耗时 0.273 - 0.223 = 0.05 ms 对应有效算力约 21.48 TFLOPs

matmul 测试

```shell
# python3 ben.py --m 4 --k 1024 --n 2048
Device: TPU_0(process=0,(0,0,0,0))
matmul: <4, 1024> x <1024, 2048>
dtype: <class 'jax.numpy.bfloat16'>

=== Benchmarking =====
  -> Warming up...
  -> Warmed up in (121574.32 us)
  -> N=5000   Time=3580.91 us
  -> Warming up...
  -> Warmed up in (119904.90 us)
  -> N=10000  Time=6884.38 us
  -> Warming up...
  -> Warmed up in (125859.40 us)
  -> N=20000  Time=13761.62 us

[Analysis Result for matmul]
  R-squared: 0.9999
  1. Base Overhead (Intercept): 142.28 us
  2. Per-Iter Latency (Slope):  680.0016 ns
  3. Measured Throughput:       24672.32 Ops/ns
                                (24.67 TFLOPS equivalent)
```

# Pure A2A (完全均衡expert size)

```shell
from __future__ import annotations
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax._src.pallas.mosaic.helpers import sync_copy
import functools
import math
from jax.sharding import PartitionSpec as P

import gzip
import json
import os
import pathlib
import random
import re
import string
from typing import Any

import jax

MARKER = "SGLANG_JAX_BENCH"


def _extract_marker_durations_ms(trace: dict[str, Any], task: str | None = None) -> list[float]:
    marker_events: list[dict[str, Any]] = []
    for e in trace.get("traceEvents", []):
        args = e.get("args", {})
        tf_op = args.get("tf_op", "")
        if MARKER in tf_op:
            marker_events.append(e)

    marker_call_done_events = [e for e in marker_events if e.get(
        "name", "").endswith("call-done")]
    if marker_call_done_events:
        marker_events = marker_call_done_events

    def _durations_by_pid(events: list[dict[str, Any]]) -> dict[int, list[float]]:
        by_pid: dict[int, list[dict[str, Any]]] = {}
        for e in events:
            pid = e.get("pid")
            if isinstance(pid, int):
                by_pid.setdefault(pid, []).append(e)

        durations: dict[int, list[float]] = {}
        for pid, pid_events in by_pid.items():
            pid_events.sort(key=lambda ev: float(ev.get("ts", 0.0)))
            pid_durations: list[float] = []
            for e in pid_events:
                args = e.get("args", {})
                if args.get("device_duration_ps"):
                    pid_durations.append(
                        float(args["device_duration_ps"]) / 1e9)
                elif "dur" in e:
                    pid_durations.append(float(e["dur"]) / 1e3)
            if pid_durations:
                durations[pid] = pid_durations
        return durations

    if not marker_events:
        if not task:
            return []
        event_matcher = re.compile(task)
        events = []
        for e in trace.get("traceEvents", []):
            if "name" in e and event_matcher.match(e["name"]):
                events.append(e)
        durations_by_pid = _durations_by_pid(events)
        if not durations_by_pid:
            return []
        return max(sorted(durations_by_pid.items()), key=lambda kv: len(kv[1]))[1]

    durations_by_pid = _durations_by_pid(marker_events)
    if not durations_by_pid:
        return []
    return max(sorted(durations_by_pid.items()), key=lambda kv: len(kv[1]))[1]


def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")

    combined: dict[str, Any] = {"traceEvents": []}
    for trace_file in sorted(trace_files):
        with gzip.open(trace_file, "rb") as f:
            shard = json.load(f)
        shard_events = shard.get("traceEvents", [])
        if isinstance(shard_events, list):
            combined["traceEvents"].extend(shard_events)
        if "displayTimeUnit" in shard and "displayTimeUnit" not in combined:
            combined["displayTimeUnit"] = shard["displayTimeUnit"]
        if "otherData" in shard and "otherData" not in combined:
            combined["otherData"] = shard["otherData"]
    return combined


def multiple_iteration_timeit_from_trace(
    compute_func,
    data_generator,
    task: str,
    tries: int = 5,
    warmup: int = 0,
    trace_root: str = "/tmp/sglang_jax_moe_trace",
) -> list[float]:
    """
    Profile multiple iterations and pull per-iteration kernel time from trace.
    """
    trace_name = f"{task}_" + \
        "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    trace_dir = os.path.join(trace_root, trace_name)
    os.makedirs(trace_dir, exist_ok=True)

    for _ in range(max(0, int(warmup))):
        data_args = data_generator()
        out = compute_func(*data_args)
        jax.block_until_ready(out)

    with jax.profiler.trace(trace_dir):
        for i in range(tries):
            data_args = data_generator()
            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):
                    out = compute_func(*data_args)
                    jax.block_until_ready(out)

    trace = _load_trace(trace_dir)
    return _extract_marker_durations_ms(trace, task=task)


ep_axis_name = "ep"
data_axis_name = "data"

# Pallas Kernel Definition


def _remote_copy_kernel(
    # Prefetch SMEM
    routing_ref,
    # Inputs (in HBM)
    send_buf_ref,       # Ref to source data
    recv_buf_ref,       # Ref to destination data (on remote device)
    # Outputs
    # Output ref (if we want to return something or modify in place)
    out_ref,
    # scratch
    send_sem_ref,  # Semaphore on source device
    recv_sem_ref,  # Semaphore on destination device
    *,
    top_k: int = 8,
    ep_size: int = 32,
    num_experts: int = 256,
):
    local_token_num = send_buf_ref.shape[0]
    # Initiate the asynchronous remote copy.
    # This copies content from src_ref (local) to dst_ref (remote).
    # send_sem_ref is signaled when the data leaves the source.
    # recv_sem_ref is signaled when the data arrives at the destination.
    my_id = lax.axis_index(ep_axis_name)
    local_num_experts = num_experts // ep_size
    tokens_per_expert = local_token_num * ep_size * top_k // num_experts

    def a2a_expert(local_e_id):
        send_size = tokens_per_expert // ep_size * (ep_size - 1)
        for t_id in range(local_token_num):
            for iter in range(top_k):
                e_id = routing_ref[t_id, iter]
                is_active_expert = e_id % local_num_experts == local_e_id
                sz = lax.select(is_active_expert, jnp.int32(1), jnp.int32(0))
                target_device_id = e_id // local_num_experts
                is_local = target_device_id == my_id
                local_sz = lax.select(is_local, sz, 0)
                remote_sz = lax.select(is_local, 0, sz)
                recv_start = (e_id % local_num_experts) * tokens_per_expert
                pltpu.make_async_copy(
                    src_ref=send_buf_ref.at[pl.ds(t_id, local_sz)],
                    dst_ref=recv_buf_ref.at[pl.ds(recv_start, local_sz)],
                    sem=recv_sem_ref,
                ).start()
                pltpu.make_async_remote_copy(
                    send_buf_ref.at[pl.ds(t_id, remote_sz)],
                    recv_buf_ref.at[pl.ds(recv_start, remote_sz)],
                    send_sem_ref,
                    recv_sem_ref,
                    device_id=(target_device_id,),
                    # Use mesh coordinates or linear index depending on setup
                    device_id_type=pltpu.DeviceIdType.MESH
                ).start()
        pltpu.make_async_copy(
            src_ref=recv_buf_ref.at[pl.ds(0, tokens_per_expert)],
            dst_ref=recv_buf_ref.at[pl.ds(0, tokens_per_expert)],
            sem=recv_sem_ref,
        ).wait()

        pltpu.make_async_copy(
            src_ref=recv_buf_ref.at[pl.ds(0, send_size)],
            dst_ref=recv_buf_ref.at[pl.ds(0, send_size)],
            sem=send_sem_ref,
        ).wait()

    def a2a():
        for local_e_id in range(local_num_experts):
            a2a_expert(local_e_id)
    a2a()


@functools.partial(
    jax.jit,
    static_argnames=[
        "mesh",
        "ep_size",
    ],
)
# Example usage wrapper
def run_remote_copy_demo(mesh, ep_size):
    # Data shape
    token_size = 4096
    hidden_size = 8192
    packing = 2
    top_k = 8
    send_buf_shape = (token_size, packing, hidden_size//packing)
    dtype = jnp.bfloat16
    num_experts = 256

    scratch_shapes = (
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    )

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.ANY)
    ]

    pl_call = pl.pallas_call(
        functools.partial(
            _remote_copy_kernel,
            top_k=top_k,
            ep_size=ep_size,
            num_experts=num_experts,
        ),
        out_shape=[
            # dst (updated) - conceptually
            jax.ShapeDtypeStruct(
                (token_size, ep_size, packing, hidden_size), dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=in_specs,  # src, dst, send_sem, recv_sem
            out_specs=out_specs,  # output
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(),
        ),
        name="test"
    )

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=P(ep_axis_name,),
        out_specs=P(ep_axis_name,),
        check_vma=False,
    )
    def kernel(send_buf):
        tokens_per_device = send_buf.shape[0]
        recv_buf_shape = (tokens_per_device*top_k,
                          packing, hidden_size//packing)
        recv_buf = jnp.zeros(recv_buf_shape, dtype=dtype)

        padded_top_k = 128

        # 构造 t2e_routing: (output_bt, padded_top_k)
        # 每个 token i 的第 k 个专家设定为 (i * top_k + k) % num_experts
        t_iota = jax.lax.broadcasted_iota(jnp.int32, (tokens_per_device, padded_top_k), 0)
        k_iota = jax.lax.broadcasted_iota(jnp.int32, (tokens_per_device, padded_top_k), 1)
        t2e_routing = (t_iota * top_k + k_iota) % num_experts
        output = pl_call(t2e_routing, send_buf, recv_buf)
        return output

    send_buf = jnp.arange(0, math.prod(send_buf_shape),
                          dtype=dtype).reshape(send_buf_shape)
    return kernel(send_buf)[0]


if __name__ == "__main__":
    ep_size = jax.device_count()
    mesh = jax.make_mesh((ep_size,), (ep_axis_name))
    with jax.set_mesh(mesh):
        output = run_remote_copy_demo(mesh, ep_size)
        output.block_until_ready()
        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: run_remote_copy_demo(mesh, ep_size),
            data_generator=lambda: (),
            task=f"test",
            tries=1,
        )
        if len(times) > 1:
            times = times[1:]
        mean_ms = float(np.mean(times)) if times else float("nan")
        print(f"{mean_ms:.8}ms")

```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/yBRq1ZP4GyBx4Odv/img/860bde97-4e32-41fc-9e4b-4d0fc179380e.png)

### Config

*   token num: 4096

*   ep size: 32

*   expert number: 256

*   hidden size: 8192

*   top k：8

*   dtype：bf16


### 耗时：0.387ms

### 吞吐：40GB/s

## 优化思路

1.  prefill expert\_ffn 计算占了大头，提升比较有效的方法是增加 bd/bf block size，这受限于 vmem 大小，我们需要修改一些逻辑将 vmem 腾挪出来给到这部分

2.  decode 


## 端到端测试

### profiling

decode (batch\_size=128)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/yBRq1ZP4GyBx4Odv/img/1b92b00f-1571-4049-a85e-637a77eb8def.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/yBRq1ZP4GyBx4Odv/img/49452e59-ba2c-43f5-bb3b-e53391851411.png)

prefill(num\_tokens=4096)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/yBRq1ZP4GyBx4Odv/img/4cea703b-5945-4478-97dc-638ae66b3a10.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/yBRq1ZP4GyBx4Odv/img/77f13062-42a8-4b5f-83dc-edf506fdf001.png)

### benchmark

```shell
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server --model-path /models/model_scope/Ling-1T --trust-remote-code --tp-size=32 --ep-size=32 --device=tpu --random-seed=3 --mem-fraction-static=0.9 --chunked-prefill-size=16384 --dtype=bfloat16 --max-running-requests=256 --skip-server-warmup --page-size=128 --dist-init-addr=10.116.9.5:10011 --moe-backend=fused --nnodes=4 --node-rank=0
```

```shell
python3 -m sgl_jax.bench_serving --backend sgl-jax --dataset-name random --num-prompts 512 --random-input 4096 --random-output 1024 --max-concurrency 128 --random-range-ratio 1 --warmup-requests 0
```

```shell
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 128
Successful requests:                     512
Benchmark duration (s):                  898.51
Total input tokens:                      2097152
Total generated tokens:                  524288
Total generated tokens (retokenized):    516367
Request throughput (req/s):              0.57
Input token throughput (tok/s):          2334.03
Output token throughput (tok/s):         583.51
Total token throughput (tok/s):          2917.53
Concurrency:                             122.35
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   214716.03
Median E2E Latency (ms):                 180750.82
---------------Time to First Token----------------
Mean TTFT (ms):                          58222.21
Median TTFT (ms):                        36355.66
P99 TTFT (ms):                           188881.40
---------------Inter-Token Latency----------------
Mean ITL (ms):                           152.98
Median ITL (ms):                         126.28
P95 ITL (ms):                            131.95
P99 ITL (ms):                            133.67
Max ITL (ms):                            60745.61
==================================================
```

python -m sgl\_jax.bench\_one\_batch\_server --model None --base-url [http://localhost:30000](http://localhost:30000) --batch-size 64 128 256 --input-len 4096 --output-len 1024 --show-report

**Main 分支Baseline**

| batch size | latency (s) | input throughput (tok/s) | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 140.46 | 8430.00 | 599.27 | n/a | 106.80 | 0.09 | 0.93 |
| 128 | 373.51 | 2775.12 | 710.07 | n/a | 180.26 | 0.29 | 0.78 |
| 256 | 511.15 | 2727.32 | 2069.43 | n/a | 123.71 | 0.29 | 0.27 |

## Attention DP + MOE EP

### Profiling

decode (bs 256) evalscope中测试的，seqs长度对不齐

### Benchmark

```Plain
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server --model-path /models/model_scope/Ling-1T/ --trust-remote-code --ep-size=32 --tp-size=32 --dp-size=4 --device=tpu --random-seed=3 --mem-fraction-static=0.92 --chunked-prefill-size=8192 --dtype=bfloat16 --max-running-requests=512 --skip-server-warmup --page-size=128 --dist-init-addr=10.116.9.5:10011 --disable-radix-cache --nnodes=4 --moe-backend=fused --node-rank=0
```

```Plain
python3 -m sgl_jax.bench_serving --backend sgl-jax --dataset-name random --num-prompts 512 --random-input 4096 --random-output 1024 --max-concurrency 128 --random-range-ratio 1 --warmup-requests 0

============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 128
Successful requests:                     512
Benchmark duration (s):                  1357.86
Total input tokens:                      2097152
Total generated tokens:                  524288
Total generated tokens (retokenized):    505445
Request throughput (req/s):              0.38
Input token throughput (tok/s):          1544.46
Output token throughput (tok/s):         386.11
Total token throughput (tok/s):          1930.57
Concurrency:                             127.99
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   339429.74
Median E2E Latency (ms):                 342238.01
---------------Time to First Token----------------
Mean TTFT (ms):                          126465.60
Median TTFT (ms):                        124556.35
P99 TTFT (ms):                           239198.47
---------------Inter-Token Latency----------------
Mean ITL (ms):                           208.18
Median ITL (ms):                         99.36
P95 ITL (ms):                            104.46
P99 ITL (ms):                            106.93
Max ITL (ms):                            233178.83
==================================================

python3 -m sgl_jax.bench_serving --backend sgl-jax --dataset-name random --num-prompts 512 --random-input 4096 --random-output 1024 --max-concurrency 256 --random-range-ratio 1 --warmup-requests 0

============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 256
Successful requests:                     512
Benchmark duration (s):                  991.24
Total input tokens:                      2097152
Total generated tokens:                  524288
Total generated tokens (retokenized):    493104
Request throughput (req/s):              0.52
Input token throughput (tok/s):          2115.69
Output token throughput (tok/s):         528.92
Total token throughput (tok/s):          2644.61
Concurrency:                             255.96
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   495535.46
Median E2E Latency (ms):                 495470.58
---------------Time to First Token----------------
Mean TTFT (ms):                          177263.57
Median TTFT (ms):                        179025.62
P99 TTFT (ms):                           343106.94
---------------Inter-Token Latency----------------
Mean ITL (ms):                           311.12
Median ITL (ms):                         149.63
P95 ITL (ms):                            151.57
P99 ITL (ms):                            153.17
Max ITL (ms):                            337855.00
==================================================

```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/yBRq1ZP4GyBx4Odv/img/bfefefc5-7e30-417a-b319-c0fed625e144.png)

使用bench one batch测试

python -m sgl\_jax.bench\_one\_batch\_server --model None --base-url [http://localhost:30000](http://localhost:30000) --batch-size 64 128 256 --input-len 4096 --output-len 1024 --show-report

**DP=4**

| batch size | latency (s) | input throughput (tok/s) | output throughput (tok/s) | acc length | ITL (ms) |
| --- | --- | --- | --- | --- | --- |
| 64 | 213.38 | 2203.36 | 694.17 | n/a | 92.20 |
| 128 | 329.76 | 2287.17 | 1303.74 | n/a | 98.18 |
| 256 | 573.58 | 2294.04 | 2250.29 | n/a | 113.76 |

### 修复调度bug：

使用bench one batch测试

python -m sgl\_jax.bench\_one\_batch\_server --model None --base-url [http://localhost:30000](http://localhost:30000) --batch-size 64 128 256 --input-len 4096 --output-len 1024 --show-report

**DP=4**

**(最大decde bs跑到了380，显存还是不够)**

| batch size | latency (s) | input throughput (tok/s) | output throughput (tok/s) | acc length | ITL (ms) |
| --- | --- | --- | --- | --- | --- |
| 64 | 170.52 | 3404.02 | 700.83 | n/a | 91.32 |
| 128 | 251.35 | 3540.70 | 1269.19 | n/a | 100.85 |
| 256 | 427.73 | 3384.10 | 2223.90 | n/a | 115.11 |
| 512 | 916.74 | 2806.67 | 3092.44 | n/a | 165.57 |

## 长序列benchmark

JAX\_COMPILATION\_CACHE\_DIR=/tmp/jit\_cache python -u -m sgl\_jax.launch\_server --model-path /models/model\_scope/Ling-1T/ --trust-remote-code --ep-size=32 --tp-size=32 --dp-size=4 --device=tpu --random-seed=3 --mem-fraction-static=0.92 --chunked-prefill-size=8192 --dtype=bfloat16 --max-running-requests=512 --skip-server-warmup --page-size=256 --dist-init-addr=10.116.9.5:10011 --disable-radix-cache --nnodes=4 --moe-backend=fused --node-rank=0

### 32chip

JAX\_COMPILATION\_CACHE\_DIR=/tmp/jit\_cache python -u -m sgl\_jax.launch\_server --model-path /models/model\_scope/Ling-1T/ --trust-remote-code --ep-size=64 --tp-size=64 --dp-size=8 --device=tpu --random-seed=3 --mem-fraction-static=0.95 --chunked-prefill-size=8192 --dtype=bfloat16 --max-running-requests=256 --skip-server-warmup --page-size=256 --dist-init-addr=10.116.9.5:10011 --disable-radix-cache --nnodes=8 --moe-backend=fused --node-rank=0
