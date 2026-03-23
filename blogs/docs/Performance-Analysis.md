# Performance Analysis

## Motivation

**为何要做这个工作?**

目前我们已经可以通过 benchmark + profiling 知晓系统端到端、每个模块的性能表现，但我们仍然回答不了这是否已经足够(达到硬件的极限)，没有目标也就没有办法去做优化决策，更急切的，当系统适配到新的硬件时，我们应该如何去更有效地根据硬件特性来调整优化策略？

**如何做？**

推理系统，性能优化的核心目的之一是降低延迟，无论是 TTFT 还是 ITL。对于任意 workload，可以将操作拆分为数据供给和数据处理两个部分，大多数情况这两部分可以做到 overlap，所以整体的耗时其实是由这两部分耗时更长者决定，这就是计算访存比或者 roofline 临界值计算的核心诉求，当算法强度小于这个值时，耗时主要由数据供给端决定，反之则由数据处理端决定。当然，我们也绝不是一味地追求算法强度刚好达到平衡或者 compute bound，这也需要根据实际的 workload 来定，比如某一些 mask 操作，其本质实际是纯粹的访存操作，计算完全可以忽略不计。所以，我们追求的是适配 workload 的计算、访存特性，以最快地速度进行处理。roofline 给了我们一种方式，可以在已知算法强度时，知道当前受限的资源是什么以及处理性能，从而可以推导出实际处理时间。

当然，单一过程性能最优并不等同于全局性能最优，且推理系统还要兼顾吞吐指标的优化，所以我们需要一套工具链，以全局视角出发，将优化策略放置在实际的端到端场景中，以追求全局最优解。

## Key Concepts

### Roofline

#### What is roofline?

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDejGRL7j1OVx/img/7fc4e163-4d4e-4aee-9187-d4568d7c1334.png)

$Arithmetic\ Intensity = \frac{Computation\ FLOPs}{Communication/Memory\ Bytes}$

算法的算术强度等于其执行的总 FLOPs 与它需要访存或通信的字节数之比，

当硬件、算法数据供给、处理方式确定时，其 roofline 曲线便是确定的，算法实际性能为:

$FLOPS = min(Accelerator\ FLOPS,  Arithmetic\ Intensity \* Bandwidth)$

上图也是由这个公式画出，不同的曲线对应的不同的访存 $Bandwidth$.

临界(上图转折点)算法强度 

$Critical\ Arithmetic\ Intensity = \frac{Accelerator\ FLOPS}{Bandwidth}$

当算法强度 < 临界算法强度时，则为 memory 或 communication bound

当算法强度 > 临界算法强度时，则为 computation bound

#### Analysis example?

##### What capabilities does JAX provide to help us analyze?

1.  cost\_analysis 函数

    该函数可以返回 jit 之后的算子执行 FLOPs 数，以及访存信息，由此可以计算出算子的算法强度，如下例所示: (需要注意的是，目前 cost\_analysis 不支持 custom kernel)

    ```python
    import jax
    from jax import numpy as jnp

    def dot(x: jax.Array, y: jax.Array):
        return x @ y

    x = jnp.ones((4, 8), dtype=jnp.bfloat16)
    y = jnp.ones((8, 1), dtype=jnp.bfloat16)

    lower = jax.jit(dot).lower(x, y)
    compiled = lower.compile()
    cost = compiled.cost_analysis()
    print(f"{cost=}")

    # cost={'utilization1{}': 1.0, 'bytes accessed1{}': 16.0, 'utilization0{}': 2.0, 'bytes accessed': 120.0, 'bytes accessedout{}': 24.0, 'bytes accessed0{}': 80.0, 'flops': 108.0}
    ```

2.  jax profiler


jax profiler 提供了整体过程、单个模块的 roofline 信息 (可以参考下述 example 里的展示) 

##### example

###### example -- decode mlp

先分析如下过程(mlp down\_proj)，其 roofline 如下:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDejGRL7j1OVx/img/1162bb34-0094-4566-b8ad-21984a444b2b.png)

**Bottleneck Operational Intensity (FLOP/Byte)**: 64.0104

而 v6e HBM 的 **Operational Intensity (FLOP/Byte)**: 577.96

显然是 memory bound 的

先理论计算一下算法强度:

dot = \[64, 12288\] x \[12288, 4096\]

B = 64, D = 12288, F = 4096

算法强度 = 2BDF/(2BD + 2DF + 2BF) = BDF/(BD + DF + BF) = 62.6938

算法强度值与图上所示基本一致

当 B << D, B << F 的情况下，算法强度 $\approx$ BDF/DF = B，也就是说对于 logits 里的 dot 而言，其算法强度只与 B 有关(纯理论计算):

```python
     B | OI(F/byte) |   FLOPs(G) |  Bytes(GB) | F_eff(TF/s) |  t_est(ms) |    bound
-----------------------------------------------------------------------------------
     1 |      1.000 |      0.101 |      0.101 |       1.637 |      0.061 |   memory
     2 |      1.999 |      0.201 |      0.101 |       3.274 |      0.061 |   memory
     4 |      3.995 |      0.403 |      0.101 |       6.543 |      0.062 |   memory
     8 |      7.979 |      0.805 |      0.101 |      13.070 |      0.062 |   memory
    16 |     15.917 |      1.611 |      0.101 |      26.072 |      0.062 |   memory
    32 |     31.670 |      3.221 |      0.102 |      51.875 |      0.062 |   memory
    64 |     62.694 |      6.442 |      0.103 |     102.692 |      0.063 |   memory
   128 |    122.880 |     12.885 |      0.105 |     201.277 |      0.064 |   memory
   256 |    236.308 |     25.770 |      0.109 |     387.070 |      0.067 |   memory
   512 |    438.857 |     51.540 |      0.117 |     718.845 |      0.072 |   memory
  1024 |    768.000 |    103.079 |      0.134 |     946.700 |      0.109 |  compute
```

增加 B 可以增加 dot 的算法强度，在大多数情况下仍然是 memory bound 的，耗时实际仍然由数据访问(读/写)量决定，所以 B 增加后耗时并没有显著增加。

此外，我们还可以对比一下切分 D 维度(按 tp 维度去切分)所带来的收益，这里情况会稍微复杂一些，因为会增加一次 all-reduce 通信(卡间 ICI 带宽为 3200.0 Gbps):

```python
OI_allreduce = flops_allreduce_per_dev / comm_bytes_per_dev
             = (N * (P - 1) / P) / (2 * 2 * N * (P - 1) / P) # N 为元素个数，P 为卡数, 每个元素需要做 (P - 1) 次加法
             = 1/4 FLOPs/Byte
OI_crit_network = PEAK_FLOPs / BW_ici
                = 946.7e12 / 400e9
                = 2367 FLOPs/Byte
```

可以看到对于 all-reduce 过程而言，communication bound 严重，计算相对通信可以忽略不计，从耗时上评估的话可以只关注通信过程。B=128 时，耗时与 tp 的关系:

```python
  tp |  OI_dev(F/B) | FLOPs_total(G) | FLOPs_dev(G) | HBM_dev(GB) |   t_comm(ms) | F_eff_dev(TF/s) | t_total(ms) | bound
--------------------------------------------------------------------------------------------------------------------------
   1 |      122.880 |       12.885 |       12.885 |       0.105 |        0.000 |         201.277 |       0.064 |   memory
   2 |      119.301 |       12.885 |        6.442 |       0.054 |        0.008 |         157.778 |       0.041 |   memory
   4 |      112.734 |       12.885 |        3.221 |       0.029 |        0.012 |         110.162 |       0.029 |   memory
   8 |      101.554 |       12.885 |        1.611 |       0.016 |        0.014 |          68.698 |       0.023 |   memory
  16 |       84.745 |       12.885 |        0.805 |       0.010 |        0.015 |          39.193 |       0.021 |   memory
  32 |       63.668 |       12.885 |        0.403 |       0.006 |        0.015 |          21.083 |       0.019 |   memory
```

可以看到: 

1. 增加 tp 不会显著增加通信耗时，因为 all-reduce 在 tp 较大时通信量仅与元素个数(即 BF)有关，另外由于 tp 增加并不显著改变算法强度，因此整体耗时仍然由访存决定(memory bound)，而 tp 增加显著降低了单卡访存量，所以耗时显著地降低了。

2. 增加 tp 会显著降低单卡的算法强度

上述为理论计算，实际测试(tp=4)结果如下:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDejGRL7j1OVx/img/ef603ba0-d6a6-4c08-83a8-487ff3242143.png)

基本验证了理论推导。

总结一下: 

1. 算法强度和 roofline 决定了整个过程被什么资源 bound，耗时由被 bound 资源的数据处理速度及数据量决定。

2. 上述所有的讨论都只限于单一的过程，端到端的过程由无数个单一过程组成，但每个过程的 roofline 可能不同，bound 的资源也不同，当我们去研究某一些量对于耗时的影响时，得综合来考虑。

###### example -- prefill mlp

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDejGRL7j1OVx/img/af997e1a-ef49-4803-be18-bd6c6da47a42.png)

我们目前的 prefill 整体过程已经非常接近于算法强度临界值(462.1841 < 577.96)

我们继续摘取 mlp down\_proj dot 过程:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDejGRL7j1OVx/img/ef4f128d-20b2-4c21-8b45-8b349e614256.png)

当前 dot 对应参数:

S = 1024(chunked\_prefill), D = 12288/4 = 3072, F = 4096 (TP = 4)

dot = \[1024, 12288/4\] x \[12288/4, 4096\]

算法强度 = 646.7368 > 577.96 为 compute bound! 

我们来研究一下 tp、bsz 对该过程的影响:

```python
       B |   tp |  OI_dev(F/B) |   t_comp(ms) |   t_comm(ms) | F_eff_dev(TF/s) | t_total(ms) |    bound
-------------------------------------------------------------------------------------------------------
    1024 |    1 |      768.000 |        0.109 |        0.000 |         946.700 |       0.109 |  compute
    1024 |    2 |      722.824 |        0.054 |        0.021 |         946.700 |       0.075 |  compute
    1024 |    4 |      646.737 |        0.027 |        0.031 |         946.700 |       0.059 |  compute
    1024 |    8 |      534.261 |        0.015 |        0.037 |         875.116 |       0.051 |   memory
    1024 |   16 |      396.387 |        0.010 |        0.039 |         649.279 |       0.049 |   memory
    1024 |   32 |      261.447 |        0.008 |        0.041 |         428.248 |       0.048 |   memory
    2048 |    1 |     1228.800 |        0.218 |        0.000 |         946.700 |       0.218 |  compute
    2048 |    2 |     1117.091 |        0.109 |        0.042 |         946.700 |       0.151 |  compute
    2048 |    4 |      945.231 |        0.054 |        0.063 |         946.700 |       0.117 |  compute
    2048 |    8 |      722.824 |        0.027 |        0.073 |         946.700 |       0.101 |  compute
    2048 |   16 |      491.520 |        0.016 |        0.079 |         805.106 |       0.095 |   memory
    2048 |   32 |      299.707 |        0.013 |        0.081 |         490.919 |       0.094 |   memory
    4096 |    1 |     1755.429 |        0.436 |        0.000 |         946.700 |       0.436 |  compute
    4096 |    2 |     1536.000 |        0.218 |        0.084 |         946.700 |       0.302 |  compute
    4096 |    4 |     1228.800 |        0.109 |        0.126 |         946.700 |       0.235 |  compute
    4096 |    8 |      877.714 |        0.054 |        0.147 |         946.700 |       0.201 |  compute
    4096 |   16 |      558.545 |        0.028 |        0.157 |         914.894 |       0.185 |   memory
    4096 |   32 |      323.368 |        0.024 |        0.163 |         529.675 |       0.187 |   memory
    8192 |    1 |     2234.182 |        0.871 |        0.000 |         946.700 |       0.871 |  compute
    8192 |    2 |     1890.462 |        0.436 |        0.168 |         946.700 |       0.603 |  compute
    8192 |    4 |     1445.647 |        0.218 |        0.252 |         946.700 |       0.469 |  compute
    8192 |    8 |      983.040 |        0.109 |        0.294 |         946.700 |       0.402 |  compute
    8192 |   16 |      599.415 |        0.054 |        0.315 |         946.700 |       0.369 |  compute
    8192 |   32 |      336.658 |        0.047 |        0.325 |         551.443 |       0.372 |   memory
```

可以看到: 1. 当达到 compute bound 后，增加 B，会线性增加耗时 2. tp 增加，会降低算法强度，在某一些组合下会退化成 memory bound，虽然 tp 的增大耗时确实也是在减小的，但是边际效应比较明显。

对比 decode 阶段的结果，实际对于同一个过程，在做参数、并行调优时并没有明显的规律，需要具体情况具体分析，抛开量谈效果没有意义。

### JIT

:::
为何需要了解 JAX/XLA 编译优化过程，有两个原因: 1. 补齐编译优化知识的空白，了解优秀的编译器在编译优化方面做了哪些工作。2. TPU 实际执行的也是它编译优化之后的产物，我们的很多工作也是基于这个之上去做，所以需要去详细了解。
:::

#### Principles?

compilation process:

Jaxpr -----> StableHLO -------> HLO --------> LLO (TPU)

:::
Jaxpr → StableHLO: JAX’s MLIR bridge lowers high‑level JAX IR into a StableHLO MLIR module.

StableHLO → HLO: XLA front‑end converts StableHLO to XLA HloModule and runs many graph optimizations.

HLO → LLO: XLA backend compilers (CPU/GPU/TPU) lower HLO to LLVM IR / PTX / TPU binaries and produce the final executable.
:::

Jax:

```python
import jax
from jax import numpy as jnp

@jax.jit
def dot(x: jax.Array, y: jax.Array):
    return x @ y

dot(jnp.ones((4, 8), dtype=jnp.bfloat16), jnp.ones((8, 1), dtype=jnp.bfloat16))
```

Jaxpr:

```python
{
  lambda ; a:bf16[4,8] b:bf16[8,1]. let
    c:bf16[4,1] = jit[
      name=dot
      jaxpr={ lambda ; a:bf16[4,8] b:bf16[8,1]. let
          c:bf16[4,1] = dot_general[
            dimension_numbers=(([1], [0]), ([], []))
            preferred_element_type=bfloat16
          ] a b
        in (c,) }
    ] a b
  in (c,)
}
```

StableHLO:

```python
module @jit_dot attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x8xbf16>, %arg1: tensor<8x1xbf16>) -> (tensor<4x1xbf16> {jax.result_info = "result"})
  {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x8xbf16>, tensor<8x1xbf16>) -> tensor<4x1xbf16>
    return %0 : tensor<4x1xbf16>
  }
}
```

XLA HLO:

```python
HloModule jit_dot, is_scheduled=true, entry_computation_layout={(bf16[4,8]{1,0}, bf16[8,1]{1,0})->bf16[4,1]{1,0}}, allow_spmd_sharding_propagation_to_parameters={true,true}, allow_spmd_sharding_propagation_to_output={true}

%fused_computation (param_0.1: f32[4]) -> bf16[4,1] {
  %param_0.1 = f32[4]{0} parameter(0)
  %convert.4 = bf16[4]{0} convert(%param_0.1)
  ROOT %bitcast.2 = bf16[4,1]{1,0} bitcast(%convert.4)
}

%fused_computation.1 (param_0.4: bf16[4,8], param_1.3: bf16[8,1]) -> f32[4] {
  %param_0.4 = bf16[4,8]{1,0} parameter(0)
  %convert.5 = f32[4,8]{1,0} convert(%param_0.4)
  %param_1.3 = bf16[8,1]{1,0} parameter(1)
  %convert.6 = f32[8,1]{1,0} convert(%param_1.3)
  %bitcast.3 = f32[8]{0} bitcast(%convert.6)
  ROOT %dot.1 = f32[4]{0} dot(%convert.5, %bitcast.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(dot)/dot_general"}
}

ENTRY %main.1 (x.1: bf16[4,8], y.1: bf16[8,1]) -> bf16[4,1] {
  %x.1 = bf16[4,8]{1,0} parameter(0), metadata={op_name="x"}
  %y.1 = bf16[8,1]{1,0} parameter(1), metadata={op_name="y"}
  %convert_dot_fusion = f32[4]{0} fusion(%x.1, %y.1), kind=kLoop, calls=%fused_computation.1, metadata={op_name="jit(dot)/dot_general"}
  ROOT %convert_bitcast_fusion = bf16[4,1]{1,0} fusion(%convert_dot_fusion), kind=kLoop, calls=%fused_computation
}
```

我们面对的不仅是单纯的 jax 算子，实际运行的其实是 XLA 编译优化过的 HLO，里面会包含算子融合、重整。

#### How does XLA perform compilation optimizations?

## 工具链

目前分析工具的局限性: 

1. custom kernel 无法获取到算法强度，如计算量以及访存量 

2. 可以输出每个模块的 roofline，但缺乏全局视角，无法给出优化决策

我现在还没有想好这个工具链具体长什么样子，需要进一步地讨论，但以我的理解，它至少需要解决以下问题:

1.  在当前运行配置下，单个模块/算子是否已达到极限性能？ -->  指导单个算子的优化，如提升算法强度

2.  在当前运行配置下，算子融合方式是否最优？--> 虽然已经有了 jax.jit，但融合效果还是会受用户代码影响，需要拆开审视

3.  在当前运行配置下，并行策略是否是最优的？--> 根据核心优化目标(延迟、吞吐)，优化并行策略

4.  在更复杂的应用场景下，如何评估影响？如 EP 或 DP 负载不均衡如何影响性能指标(定量分析)


其中，第 2、3、4 点都依赖于计算图的输入

## Computation Graph

<绘制完整计算图，下为示意图>

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDejGRL7j1OVx/img/06aecf3f-8ec5-4598-9bd3-b0c0017b6c12.png)

## Reference

\[1\] [https://jax-ml.github.io/scaling-book/roofline/](https://jax-ml.github.io/scaling-book/roofline/)
