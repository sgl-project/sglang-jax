# tpu-inference Fused MoE Profiling 记录

## 环境
- 单机 4 卡 TPU v7（8 cores），EP=8
- VMEM 上限：64MB per core
- kernel: `blogs/tpu-inference/fused_moe/v1/kernel.py`

## Ring-1T-FP8 参数
- num_experts: 64 (受限于机器规模，原始 256)
- top_k: 8
- hidden_size: 8192
- intermediate_size: 2048
- weight_dtype: float8_e4m3fn (w_packing=4)
- scoring_fn: sigmoid
- t_packing: 2 (bfloat16 tokens)

## VMEM 瓶颈分析

kernel 有两个最大的 a2a VMEM scratch buffer（`a2a_s_x2_vmem` + `a2a_s_acc_x2_vmem`）：
- shape: `(2, bt * ep_size, t_packing, hidden_size // t_packing)` in bf16
- 每个 size: `bt * 8 * 2 * 4096 * 2 = bt * 0.25 MB`
- bt=64 → 每个 16MB，两个共 32MB
- bt=128 → 每个 32MB，两个共 64MB（= 全部 VMEM，不可行）

其他主要 buffer:
- `a2a_g_acc_vmem`: `(top_k, bt, 2, 4096)` bf16 → bt=64 时 8MB
- `b_output_x2_vmem`: `(2, bt, 8192)` bf16 → bt=64 时 2MB
- weight buffers (fp8): 随 bf/bd1/bd2 变化，通常 2-8MB

bt=64 时总 VMEM ≈ 32 + 8 + 2 + 8 = ~50MB，在 64MB 内。

## 成功运行的配置

| tokens | local_tokens | bt | bf | bd1 | bd2 | btc | bfc | bd1c | bd2c | 编译时间 |
|--------|-------------|----|----|-----|-----|-----|-----|------|------|---------|
| 512 | 64 | 64 | 1024 | 1024 | 2048 | 32 | 1024 | 1024 | 2048 | ~1min |
| 1024 | 128 | 64 | 1024 | 1024 | 2048 | 32 | 1024 | 1024 | 2048 | ~2min |
| 2048 | 256 | 64 | 1024 | 1024 | 2048 | 32 | 1024 | 1024 | 2048 | ~3min |
| 4096 | 512 | 64 | 1024 | 1024 | 2048 | 32 | 1024 | 1024 | 2048 | ~5min |
| 8192 | 1024 | 64 | 1024 | 1024 | 2048 | 32 | 1024 | 1024 | 2048 | ~20min |

## 失败记录

| tokens | config | 原因 |
|--------|--------|------|
| 1024+ | bt=128 (default) | VMEM 108.55M > 64M，两个 a2a buffer 各 32MB |
| 8192 | bt=128, bf=512, bd1/bd2=512 | VMEM 97.89M > 64M |
| 8192 | bt=64, bf=512, bd1=512, bd2=1024 | 编译 hang >20min（内层循环次数增多） |

## 关键发现

1. **bt 上限**: hidden_size=8192 + ep_size=8 时，bt 最大只能 64。bt=128 时仅两个 a2a buffer 就占满 64MB VMEM
2. **编译时间**: 外层循环 = local_num_tokens / bt，内层循环 = hidden/bd + intermediate/bf。两者都影响编译时间
   - 缩小 bf/bd 虽然节省 VMEM，但增加内层循环，反而让编译更慢
   - 应保持 bf/bd 尽可能大，仅缩小 bt 来适配 VMEM
3. **8192 tokens 可行**: bt=64 + 最大 bf/bd 约需 20 分钟编译，但能成功运行
4. **default 公式不适用**: `get_default_block_sizes()` 对 hidden=8192 给出 bt=128，会爆 VMEM
