# ling_v3_flash Long-Context MoE Backend Benchmark

单脚本跑 **v1 / v2 / v4** 三方 MoE backend head-to-head 跑分。
模型：ling_v3_flash, bs=1, in=4096, out=2048, tp=4。

## 用法

```bash
# 一键跑全部 3 个 backend
./ling-test/longctx-bench/run.sh

# 只跑指定 backend
./ling-test/longctx-bench/run.sh v1          # 只跑 fused (v1 EP)
./ling-test/longctx-bench/run.sh v2 v4       # 只跑 fused_v2 + fused_v4

# 部署前本地验证（强烈建议！）
./ling-test/deploy.sh --dry-run
```

## 工作原理

- `run.sh` 是**单一真相源**——3 个 backend 的差异只有 5 个字段（见脚本顶部 `BACKENDS` 数组）
- 运行时动态生成 YAML、kubectl apply、自动监控直到完成
- 每 backend ~25-35 分钟（权重加载 + 编译 + 3 轮 timed）

## 前置条件

- kubectl 可访问集群
- TPU 节点池 tpu-v7-4-b 和 tpu-v7-4-c 就绪（v4 用 a）
- GCS tarball 已最新（`./ling-test/deploy.sh` 走一遍）

## 完整跑分记录 (2026-06-25)

**ling_v3_flash, bs=1, in=4096, out=2048, tp=4, bf16**

| backend | TTFT | decode/tok | prefill TPS | decode TPS | 说明 |
|---------|------|------------|-------------|------------|------|
| v1 (DEFAULT bf=512) | 1.918s | 18.36ms | 2,135 | 54.5 | 旧 default，bf 不整除 I=768 |
| **v1 (AInfer full-tile)** | **1.005s** | **7.92ms** | **4,077** | **126.2** | bf=768/bd=2560，移植自 AInfer sweep |
| v2 (fused_v2) | 1.068s | 8.78ms | 3,836 | 113.9 | Strix double-buffer EP |
| **v4 (fused_v4 TP)** | 1.108s | **5.66ms** | 3,696 | **176.8** | TP-MoE，decode 冠军 |

### Key Findings

1. **v1 full-tile 配置使得 v1 ≈ v2（甚至略优）**，与 AInfer 上的测试结论一致。
   AInfer 离线 sweep (2026-06-08) 发现对 ling_v3_flash (H=2560, I=768)，
   不切块、整块载入（bf=768=完整 I, bd=2560=完整 H）反而最快——I=768 太小，
   tiling 只有 DMA 往返损失，没有收益。
2. **v4 仍是 decode 冠军**，但 v1 full-tile 把差距从 3.25x 缩小到 1.40x。
3. v1 DEFAULT 的 bf=512/bd=1024 配置对 I=768 完全不合适（浪费 33% padding）。
4. v2 需在 E=512 大 expert 数时关闭 sflag-heavy 特性（`cross_expert_prefetch_mode="none"`）。

### Block Configs

v1 full-tile configs (ported from AInfer `ainfer/kernels/tpu/fused_moe/v1/tuned_block_configs.py`):
```
('bfloat16','bfloat16', 8,    512, 8, 2560, 768, 4, False, True): (2,  768, 2560, 2560, ...)
('bfloat16','bfloat16', 16,   512, 8, 2560, 768, 4, False, True): (4,  768, 2560, 2560, ...)
('bfloat16','bfloat16', 32,   512, 8, 2560, 768, 4, False, True): (8,  768, 2560, 2560, ...)
('bfloat16','bfloat16', 128,  512, 8, 2560, 768, 4, False, True): (32, 768, 2560, 2560, ...)
('bfloat16','bfloat16', 512,  512, 8, 2560, 768, 4, False, True): (32, 768, 2560, 2560, ...)
('bfloat16','bfloat16', 2048, 512, 8, 2560, 768, 4, False, True): (32, 768, 2560, 2560, ...)
```
