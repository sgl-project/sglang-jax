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
- 每 backend ~20-30 分钟（权重加载 + 编译 + 3 轮 timed），全部 3 个 ~60 分钟

## 前置条件

- kubectl 可访问集群
- TPU 节点池 tpu-v7-4-b 和 tpu-v7-4-c 就绪
- GCS tarball 已最新（`./ling-test/deploy.sh` 走一遍）

## 上次完整跑分 (2026-06-23)

| backend   | TTFT   | decode/tok | prefill TPS | decode TPS |
|-----------|--------|-----------|-------------|------------|
| fused (v1)| 1.918s | 18.36ms   | 2,135       | 54.5       |
| fused_v2  | 0.730s | 8.34ms    | 5,613       | 119.9      |
| fused_v4  | 1.105s | 5.59ms    | 3,708       | 178.9      |
