# Qwen3 MoE / MiMo-V2-Flash TPU Fused 调试复现与现状（v6e-8）

本文记录当前在 TPU (`sky-900d-jiongxuan`) 上用于复现与对照的脚本、模型下载方式、现状结论，以及下一步工作项。

## 1. 环境与机器信息

- 机器：`sky-900d-jiongxuan`
- 仓库：`~/sky_workdir/sgl-jax`
- 环境激活：

```bash
cd ~/sky_workdir/sgl-jax
source .venv/bin/activate
```

- 当前 TPU（实测）：
  - `TPU v6 lite`
  - 每卡 HBM `bytes_limit = 33,550,233,600`（约 `31.25 GiB`）

## 2. 模型权重下载（MiMo-V2-Flash 到 /dev/shm）

目标：把 `XiaomiMiMo/MiMo-V2-Flash` 下载到 `/dev/shm`（避免慢盘 I/O，并为后续多次实验复用）。

### 2.1 检查空间

```bash
df -h /dev/shm
```

### 2.2 后台下载命令（示例）

```bash
cd ~/sky_workdir/sgl-jax
source .venv/bin/activate

nohup python - <<'PY' > ~/sky_workdir/mimo_v2_flash_hf_download.log 2>&1 &
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="XiaomiMiMo/MiMo-V2-Flash",
    local_dir="/dev/shm/MiMo-V2-Flash",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("DONE")
PY
echo $!
```

### 2.3 进度查看

```bash
tail -f ~/sky_workdir/mimo_v2_flash_hf_download.log
du -sh /dev/shm/MiMo-V2-Flash
```

### 2.4 当前状态

- 已完成下载（`157/157`）
- 路径：`/dev/shm/MiMo-V2-Flash`

## 3. Qwen3 MoE 复现脚本（tp=8, ep=8）

用途：作为 fused MoE 对照基线。已验证在同一台 `v6e-8` 上 `epmoe` 和 `fused` 都可成功返回。

### 3.1 Qwen3 + EPMoE（成功）

远端脚本：

- `/tmp/run_qwen3_epmoe_singleproc_t1_tp8_ep8.sh`

关键点：

- `--tp-size 8`
- `--ep-size 8`
- `--moe-backend epmoe`
- 单进程（便于快速调试）

成功日志：

- `~/sky_workdir/server_qwen3_epmoe_single_30128.log`

结果：

- `POST /v1/chat/completions` 返回 `200 OK`

### 3.2 Qwen3 + FusedEPMoE（成功）

远端脚本：

- `/tmp/run_qwen3_fused_singleproc_t1_tp8_ep8_bc256.sh`

关键点：

- `--tp-size 8`
- `--ep-size 8`
- `--moe-backend fused`
- 使用上层 debug block-config 覆盖（适配 `intermediate_size=768` 的 fused 校验）
  - `bf/bfc/bse=256`（通过环境变量控制，未改 kernel）

成功日志：

- `~/sky_workdir/server_qwen3_fused_single_30129.log`

结果：

- `POST /v1/chat/completions` 返回 `200 OK`

## 4. MiMo-V2-Flash 复现脚本（tp=8, ep=8, fused）

用途：在正确 `256 experts` 配置下，复现 MiMo fused 路径问题。

### 4.1 标准 keepalive 脚本（/dev/shm 模型）

远端脚本：

- `/tmp/run_mimo_fused_tp8_ep8_shm_keepalive.sh`

关键点：

- 模型路径：`/dev/shm/MiMo-V2-Flash`
- `--tp-size 8`
- `--ep-size 8`
- `--moe-backend fused`
- 开启调试审计：
  - `SGL_FUSED_MOE_DEBUG_LOG_PARAM_SHARDING=1`
  - `SGL_FUSED_MOE_DEBUG_LOG_CALL=1`
  - `SGL_FUSED_MOE_DEBUG_LOG_SHARDING=1`
- 使用上层 block-config 覆盖（未改 kernel）：
  - `BT/BTS/BTC=32`
  - `BF/BFC/BSE=256`

### 4.2 低缓存版本（尝试降低 HBM 占用）

远端脚本：

- `/tmp/run_mimo_fused_tp8_ep8_shm_keepalive_mem005.sh`

相对 4.1 的差异：

- `--mem-fraction-static 0.05`
- `--context-length 1024`

目的：

- 验证是否只是 KV cache/静态缓存导致的加载 OOM

## 5. 当前关键代码改动（与复现强相关）

### 5.1 MiMo 专家数字段修复（关键）

文件：

- `python/sgl_jax/srt/models/mimo_v2_flash.py`

问题：

- MiMo 配置实际使用 `n_routed_experts=256`
- 代码此前读取 `num_experts`，会错误 fallback 到 `8`

修复：

- 新增 `_get_mimo_num_experts(config)`，优先读 `num_experts`，否则读 `n_routed_experts`
- 替换 `MiMoV2Moe` 构造与 MoE weight mapping 路径中的专家数读取

影响：

- 修复后 MiMo 确实按 `256 experts` 加载（日志已确认）

### 5.2 Qwen3 / Qwen3_MoE 的 TP>KV 适配（tp=8, kv=4）

文件：

- `python/sgl_jax/srt/models/qwen3.py`
- `python/sgl_jax/srt/models/qwen3_moe.py`

内容：

- `reshape` 改为 `jax.lax.reshape(..., out_sharding=...)`
- `k/v` 在运行时做 head repeat（`4 -> 8`）

目的：

- 让 `Qwen3` 在 `tp=8` 下能稳定走 attention/fused 对照路径

### 5.3 FusedEPMoE 静态量化初始化内存优化（已尝试）

文件：

- `python/sgl_jax/srt/layers/moe.py`

内容（上层，不改 kernel）：

- 对静态量化 checkpoint：
  - 初始化专家权重用 `quantized_dtype`（如 fp8）而不是 `bf16`
  - 使用 `zeros` 占位，避免大张量随机初始化的额外开销

目的：

- 降低模型加载阶段 HBM 峰值

结果：

- 未解决当前 MiMo `v6e-8` 的加载 OOM（见下一节）

## 6. 当前现状（最重要）

### 6.1 Qwen3（同机同 `tp=8, ep=8`）

- `epmoe`：成功（`200 OK`）
- `fused`：成功（`200 OK`）

结论：

- 同一台 TPU 上 fused MoE 能跑通
- MiMo 的问题是 MiMo 特有路径（量化/权重规模/加载策略/配置映射），不是 fused 框架普遍不可用

### 6.2 MiMo（修正为 `256 experts` 后）

MiMo 目前尚未进入 fused runtime 调试阶段，先在权重加载阶段 OOM。

错误（远端日志）：

- `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED`
- 申请 `256.00M` 设备 buffer 失败，剩余约 `205.90M`

典型日志文件：

- `~/sky_workdir/server_mimo_fused_tp8_ep8_30132.log`

失败位置：

- `python/sgl_jax/srt/utils/weight_utils.py:714`
- `_create_stacked_moe_lazy_tensor(...)`
- `jax.make_array_from_callback(...).astype(...)`

### 6.3 这是“容量不足”还是“峰值过高”？

结论：当前更像是 **最终常驻容量就不够**，不只是加载峰值问题。

原因（按 `tp=8, ep=8`，MiMo `256 experts` 推算）：

- 每卡本地专家数：`256 / 8 = 32`
- 每层本地 MoE（仅 `w1/w2/w3 + scales`）约：
  - weights（fp8）：`~0.75 GiB`
  - scales（fp32）：`~0.023 GiB`
- 47 个 MoE 层总计约：
  - **`~36.35 GiB / card`**

而 `TPU v6e-8` 实测每卡仅约 `31.25 GiB` HBM。

因此：

- 即使压 `mem-fraction-static` 和 `context-length`
- 以及减少静态初始化峰值
- 仍会在加载后期 OOM（当前稳定复现）

## 7. 下一步工作项（按优先级）

### 7.1 更换更大显存 TPU（必须）

优先建议：

1. `tpu-v5p-8`（推荐）
2. `tpu-v7x-8`（如果账号/配额可用）
3. 或继续 v6e，但至少升到更多芯片并提高 `ep_size`（如 `v6e-16` + `ep_size=16`）

说明：

- `v6e-8` 在正确 `256 experts` 下不够放
- 若升到更多芯片，运行参数要同步调整 `--ep-size`

### 7.2 在新机器上第一步要做的事

1. 先同步当前代码（至少以下文件）
   - `python/sgl_jax/srt/models/mimo_v2_flash.py`
   - `python/sgl_jax/srt/layers/moe.py`
   - `python/sgl_jax/srt/models/qwen3.py`
   - `python/sgl_jax/srt/models/qwen3_moe.py`
2. 下载 MiMo 权重到本地高速盘（可继续用 `/dev/shm`）
3. 先跑 `Qwen3 fused` 确认 TPU 环境和 fused baseline 正常
4. 再跑 MiMo fused（`tp/ep` 按新设备容量重新配置）

### 7.3 MiMo 在更大设备上恢复的调试顺序

1. 确认 `FUSED_MOE_PARAM_AUDIT` 中 `num_experts=256`
2. 确认 MoE scale 形状仍为：
   - `w1/w3_scale`: `(256, 32, 1, 2048)`（加载后）
   - `w2_scale`: `(256, 16, 1, 4096)`（加载后）
3. 继续跑到 fused runtime，观察是否回到 `TensorCoreSequencer` 崩溃
4. 再对比 `Qwen3 fused` 与 `MiMo fused` 的运行时入参 / block-config / sharding

## 8. 参考日志文件（当前机器）

- Qwen3 EPMoE 成功：`~/sky_workdir/server_qwen3_epmoe_single_30128.log`
- Qwen3 Fused 成功：`~/sky_workdir/server_qwen3_fused_single_30129.log`
- MiMo Fused（/dev/shm，加载 OOM）：`~/sky_workdir/server_mimo_fused_tp8_ep8_30132.log`
- MiMo 下载日志：`~/sky_workdir/mimo_v2_flash_hf_download.log`

## 9. Qwen3 MoE Fused（Master 基线）在 TPU v5p-4 的最新结论（2026-02-26）

机器与配置：

- 机器：`sky-02da-jiongxuan`（单机可见 `TPU v5 x4`）
- 模型：`/models/Qwen3-30B-A3B`
- 并行：`tp=4, ep=4`
- 目标：`--moe-backend fused`

### 9.1 先决修复（主线必须）

未修改 master 直接跑 `Qwen3 fused`，至少会先撞两个问题：

1. `FusedEPMoE.__call__` 不接受 `token_valid_mask`
2. `moe_intermediate_size=768` 与默认 fused block config 的 `bf=512` 不对齐

已在主线代码路径上做的最小修复（不改 kernel）：

- `python/sgl_jax/srt/layers/moe.py`
  - `FusedEPMoE.__call__` 兼容 `token_valid_mask` 参数（先忽略）
- `python/sgl_jax/srt/models/qwen3_moe.py`
  - 模型层为 `Qwen3 MoE` 注入兼容的 `FusedMoEBlockConfig`（`bf/bfc/bse=256`）
  - 新增环境变量覆盖入口（便于 sweep，不改 kernel）

### 9.2 成功跑通（已验证）的参数组

在 `main` 基线（含上述最小修复）上，以下参数组合可让 `Qwen3 + fused + fa` 在 **禁用 overlap** 时返回 `200 OK`：

- 启动脚本：`/tmp/run_qwen3_main_fused_tp4_t1_nooverlap.sh`
- 必须环境变量：

```bash
export PYTHONPATH=~/sky_workdir/sgl-jax-main/python

export SGL_QWEN3_FUSED_BC_BT=8
export SGL_QWEN3_FUSED_BC_BTS=8
export SGL_QWEN3_FUSED_BC_BTC=8
export SGL_QWEN3_FUSED_BC_BF=256
export SGL_QWEN3_FUSED_BC_BFC=256
export SGL_QWEN3_FUSED_BC_BSE=256
export SGL_QWEN3_FUSED_BC_BD1=2048
export SGL_QWEN3_FUSED_BC_BD1C=2048
export SGL_QWEN3_FUSED_BC_BD2=2048
export SGL_QWEN3_FUSED_BC_BD2C=2048
```

对应日志确认（示例）：

- `Using Qwen3 fused MoE compatible block config ... {'bt': 8, 'bts': 8, 'bf': 256, 'bd1': 2048, 'bd2': 2048, 'btc': 8, 'bfc': 256, 'bd1c': 2048, 'bd2c': 2048, 'bse': 256}`
- `POST /v1/chat/completions HTTP/1.1" 200 OK`

### 9.3 关键参数扫描结论（当前范围）

在 `tp=4, ep=4, max_tokens=1, no-overlap` 下对多组 block config 做了 sweep（模型层 env 覆盖）：

- `bt=8, bts=8, btc=8` + `bd*=2048` + `bf/bfc/bse=256`：**成功**
- `bt=16`（无论 `bd*=1024` 或 `2048`）：失败（`Anomalies`）
- `bt=32` + `bd*=2048`：失败（`FAILED_PRECONDITION`）
- `bt=8` 但 `bd*=1024`：失败（`Anomalies`）
- 仅调整 `bf`（如 `128` / `384`）而保持其余基线：失败（`Anomalies`）

当前可得结论：

- 成功并非单一参数（如只改 `bf` 或只改 `bt`）即可触发
- 在已测范围内，至少需要同时满足：
  - `bt/bts/btc = 8`
  - `bd1/bd1c/bd2/bd2c = 2048`
  - `bf/bfc/bse = 256`

### 9.4 是否“完整在 master 上跑通”？

结论需要分情况说明：

- `Qwen3 MoE + fused` 在 **未修改 master** 上：**不能完整跑通**
  - 先会卡在 `token_valid_mask` 签名 / `bf=512` 对齐问题

- `Qwen3 MoE + fused` 在 **master 基础 + 上述最小修复** 上：
  - `fa + no-overlap`：使用成功参数组时，**可以完整返回 `200 OK`**
  - `fa + 默认 overlap`：**当前仍失败**（`Anomalies` / `Empty reply`）

因此当前阶段更准确的说法是：

- **主线（加最小修复）已能在 TPU v5p-4 上跑通 Qwen3 MoE fused 的 no-overlap 场景**
- **默认 overlap 场景还未跑通，仍需继续调试**
