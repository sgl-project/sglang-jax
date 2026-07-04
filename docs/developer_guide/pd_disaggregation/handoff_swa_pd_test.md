# SWA PD 分离 实现与测试 Handoff

**日期**: 2026-07-04
**分支**: `epic/mimo-pd-disggragation` (已 push)
**作者**: jiongxuan + Claude

---

## 1. 代码改动概要

### 目标
让 sglang-jax 的 PD 分离模式支持 SWA (Sliding Window Attention) hybrid 模型（MiMo-V2-Flash）的 KV transfer。

### 核心设计
- raiden (tpu-raiden) 的 `register_read` 会对所有 layer 广播相同的 block_ids。SWA 模型中 full layer 和 SWA layer 使用不同的 page index space，因此需要创建 **2 个 KVCacheManager**（full 用 + SWA 用），分别管理各自 pool 的 kv_caches
- SWA layer 只传输 sliding window 尾部（不全量传输），通过 `full_to_swa_index_mapping` 做 index 翻译
- 非 SWA 模型完全不受影响：`is_hybrid_swa=False` 时走原有的单引擎路径

### 改动文件 (7 files, 6 modified + 1 test)

| 文件 | 改动内容 |
|------|---------|
| `srt/disaggregation/jax_transfer/wrapper.py` | `RaidenTransferWrapper` 新增 `_engine_full` + `_engine_swa` 双 KVCacheManager。`register_read`/`start_read`/`poll_stats` 透明 dispatch 到两个引擎 |
| `srt/disaggregation/prefill.py` | 新增 `_extract_swa_block_ids_for_chunk` — 通过 `full_to_swa_index_mapping` 翻译 index，只提取 sliding window 尾部。`_raiden_handoff_chunk` 同时注册 SWA blocks |
| `srt/disaggregation/decode.py` | `_admit_one_raiden` 中构建 `swa_local_pages` + `swa_remote_endpoint`，通过 PMetadata 传给 receiver |
| `srt/disaggregation/jax_transfer/conn.py` | `PMetadata` 新增 `swa_remote_endpoint`/`swa_local_pages` 字段。`send_chunk`/`producer_register_read`/`_poll_raiden` 透传 SWA 参数 |
| `srt/disaggregation/bootstrap.py` | `RegisterTransferRequest`/`BootstrapClient.register_transfer` 新增 `swa_block_ids`/`swa_raiden_endpoints_json` 字段 |
| `srt/disaggregation/runtime.py` | 检测 `SWAKVPool`（`hasattr(pool, "full_kv_pool")`）时，将 `kv_caches_swa` 传给 raiden wrapper |
| `test/test_pd_swa_basic.py` | CPU 侧单元测试 8 件，覆盖 tail filtering / chunk boundary / 非 SWA 向后兼容 |

### 提交记录
```
8d0c948ec docs(pd/swa): SWA PD 实现与测试 handoff 文档
c68700b2b test(pd/swa): CPU 侧单元测试验证 SWA block extraction 逻辑
34920af14 feat(pd/swa): raiden 双引擎支持 SWA hybrid attention 模型的 KV transfer
```

---

## 2. 验证状态

| 项目 | 环境 | 结果 |
|------|------|------|
| CPU 单元测试 | Mac | ✅ 8/8 通过 |
| 非 SWA 回归测试 (DeepSeek-1.5B) | GKE v6e-1, raiden PD | ✅ GSM8K 0.67, TTFT 正常, OOM 0, `is_hybrid_swa=False` |
| raiden v7x 缓存构建 | Falcon v7x-8 | ✅ 已构建并上传到 GCS |
| MiMo-V2-Flash 模型下载 | Falcon v7x-8 | 🔄 进行中（详见第 3 节） |
| SWA PD prefill-only 测试 | Falcon v7x-8 | ⏳ 模型下载完成后执行 |

---

## 3. 模型下载状态

### MiMo-V2-Flash 模型信息

- **HF 模型 ID**: `XiaomiMiMo/MiMo-V2-Flash`（公开，157 个文件）
- **模型大小**: 约 291GB（256 experts × 48 层 MoE 架构，FP8 量化）
- **关键配置**:
  - `model_type=mimo_v2_flash`，匹配 sglang-jax 的模型实现
  - 48 层：9 Full + 39 SWA（`hybrid_layer_pattern`）
  - `sliding_window_size=128`
  - SWA 层: `swa_num_attention_heads=64, swa_num_key_value_heads=8, swa_head_dim=192`
  - Full 层: `num_attention_heads=64, num_key_value_heads=4, head_dim=192`
  - MoE: `n_routed_experts=256, num_experts_per_tok=8`
  - 支持 `--load-format auto` 直接加载真实权重

### 下载历程与当前状态

| 尝试 | Falcon Job | 方式 | 结果 |
|------|-----------|------|------|
| 1 | `exp-jfuzxv6rr1` | HF → `/tmp/mimo-model` → `gsutil rsync` GCS | pod 在 SIGABRT 后死亡，部分文件已上传 GCS |
| 2 | `exp-yllelfoc4z` | HF 断点续传 → `gsutil rsync` GCS | pod 死亡，GCS 中约 160 个文件 |
| 3 | 待提交 | 纯 CPU job（不碰 TPU） | 计划中 |

**当前 GCS 状态**: 约 160/157 个文件已上传（`gsutil rsync` 增量模式，多次上传导致计数波动），但完整性和总大小待确认。最后一次成功查询显示约 25GB 已下载到本地。

### 断点续传方法

由于 `huggingface_hub.snapshot_download` 支持 `resume_download=True`，且 `gsutil -m rsync` 是增量同步，下载中断后可以直接重新运行以下命令继续：

```bash
# 在 falcon pod 内执行
pip install -q huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('XiaomiMiMo/MiMo-V2-Flash',
    local_dir='/tmp/mimo-model',
    local_dir_use_symlinks=False,
    resume_download=True, max_workers=4)
"
gsutil -m rsync -r /tmp/mimo-model gs://inference-model-storage-poc-tpu-hns/MiMo-V2-Flash/
```

### 遇到的坑

1. **TPU pod 不稳定**: 在 TPU 节点上同时跑模型编译和下载，SIGABRT 会导致 pod 死亡，下载中断
2. **解决方向**: 用纯 CPU job（`device_count: 0`）或仅下载的 job，不启动 TPU 服务器
3. **GCS 跨项目权限**: `model-storage-sglang` (tpu-service-473302) 无法从 falcon (poc-tpu-partner) 访问，必须用 falcon 自己的 bucket `inference-model-storage-poc-tpu-hns`

---

## 4. 服务器/集群状态

### Falcon 集群 (tpuv7x-64-node)

| 属性 | 值 |
|------|-----|
| 集群 ID | `cl-ivvc22wike` |
| 项目 | `poc-tpu-partner` |
| 区域 | `us-central1` |
| GCS Bucket | `gs://inference-model-storage-poc-tpu-hns` |
| v7x-8 配置 | `device_type=v7x, device_count=8, device_topo=2x2x1, replica=1` |
| 可用镜像 | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1` |
| JAX 版本 | 镜像自带 0.9.0，需升级到 0.10.2: `pip install 'jax[tpu]==0.10.2'` |

### GCS Bucket 目录结构

```
gs://inference-model-storage-poc-tpu-hns/
├── raiden-cache/
│   └── raiden-v7x-jax0.10.2.tar.gz    # 27MB, raiden 预编译缓存
├── MiMo-V2-Flash/                      # 模型下载中 (约 160 文件)
│   ├── config.json
│   ├── model-*-of-*.safetensors
│   └── ...
├── MiMo-Audio-7B-Instruct/            # 音频版 MiMo (无 SWA, 不可用)
├── raiden-cache/
│   └── raiden-v7x-jax0.10.2.tar.gz    
└── experiments/
    └── exp-*/
```

### 当前/历史 Falcon 实验

| exp_id | 状态 | 说明 |
|--------|------|------|
| `exp-dgvv0u0olj` | running | Qwen3-8B 测试，SIGABRT 崩溃（非 SWA 模型） |
| `exp-3hyx4untlm` | pending | GCS 跨项目权限错误，卡住 |
| `exp-jfuzxv6rr1` | failed | raiden 构建成功，模型下载中断 |
| `exp-yllelfoc4z` | failed | 模型下载中断 |
| 其他 | failed | pipefail / JAX 版本问题 |

---

## 5. 操作方法

### Falcon 实验管理

```bash
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:/usr/bin:$PATH"

# 查看实验状态
falcon exp get <exp_id> --output json

# 查看日志（最近 30 行）
falcon exp logs <exp_id> --tail 30

# 在 pod 中执行命令
falcon exp exec <exp_id> -- <command>

# 列出最近的实验
falcon exp list --limit 10 --cluster tpuv7x-64-node --output json

# 提交新实验
falcon workflow profile submit -f <manifest.yaml> --output json

# 等待实验完成
falcon workflow profile collect <exp_id> --timeout 60m --output json
```

### GCS 操作

```bash
# 查看 raiden 缓存
gsutil ls gs://inference-model-storage-poc-tpu-hns/raiden-cache/

# 查看 MiMo 模型文件数
gsutil ls gs://inference-model-storage-poc-tpu-hns/MiMo-V2-Flash/ | wc -l

# 查看 MiMo 模型总大小
gsutil du -sh gs://inference-model-storage-poc-tpu-hns/MiMo-V2-Flash/
```

---

## 6. SWA PD 测试计划

### 测试 1: Prefill-Only（1 台 v7x-8）

**目标**: 验证 SWA 代码路径基本功能

**Manifest 关键配置**:
```yaml
# model-path: /models/MiMo-V2-Flash (GCS mount)
# tp-size: 8, dp-size: 1 (PD 模式不支持 DP>1)
# ep-size: 8, moe-backend: fused_v2
# disaggregation-mode: prefill
# disaggregation-use-raiden: true
# swa-full-tokens-ratio: 0.2
# 挂载:
#   gs://.../MiMo-V2-Flash → /models/MiMo-V2-Flash (read-only)
#   gs://.../raiden-cache  → /models/raiden (read-only)
```

**验证项**:
1. 日志出现 `is_hybrid_swa=True, swa_layer_num=39`
2. 日志出现 `RaidenTransferWrapper started ... is_hybrid_swa=True`
3. 日志出现 `RAIDEN-P register_read ... n_swa>0`（SWA blocks 被注册）
4. 无 OOM / 无 Traceback

### 测试 2: Full PD E2E（2 台 v7x-8）

**目标**: prefill + decode 完整 SWA KV transfer

需要 2 台 v7x-8 机器。DP attention 的 PD 适配完成后再执行。

---

## 7. 已知问题与注意事项

1. **DP attention PD 未适配**: `dp_size > 1` 在 PD 模式下不能正常工作。`pd/support_dp` 分支有部分实现，但 E2E 未验证。当前测试使用 `dp_size=1`
2. **跨项目 GCS 权限**: `model-storage-sglang` (tpu-service-473302) 无法从 falcon (poc-tpu-partner) 访问。所有资源放在 `inference-model-storage-poc-tpu-hns`
3. **镜像 JAX 版本**: jax0.9.0-rev1 镜像自带 JAX 0.9.0，需要 `pip install 'jax[tpu]==0.10.2'` 升级
4. **MiMo 模型巨大**: 总重量约 291GB，首次下载需数小时。`resume_download=True` + `gsutil -m rsync` 支持断点续传
5. **raiden 缓存**: v6e 和 v7x 架构不同，需分别编译。v7x 用缓存已上传到 `gs://inference-model-storage-poc-tpu-hns/raiden-cache/`
6. **TPU pod 做下载不稳定**: 建议用纯 CPU job 或先下载模型再启动 TPU 服务器，避免 SIGABRT 中断下载
