# TPU Benchmark 部署操作手册

> 本手册基于 2026-06-24~25 session 中 ~15 次 TPU 部署踩坑经验编写。
> **先读这个再部署，可避免 90% 的错误**。

## 1. 部署前必做（15 秒本地验证）

```bash
cd /Users/wenqiao/gcp/sglang-jax
./ling-test/deploy.sh --dry-run
```

这会检查：
- 关键 .py 文件语法正确
- tarball 结构（`tar tzf` 顶层 = `sglang-jax/`）
- tarball 包含所有必要文件
- 远端 GCS tarball 与本地一致

**规则：dry-run 不过，不上 TPU。违反此规则者必等 20 分钟才发现低级错误。**

## 2. 一键跑 benchmark

```bash
cd /Users/wenqiao/gcp/sglang-jax
./ling-test/longctx-bench/run.sh          # 全跑 (v1 → v2 → v4)
./ling-test/longctx-bench/run.sh v1       # 只跑 v1
./ling-test/longctx-bench/run.sh v2 v4    # 只跑 v2+v4
```

## 3. 代码修改后如何更新 tarball

```bash
# 方式 1: deploy.sh 全自动
./ling-test/deploy.sh --dry-run    # 先干跑
./ling-test/deploy.sh              # 完整流程 (打包+上传+部署+监控)

# 方式 2: 手动 (deploy.sh 网络不通时)
parent_dir="$(dirname /Users/wenqiao/gcp/sglang-jax)"
repo_name="$(basename /Users/wenqiao/gcp/sglang-jax)"
tar czf /tmp/sglang-jax-v4.tar.gz -C "$parent_dir" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    "$repo_name/"
gsutil cp /tmp/sglang-jax-v4.tar.gz gs://ainfer-tpu-bench-code/sglang-jax-v4.tar.gz

# 验证远端
curl -fsSL "https://storage.googleapis.com/ainfer-tpu-bench-code/sglang-jax-v4.tar.gz" \
    -o /tmp/check.tar.gz && tar tzf /tmp/check.tar.gz | grep "关键文件路径"
```

## 4. 常见踩坑 + 解决

### 4.1 Pod 报 `ModuleNotFoundError: No module named 'fla'`
> **根因**：ling_v3_flash checkpoint 的 modeling 代码 import `fla.ops.simple_gla`。
> **修复**：v2 YAML 需要 `pipi fla-core triton`（已内置在 run.sh 的 BACKENDS 定义中）。

### 4.2 Pod 报 `cd /app/sglang-jax/python: No such file`
> **根因**：tarball 顶层是 `sglang-jax/`，解压到 `/app/sglang-jax/` 后真实路径是
> `/app/sglang-jax/sglang-jax/python`。若 YAML 写 `cd /app/sglang-jax/python` 则多一层。
> **修复**：YAML 中统一用 `cd /app/sglang-jax/sglang-jax/python`（run.sh 已内置）。

### 4.3 v2 报 `local_num_tokens=0 > 0` (bs=1 单流 decode)
> **根因**：`FusedEPMoEV2.__call__` 缺 bs=1 padding 补丁。
> `get_ep_size(mesh) * get_dtype_packing(dtype)` 算出 token_multiple=8，
> bs=1 被 padding 到 8，否则 `num_tokens/ep_size = 1/4 = 0`。
> **修复**：已在 `fused_moe.py` 中实现（`if pad: hidden_states = jnp.concatenate(...)`）。

### 4.4 v2 编译时 smem OOM (`Ran out of memory in memory space smem`)
> **根因**：ling_v3_flash 有 E=512, ep=4 → local_num_experts=128，
> v2 的 sflag 预算（16KB）超限。
> **修复**：已在 `fused_moe.py` FusedEPMoEV2.__call__ 中设：
> ```python
> cross_expert_prefetch_mode=("none" if self.num_experts // self.ep_size >= 64 else "full"),
> enable_bt_scatter_overlap=(False if self.num_experts // self.ep_size >= 64 else True),
> interleave_bt=(False if self.num_experts // self.ep_size >= 64 else True),
> ```

### 4.5 v1 性能差（decode>17ms）
> **根因**：sglang-jax v1 没有 ling_v3_flash 的 tuned block config，
> 退化到 DEFAULT `bf=512`（I=768 浪费 33% padding）。
> **修复**：已从 AInfer 移植 full-tile config (bf=768, bd=2560)。
> 位于 `python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py`。

### 4.6 tarball 和 Pod 里跑的代码不一致（最隐蔽！）
> **根因**：本地改了代码但忘记重打包上传；或 HTTP CDN 缓存了旧版。
> **检测方式**：
> 1. deploy.sh --dry-run 的 Phase 2 会验证远端
> 2. .GIT_COMMIT 文件已打包装入 tarball
> 3. Pod 启动日志里 run.sh 会打印 `commit: 20XXXXXXXX-XXXXXX-<hash>`
> **修复**：如果 commit hash 不对，重新 `./ling-test/deploy.sh`。

### 4.7 GitHub push 被 secret-scanning 拦 (HF token)
> **根因**：父 commit `0412eb9e` 的 YAML 里有 HF token。
> **绕过**：去 GitHub 给的 URL 一键允许，或 squash-rebase 到干净 commit。
> 注意：允许只对此 branch 有效，切换到其他 base 后可能再出现。

## 5. AInfer vs tpu-inference 代码库区分

> **这是一个在 session 中导致 2h+ 无效分析的认知错误。**

| 代码库 | 路径 | 是什么 |
|--------|------|--------|
| **AInfer**（蚂蚁内部推理） | `/Users/wenqiao/codefuse/AInfer/` | 蚂蚁自研，v1 tuned 165 条含 ling_v3_flash |
| **tpu-inference**（Google vLLM TPU）| `/Users/wenqiao/gcp/tpu-inference/` | Google 开源原型，v1 tuned 30 条，无 ling_v3 |
| **sglang-jax**（本次目标） | `/Users/wenqiao/gcp/sglang-jax/` | 继承 tpu-inference，我们加 v4 + 搬 AInfer tuning |

**规则：对比 tuned config 时看 AInfer，不看 tpu-inference。**

## 6. 网络恢复提示

如果 kubectl/gsutil 报 OAuth 超时：
```bash
curl -s --connect-timeout 5 https://oauth2.googleapis.com/ > /dev/null || echo "网络不通"
```
网络恢复优先验证 `kubectl get pods`，再继续部署。
