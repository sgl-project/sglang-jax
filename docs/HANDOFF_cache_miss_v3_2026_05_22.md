# HANDOFF — cache miss v3 修复 (2026-05-22)

## 一句话状态

`dev/fix-cache-miss-v3` 基于 `origin/dev/spec-info-option-c`，10 commits ahead，prefill cache miss 已 100% 清零，decode cache miss 减 98%（392 → 8），剩 8 trace 待 fix #6 处理；pod `niu-v6e4-sleep` 当前 `UnexpectedAdmissionError` 需重启。

---

## 分支 / Worktree

- **本地 worktree**：`/Users/niu/code/sglang-jax/.worktrees/dev-fix-cache-miss-v3`
- **分支**：`dev/fix-cache-miss-v3` (本地)
- **远端**：尚未 push（HEAD `af5c1b81`）
- **起点**：`origin/dev/spec-info-option-c` @ `ed97c63f`
- **保留 reference**：
  - `dev-fix-cache-miss-v2` (旧实现 7 commits，已废弃)
  - `dev-fix-spec-precompile-dp` (原 v1 ref)
  - `dp-only-test` (detached HEAD diagnose worktree)

## 完成的 commits（HEAD → 老）

```
af5c1b81 fix(spec): pin total_bs to max precompile bucket in _get_spec_decode_mwb_dp
9025d067 fix(spec): pin build_tree_kernel max_context_len to max precompile bucket
8433d21f fix(spec): pad out_cache_loc to bucket in _get_spec_decode_mwb_dp + precompile
b13e0994 refactor(spec): unify dp=1 spec decode path with _get_spec_decode_mwb_dp
cceadc76 Revert "fix(spec): unify dp=1 spec decode path + pad out_cache_loc to bucket"
5ac839b0 fix(spec): unify dp=1 spec decode path + pad out_cache_loc to bucket  ← 有 bug, 被 cceadc76 revert
b9aad0c2 fix(spec): pad logits in draft_extend_for_decode topk path
f20206f7 fix(spec): propagate spec_algorithm in get_model_worker_batch
81f18f23 fix(spec): pad logits in capture_for_decode, gather to real_bs after jit
c1f7d08c fix(spec): pad spec_info_padded.verified_id in draft_extend_for_prefill
```

**有效 commits = 8**（5ac839b0 被 cceadc76 revert）

## 进度数据

| 场景 | baseline | 当前 | 削减 |
|---|---|---|---|
| dp=1 mnt=1 cold（Step 4 验收过） | 64 trace | **0 trace** | -100% |
| dp=1 mnt=20 cold（Step 5） | 392 trace | **8 trace** | -98% |
| dp=1 mnt=20 warm | — | **8 trace** | — |
| dp=2 mnt=1 cold | 42 trace | **0 trace** | -100% |
| dp=2 mnt>1 完整 sweep | 未充分测 | pending（pod 挂了） | — |
| dp=1 first_token 正确性 | — | 50/50 ✓ | — |
| dp=2 first_token 正确性 | — | 24/24 ✓ | — |

## 待做工作

### fix #6 — 固定 `per_req_tokens = draft_token_num`（HTML 计划 Step 5 待做章节有详情）

剩 8 trace 的 root cause：`_get_spec_decode_mwb_dp` 内 `per_req_tokens` 从 `info.out_cache_loc` chunk 长度动态反推，在 mwb 不同状态下结果不同（DECODE 后 alloc=1 vs verify 后 alloc=3），导致 `target_per_rank_ocl` 跳变（16/32/48）。

**修法**：把 `per_req_tokens` 改为 caller 传入的 `draft_token_num`（=3 in our config），不再依赖 chunk 反推。

**3 处改动**：
1. `scheduler.py:1836` caller 传 `draft_token_num=self.draft_worker.speculative_num_draft_tokens`
2. `schedule_batch.py:get_spec_model_worker_batch` 加 `draft_token_num: int = 1` 参数，forward 给 `_get_spec_decode_mwb_dp`
3. `schedule_batch.py:_get_spec_decode_mwb_dp` 内 `target_per_rank_ocl = max(per_dp_bs * draft_token_num, max_chunk_len)`（max_chunk_len 防御保留）

**预期结果**：dp=1 mnt=20 cold 8 → ≤2 trace。dp=2 mnt=1 维持 0 trace。

### dp=2 mnt>1 完整 sweep

pod 恢复后跑 `--max-new-tokens 20`（sweep 脚本 timeout 已改 1800s）。

dp=2 + 多 token 性能本身慢（option C base 留下，跟 cache miss 无关），但 timeout 拉大后应能跑完。验证 cache miss 数 + correctness。

---

## Pod 状态

```
NAME             READY   STATUS                     RESTARTS   AGE
niu-v6e4-sleep   0/3     UnexpectedAdmissionError   0          3d1h
```

**需要操作**：
- `kubectl delete pod niu-v6e4-sleep` + 重新部署
- 或换其他 v6e-4 pod（pod 上需复制 `/tmp/sglang-jax-baseline/` + `/tmp/start_dev_dp{1,2}.sh` + `/tmp/baseline_sweep.py`）

恢复后操作：
```bash
# 1. 推 v3 代码
cd /Users/niu/code/sglang-jax/.worktrees/dev-fix-cache-miss-v3
tar czf /tmp/v3_python.tgz python/
kubectl cp /tmp/v3_python.tgz <pod>:/tmp/v3_python.tgz
kubectl exec <pod> -- bash -c 'cd /tmp/sglang-jax-baseline && rm -rf python && tar xzf /tmp/v3_python.tgz'

# 2. 推 sweep 脚本（已加 1800s timeout + --max-new-tokens arg）
kubectl cp scripts/baseline_sweep.py <pod>:/tmp/baseline_sweep.py
```

---

## 关键文件路径

| 类型 | 路径 |
|---|---|
| 计划 HTML | `docs/plan_cache_miss_v3_2026_05_21.html` |
| 此交接 doc | `docs/HANDOFF_cache_miss_v3_2026_05_22.md`（本文件） |
| Baseline 报告 | `docs/baseline_cache_miss_v3.md` |
| Baseline 数据 | `baseline_data/baseline_dp{1,2}.jsonl`（74 cell mnt=1 数据）|
| Sweep 脚本 | `scripts/baseline_sweep.py`（含 `--dp` / `--out` / `--max-new-tokens` / 1800s timeout） |
| 启动脚本（pod 上） | `/tmp/start_dev_dp1.sh` / `/tmp/start_dev_dp2.sh` |
| Server log（pod 上） | `/tmp/test_dp1.log` / `/tmp/test_dp2.log` |
| jax disk cache（pod 上） | `/tmp/jax_cache/` |

## 测试模型与 config

- model: `/models/meta-llama/Llama-3.1-8B-Instruct`
- draft: `/models/unkmaster/EAGLE3-LLaMA3.1-Instruct-8B`
- spec: `--speculative-algorithm EAGLE3 --speculative-eagle-topk 1 --speculative-num-steps 2 --speculative-num-draft-tokens 3`
- 并行: `--tp-size 4 --dp-size {1,2}`
- 内存: `--mem-fraction-static {0.7 dp=1, 0.6 dp=2} --context-length 4096 --max-prefill-tokens 4096 --chunked-prefill-size -1 --max-running-requests 16`
- precompile bucket: `--precompile-bs-paddings 1 4 8 16 --precompile-token-paddings 256 1024 2048 4096`
- 其他: `--page-size 64 --attention-backend fa --dtype bfloat16`

## 测试矩阵（baseline_sweep.py 内 DP1_CELLS / DP2_CELLS）

- seqlen ∈ {128, 256, 500, 1024, 1500, 2048, 3000, 4096}（桶边界 + 桶内中位）
- bsz × seqlen ≤ 4096 约束
- dp=1: 50 cells（bsz 1-16 全跑）
- dp=2: 24 cells（bsz 仅偶数 2/4/6/8/10/12/14/16）

---

## 验收标准（详见 HTML）

| 场景 | 目标 | 当前 |
|---|---|---|
| dp=1 mnt=1 cold runtime trace | 0 | ✅ 0 |
| dp=1 mnt=20 cold runtime trace（fix #6 后） | 0 | ⏳ 8 (待 fix #6) |
| dp=1 mnt=20 warm | 0 / 0 | ⏳ 8 trace（同 cold） |
| dp=2 mnt=1 cold | 0 trace | ✅ 0 |
| dp=2 mnt>1 完整 sweep | acceptable | ⏳ pending pod |
| 正确性 | 100% | ✅ 74/74 dp=1 + 24/24 dp=2 |

**兜底**：若 fix #6 之后仍剩 N trace（理论 cold compile minimum），acceptable ≤ 2（spec extend EXTEND × 2 worker）。

---

## 风险 / 已知问题

1. **dp=2 多 token 慢**：单 cell ~分钟级，option C base 性能问题，**非 cache miss 修复范围**。sweep timeout 已加到 1800s，应能跑完。
2. **fix #6 dp=2 风险**：max_chunk_len 兜底防御保留，但需 dp=2 sweep 验证不破。
3. **rebase 风险**：option C 分支即将 merge 进 epic（HTML 计划"分支生命周期" 节有详情）。fix #6 commit 后 push branch 等 review，merge 后 rebase 应无冲突（option C 改动跟 v3 重叠区域少）。
4. **未 push 到远端**：当前 8 commits 仅本地。push 前可视情况决定要不要 squash（5ac839b0 + cceadc76 revert 对可考虑 squash 简化历史）。

---

## 给新 Claude 线程的推荐启动 prompt

> 继续 cache miss 修复 v3 任务，详读 `docs/HANDOFF_cache_miss_v3_2026_05_22.md` 全文。当前进度：
> - 8 个 commits 在 `dev/fix-cache-miss-v3` (HEAD `af5c1b81`)
> - dp=1 mnt=1 / dp=2 mnt=1 cache miss 已 0
> - dp=1 mnt=20 剩 8 trace 待 fix #6（HTML 计划 Step 5 章节有详情 + 3 处具体改动）
> - pod `niu-v6e4-sleep` 当前 `UnexpectedAdmissionError`，需重启或换 pod
> - dp=2 mnt>1 完整 sweep 待跑（sweep timeout 已加到 1800s）
>
> 第一步：等 pod 恢复 → 实施 fix #6 → cold start dp=1 mnt=20 sweep 验证 → cold start dp=2 mnt=20 sweep（拉长 timeout）→ 验收 + push 分支。
