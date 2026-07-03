# HANDOFF — PD 分离 raiden 数据面 (W2) e2e 验证

> 交接文档。记录当前进度、集群登录/部署/测试命令、结果与已知问题。
> 最后更新:2026-07-03。分支 `epic/mimo-pd-disggragation`。

---

## 1. 现状(一句话)

mimo PD 分离的 **tpu-raiden 数据面 (W2) 已 e2e 跑通并验证**:功能正确(GSM8K 0.67)、
传输延迟相对旧 path-A 有明显收益(kv_wait 快约 3×)、中等并发(conc 16/32)稳定无崩溃。
W1(JAX 0.10.2 升级)此前已完成。

- **模型/硬件**:DeepSeek-R1-Distill-Qwen-1.5B,TP=1,GKE `pd-v6e-1` 双单芯 v6e(pod0=prefill+bootstrap,pod1=decode+router+driver)。
- **传输引擎**:tpu-raiden `KVCacheManager`(PULL,块级设备直传),`--disaggregation-use-raiden`。

### 关键 commit(本轮 raiden 修复,已在分支上)
- `88524024f` fix: max_blocks 加页余量(满长度 prompt 溢出到第 33 页)
- `430ba6e87` fix: 用 max_blocks/num_slots 构造 KVCacheManager(避开 legacy ctor 跳过 slot pool)
- `b460a6270` fix: 单 sub-manager start_read 用纯 host:port 字符串
- `c86178183` W2: raiden data plane for PD disaggregation

### 未提交的本地改动(重要)
- `scripts/disaggregation/gke/pd_singlehost_eval_job.yaml`(`git status` 显示 M):driver 已从
  conc64 洪泛 smoke 改成 **conc=1 串行 TTFT 扫描**(对齐 `pd_ttft_profiling_report.md` 方法学)。
  `deploy.sh` 渲染的是**本地** yaml,所以改动无需 push 即生效;但尚未 commit,注意别被 `git checkout` 冲掉。

---

## 2. 集群登录

```bash
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
gcloud container clusters get-credentials ainfer-tpu-bench \
  --zone asia-northeast1-b --project tpu-service-473302
```

查看/进入当前 pod(pod 名每次重部署会变,先 get 再用):
```bash
kubectl get pods | grep pd-singlehost-eval
# 例:pd-singlehost-eval-0-xxxxx (prefill) / pd-singlehost-eval-1-xxxxx (decode+driver)
PREFILL=$(kubectl get pods -o name | grep pd-singlehost-eval-0 | cut -d/ -f2)
DECODE=$(kubectl get pods -o name | grep pd-singlehost-eval-1 | cut -d/ -f2)

kubectl logs $DECODE  -c jax-tpu -f      # driver + decode server 日志
kubectl logs $PREFILL -c jax-tpu -f      # prefill server 日志
kubectl exec -it $DECODE -c jax-tpu -- bash   # 进 decode pod
```

**当前存活 pod(2026-07-03,可能已被清理)**:prefill `pd-singlehost-eval-0-mkt8g`,decode `pd-singlehost-eval-1-5p59s`。两者跑完 driver 后处于 `sleep 3600` bounded 保活,方便拉日志/手动跑 bench。

---

## 3. 部署 / 销毁

部署(会先删旧 job 再从本地 yaml 渲染应用;raiden/model 有 /models gcsfuse 缓存,setup ~几分钟):
```bash
GH_ORG=primatrix BRANCH=epic/mimo-pd-disggragation \
  bash scripts/disaggregation/gke/deploy.sh
```

销毁释放两个 v6e 节点:
```bash
bash scripts/disaggregation/gke/deploy.sh --delete-only
```

> 说明:服务端 flag 在 yaml 里,改 conn.py/decode.py/wrapper.py/runtime.py 等**代码**需要 `git push`
> (pod 会 clone 分支);只改 yaml 的 driver/env 直接重跑 deploy.sh 即可(渲染本地 yaml)。

---

## 4. 测试命令

### 4.1 自动(driver 内置,重部署即跑)
decode pod 的 driver 依次跑:
1. **GSM8K 正确性**:200 题 / parallel 16 / max-new-tokens 512。
2. **conc=1 串行 TTFT 扫描**:input 512/1024/2048/4096,out=1,warmup3+measure10,`--random-range-ratio 1.0`。

结果直接在 decode pod 日志里:
```bash
kubectl logs $DECODE -c jax-tpu | grep -A2 "GSM8K summary"
kubectl logs $DECODE -c jax-tpu | grep -E '^\{"tag"'        # 每档 TTFT jsonl
kubectl exec $DECODE -c jax-tpu -- cat /tmp/gsm8k_result.jsonl
```

### 4.2 手动吞吐/稳定性(对着现有 serve 跑,不用重部署)
在 decode pod 内对 router(localhost:30000)发压。**注意别用 conc64 × 大 payload**,decode 无准入背压会 OOM 崩(见 §6)。安全档:conc≤32、in=2048、out=256。
```bash
kubectl exec $DECODE -c jax-tpu -- bash -lc '
cd /sglang-jax && source /tmp/tpu_logs/venv/bin/activate
python -m sgl_jax.bench_serving --backend sgl-jax \
  --base-url http://localhost:30000 --model /models/DeepSeek-R1-Distill-Qwen-1.5B \
  --dataset-name random --random-input-len 2048 --random-output-len 256 \
  --random-range-ratio 1.0 --num-prompts 96 --max-concurrency 32 \
  --warmup-requests 4 --output-file /tmp/bench_thr.jsonl'
```

### 4.3 排障常用
```bash
# 服务端崩溃/OOM/raiden 失败标记
kubectl exec $DECODE  -c jax-tpu -- grep -cE 'out of memory|sigquit|Scheduler hit an exception|failed_recving' /tmp/decode_server.log
kubectl exec $PREFILL -c jax-tpu -- grep -cE 'out of memory|sigquit|Scheduler hit an exception' /tmp/prefill_server.log
# raiden 传输净耗时 / 峰值 KV 占用
kubectl exec $DECODE -c jax-tpu -- bash -lc "grep -oE 'kv_wait=[0-9.]+ms' /tmp/decode_server.log | tail -20"
kubectl exec $DECODE -c jax-tpu -- bash -lc "grep -oE 'token usage: [0-9.]+' /tmp/decode_server.log | sort -t: -k2 -n | tail -1"
```

---

## 5. 结果摘要(2026-07-03)

**正确性**:GSM8K accuracy = **0.67** / 200 题(≈ path-A baseline 0.675),TTFT 扫描 4 档 13/13 全成功、0 崩溃。

**conc=1 client TTFT(ms,中位)** vs path-A baseline(`pd_ttft_results.md`):

| input | no-PD | path-A unstaged | path-A staged | raiden |
|------:|------:|----------------:|--------------:|-------:|
| 512   | 16.4  | 69.9  | 84.6  | 44.0  |
| 1024  | 25.2  | 79.9  | 146.9 | 62.8  |
| 2048  | 46.0  | 101.1 | 183.9 | 111.0 |
| 4096  | 94.2  | 182.5 | 300.7 | 201.2 |

**kv_wait(纯传输净耗时,ms)**:raiden ~15/17/27/41,path-A unstaged 48.8/56.5/74.0/144.2 →
**raiden 快约 3×**。端到端 TTFT:小 prompt(≤1024)raiden 快 20~37%;大 prompt(2048/4096)与
unstaged 持平(受 raiden decode 侧 `prealloc_wait` ~18/38/39/75ms 拖累,正交可单独优化);全面优于 D2H staged。

**中等并发稳定性**(手动 bench,in=2048/out=256):

| 并发 | 请求 | 成功 | req/s | 总吞吐 tok/s | 中位 TTFT | raiden 失败 |
|-----:|-----:|-----:|------:|------------:|----------:|:----------:|
| 16   | 64   | 64/64 | 7.5  | 17.3k | 540ms | 0 |
| 32   | 96   | 96/96 | 10.5 | 24.2k | 836ms | 0 |

两轮服务端 0 OOM/abort/`failed_recving`,decode 峰值 token usage 0.93。

---

## 6. 已知问题 / 注意事项

- **decode 无准入背压**:conc64 × (4096 in + 1024 out) 会把 decode KV pool 灌到 token usage 1.00 →
  `RuntimeError: Prefill out of memory` → sigquit 调度器崩溃 → 在途请求 HTTP 流被切(客户端
  `ClientPayloadError`)。**这是过载问题,与 raiden 无关**(path-A 同配置也崩)。压测请控制并发×payload。
- **PD 分离不支持 chunk prefill**:prompt > `chunked_prefill_size` 被 `validate_pd_no_chunked_prefill`
  直接 abort。跑长 prompt 需调高 `--chunked-prefill-size`(当前 yaml 设 16384)。
- **raiden staging slot 尺寸**:`max_blocks = ceil(max_seq_len/page_size) + 8`、`num_slots ≥ 16`
  (代码在 `runtime.py` ~155-172、`wrapper.py::RaidenTransferWrapper.start`)。**别再用
  `host_blocks_to_allocate` 旧 ctor**(会跳过 slot pool 初始化,max_blocks_=0,start_read 立即 failed)。
- **吞吐无 baseline**:`pd_ttft_results.md` 只有 conc=1 TTFT,没有吞吐 QPS baseline。§5 的吞吐数只用于看稳定性,不作对比结论。

---

## 7. 关键代码 / 文档

- `python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py` — `RaidenTransferWrapper`(start/register_read/start_read/poll_stats)
- `python/sgl_jax/srt/disaggregation/runtime.py` — raiden 接线 + max_blocks/num_slots 计算(~148-180)
- `python/sgl_jax/srt/disaggregation/decode.py` / `jax_transfer/conn.py` — decode 侧 start_read / poll 接收
- `scripts/disaggregation/gke/pd_singlehost_eval_job.yaml` — eval job(driver:GSM8K + TTFT 扫描)
- `scripts/disaggregation/gke/deploy.sh` — 渲染 + 部署
- `docs/developer_guide/pd_disaggregation/pd_ttft_results.md` — path-A baseline 结果表
- `docs/developer_guide/pd_disaggregation/pd_ttft_profiling_report.md` — 测量方法学(conc=1 串行等)

---

## 8. 下一步候选

1. commit 那份 yaml driver 改动(conc=1 TTFT 扫描)。
2. decode 准入背压:让 decode 侧按 KV 余量拒绝/排队,而非硬 OOM 崩(修 §6 第一条)。
3. raiden decode `prealloc_wait` 优化(大 prompt 端到端 TTFT 的主要正交开销)。
4. 若要吞吐结论:建 raiden vs path-A 的中等并发 A/B baseline。
5. 多芯片 / TP>1 PD:start_read 的 remote_endpoint 要回到 shard-matched list-of-dict(单芯才是纯字符串)。
