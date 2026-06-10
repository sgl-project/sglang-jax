# PD 分离 TTFT 干净 Profiling 报告(v6e-1)

**日期**:2026-06-10
**模型**:DeepSeek-R1-Distill-Qwen-1.5B(TP=1)
**硬件**:GKE `pd-v6e-1` 单芯 TPU v6e
**目的**:在无干扰、对齐生产的条件下,测出 no-PD / unstaged / staged 三档各阶段真实耗时,验证此前"staging ~270ms / transfer ~45ms / bench sleep 0.5s"的说法是否可靠。

---

## 1. 测量方法(三档全部对齐生产)

| 项 | 设置 | 消除的干扰 |
|----|------|-----------|
| 入口 | 走 **mini_lb router**(生产握手) | 旧 bench 的客户端 0.5s P→D sleep |
| 并发 | conc=1,**严格串行**(逐请求 await 完再发下一个) | 请求间排队/重叠 |
| Prompt | 每请求 **unique 随机 token**,`--disable-radix-cache` | 前缀缓存命中虚低 TTFT |
| 输出 | `max_new_tokens=1`(TTFT == 首 token) | decode loop 污染 |
| 采样 | warmup=3(吸收 JIT 编译 + 连接) + measure=10,取中位数 | 首请求一次性开销 |
| 数据源 | 服务端 `PD-TIME-STATS`(`--enable-request-time-stats-logging`) | 仅客户端计时看不到分阶段 |

工具:`scripts/disaggregation/bench_pd_phases.py`(`drive` 驱动 + `parse` 解析)
Manifest:`scripts/disaggregation/gke/pd_phases_{nopd,pd}_job.yaml`

**三档定义**
- **no-PD**:colocated 单 server(`--disaggregation-mode null`),无 router。
- **unstaged**:PD,D2H **关**(path B,直接 register HBM)。
- **staged**:PD,D2H **开**(path A,`copy_from_device` 到 pinned-host 后再传)。

阶段含义:
- prefill `forward`:模型前向计算
- prefill `stage`:forward_done → transfer_start(D2H 拷贝窗口)
- prefill `transfer`:transfer_start → transfer_done(实际跨 pod 搬运)
- decode `kv_wait`:transfer_entry → 首 token(接收侧 KV 传输净耗时)

---

## 2. 三档对照表(ms,中位数)

| input | 配置 | forward | stage | **transfer** | kv_wait | **client TTFT** |
|------:|------|--------:|------:|-------------:|--------:|----------------:|
| **512**  | no-PD     | —    | —   | —          | —        | **15.7** |
|          | unstaged  | 10.2 | 1.0 | **44.0**   | 48.5     | ~70 |
|          | staged    | 10.0 | 1.1 | **1389.6** | 1388.5   | ~1410 |
| **1024** | no-PD     | —    | —   | —          | —        | **25.3** |
|          | unstaged  | 17.6 | 1.2 | **43.8**   | 56.4     | ~79.5 |
|          | staged    | 17.4 | 1.1 | **1511.2** | 1513.7   | ~1538 |
| **2048** | no-PD     | —    | —   | —          | —        | **46.1** |
|          | unstaged  | 34.2 | 1.2 | **49.0**   | 73.8     | ~100 |
|          | staged    | 34.3 | 1.2 | **1543.3** | 1557.0   | ~1584 |
| **4096** | no-PD     | —    | —   | —          | —        | **70.9** |
|          | unstaged  | 崩溃 |     |            |          | **崩溃** |
|          | staged    | 不稳定 |    |            | ~7000    | **失败** |

---

## 3. client TTFT 趋势(直观对比)

```
TTFT (ms), 越短越好          512    1024   2048
─────────────────────────────────────────────
no-PD (colocated)         │  16     25     46
unstaged (D2H OFF)        │  70     80    100
staged   (D2H ON)         │ 1410   1538   1584
                            ▲       ▲      ▲
                            └─ staged ≈ 20× unstaged ≈ 30× no-PD
```

---

## 4. 核心结论

### 结论 1:"staging ~270ms" 是错的

D2H 的真实开销是 **~1.35s**,而且它**落在 `transfer` 阶段**(host 路径传输),**不在 `stage`**。

- `stage`(forward_done→transfer_start)在两档都恒为 **~1ms** —— `copy_from_device` 是异步派发,真正的 host 物化 + 跨 pod 搬运阻塞在 `transfer`。
- 因此不存在一个可分离的"270ms staging 步骤"。

### 结论 2:"transfer ~45ms" 仅对 unstaged 成立

- **unstaged**:直注 HBM,传输 **~44-49ms**,几乎与输入长度无关。✅ 旧说法对。
- **staged**:同一项变成 **~1.4s**,**慢约 30 倍**。

### 结论 3:0.5s sleep 不是主因

走 router 已无客户端 sleep,staged 仍是 ~1.4s。所以这 1.35s 是**真实代价**,不是测量伪影。早先 microbench(host vs dev source "几乎一样 ~45ms")没抓到真实成本——很可能它在计时窗口外预先 staging 了 host buffer。

### 结论 4:4096 单请求在两档都不稳

- **unstaged 4096**:立即崩溃(`JaxRuntimeError: SocketServer Connection closed` —— register 时 HBM 击穿)。
- **staged 4096**:退化到 kv_wait~7s 后连接报错。

这与 D2H 无关,是**单 v6e 芯显存余量不足 / 缺乏传输分块**的问题,上生产前需单独处理。

---

## 5. 优化建议(按收益排序)

1. **纠正 D2H 的定位**:它不是免费的稳定性开关,而是**用 ~1.35s 单请求延迟换并发准入 / 防 decode 写回 OOM**。
   - 低并发、延迟敏感 → **关 D2H**;
   - 高并发需要准入背压 → **开 D2H**。

2. **~~KV send 与 prefill compute 重叠~~(已证伪,见 §7)**
   ~~对标上游 send_kv_chunk……~~ overlap 是拿 prefill compute(~34ms)去盖 transfer(~1.4s),compute << transfer,**盖不住**。真正问题是 staged `transfer` 本身 74 MB/s 太慢(见 §7 调查),要修的是 staging 路径本身,不是去掩盖它。

3. **4096 单请求稳定性是 bug,不是容量**:4096 KV ≈ 229MB,占 32GB HBM 0.7%;且 4096=32 页 << 最大桶 512 页,不是 pool 拒绝。unstaged/staged 两档都崩,共因在传输链路对大 payload 的处理(见 §7 调查)。

---

## 7. 调查日志(long-term,持续更新)

针对 §4 留下的两个未解问题做"分析 → 实验 → 分析"循环,每轮结果追加到本节。

### 待解问题
- **Q1**:staged `transfer` 为何是 ~1.4s(74 MB/s),而 unstaged 同样 payload 仅 ~45ms(2.5 GB/s)?瓶颈在 staging 的哪一步?
- **Q2**:4096(KV ≈ 229MB)为何在两档都崩/不稳?传输链路对大单次 payload 的真实失败点在哪?

### 关键代码事实(静态分析,2026-06-10)
- staged 的 staging 成本计在 `transfer` 窗口内:`prefill.py:213` 标 `transfer_start` 后立即 `sender.send()`,`send()`(`conn.py:268`)**同步阻塞**调用 `producer_handoff` → `host_kv_pool.copy_from_device`;`transfer_done` 在 ack 回来后(`prefill.py:373`)才标。故 `stage`(forward_done→transfer_start)≈1ms 只是记账,真正 staging 阻塞在 `transfer`。
- `copy_from_device`(`host_kv_pool.py:241-`)对 28 层逐层:`device_put`(D2H)+ host 侧 `entry[i].at[:padded_pages].set(..., out_sharding=host_sharding)`。
- host buffer 按**最大桶 512 页**预分配(`runtime.py:91` `max_padded_pages=_KV_GATHER_PAGE_BUCKETS[-1]=512`)。4096 只用 32 页却写进 (512,…) 大数组 → 疑似每层整体复制 512 页缓冲。
- **嫌疑**:(a) 512 页大 host 数组上的 `.at[].set()` 整体复制/重排;(b) 带 `out_sharding` 的 host scatter 是否被派到 TPU 上算再 H2D/D2H 往返。

### 实验计划
1. **EXP-1(进行中)**:给 `copy_from_device` 加分段计时 `D2H-STAGE-TIME`(dispatch_ms / block_ms / total_ms / padded_pages / buf_pages),单跑 2048 拆开 1.4s,确认是否 staging 阻塞主导、是不是 512 页全量复制。
2. **EXP-2(计划)**:抓 4096 在 prefill+decode 两侧的完整 traceback,定位传输链路对 229MB 单次 payload 的失败点。
3. **EXP-3(计划,视 EXP-1 结论)**:若确认是 512 页全量复制 → 试 host buffer 按请求实际页数分配 / donation / 去 out_sharding,复测 transfer。

### 迭代记录
| 日期 | 实验 | 改动 | 结论 |
|------|------|------|------|
| 2026-06-10 | 三档基线 | — | staged transfer ~1.4s,unstaged ~45ms,4096 两档崩(见 §2/§4) |
| 2026-06-10 | EXP-1 插桩 | `host_kv_pool.copy_from_device` 加 `D2H-STAGE-TIME` 日志 | 待 TPU 跑数 |

---

## 6. 复现方式

```bash
# nopd 基线(单 pod)
JOB=pd-phases-nopd
sed -e "s|<JOB_NAME>|${JOB}|g" -e "s|<SSH_KEY_B64>|${KEY_B64}|g" \
    -e "s|<GH_ORG>|sgl-project|g" -e "s|<BRANCH>|pd/e2e|g" \
    scripts/disaggregation/gke/pd_phases_nopd_job.yaml | kubectl apply -f - --validate=false

# unstaged / staged(双 pod,改 <D2H_ENABLE> false/true)
JOB=pd-phases-staged
sed -e "s|<JOB_NAME>|${JOB}|g" -e "s|<SSH_KEY_B64>|${KEY_B64}|g" \
    -e "s|<GH_ORG>|sgl-project|g" -e "s|<BRANCH>|pd/e2e|g" -e "s|<D2H_ENABLE>|true|g" \
    scripts/disaggregation/gke/pd_phases_pd_job.yaml | kubectl apply -f - --validate=false

# 采集:pod0 prefill 阶段
kubectl logs <job>-0-xxxxx -c jax-tpu > prefill.log
python scripts/disaggregation/bench_pd_phases.py parse --log prefill.log --role prefill \
    --input-lens 512,1024,2048,4096 --warmup 3 --measure 10
# decode 阶段 + client TTFT 在 pod1 stdout
```

> 注:pd-v6e-1 = 2 节点,每个 PD job 占满两节点,三档需**串行**跑。
