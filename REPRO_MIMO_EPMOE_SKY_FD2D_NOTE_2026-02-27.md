# MiMo-V2-Flash EPMoE 复现记录

## 1. 当前可复现环境
- 最近一次推进到 `MoE Weights` 加载阶段的机器：`sky-cc11-jiongxuan`
- 远端代码路径：`~/sky_workdir/sgl-jax`
- 环境激活：
  ```bash
  cd ~/sky_workdir/sgl-jax
  source .venv/bin/activate
  export PYTHONPATH=~/sky_workdir/sgl-jax/python
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  ```
- 模型路径：`/models/MiMo-V2-Flash`

说明：
- 这里记录的是“去掉加载 workaround，按原始并发加载方式”的复现命令。
- 也就是不再使用 `SGL_JAX_MOE_LOAD_WORKERS` / `SGL_JAX_MOE_SPLIT_LOAD_WORKERS`。

## 2. 直接前台复现命令
```bash
cd ~/sky_workdir/sgl-jax
source .venv/bin/activate
export PYTHONPATH=~/sky_workdir/sgl-jax/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 4 --ep-size 4 --moe-backend epmoe \
  --nnodes 1 --node-rank 0 --dist-init-addr 10.128.15.204:10011 \
  --host 127.0.0.1 --port 30271 \
  --context-length 1024 \
  --max-total-tokens 2048 \
  --max-prefill-tokens 256 \
  --mem-fraction-static 0.3 \
  --disable-precompile \
  --skip-server-warmup \
  --log-level info
```

## 3. 后台复现命令
```bash
cd ~/sky_workdir/sgl-jax
source .venv/bin/activate
export PYTHONPATH=~/sky_workdir/sgl-jax/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
rm -f ~/sky_workdir/server_mimo_epmoe_models.log

nohup python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 4 --ep-size 4 --moe-backend epmoe \
  --nnodes 1 --node-rank 0 --dist-init-addr 10.128.15.204:10011 \
  --host 127.0.0.1 --port 30271 \
  --context-length 1024 \
  --max-total-tokens 2048 \
  --max-prefill-tokens 256 \
  --mem-fraction-static 0.3 \
  --disable-precompile \
  --skip-server-warmup \
  --log-level info \
  > ~/sky_workdir/server_mimo_epmoe_models.log 2>&1 < /dev/null &
```

## 4. 常用监控命令
```bash
# 看 server / scheduler 进程
pgrep -af "sgl_jax.launch_server|sglang::scheduler"

# 看是否已经进入 MoE 权重加载
grep -n "Loading MoE Weights" ~/sky_workdir/server_mimo_epmoe_models.log
grep -n "Assigned MoE group" ~/sky_workdir/server_mimo_epmoe_models.log | tail -n 20

# 看最新日志
tail -n 80 ~/sky_workdir/server_mimo_epmoe_models.log

# 看 /models 与 /dev/shm
df -h /models /dev/shm
mount | grep /models
ps aux | grep gcsfuse | grep -v grep
```

## 5. 最近一次实际推进结果
- 已确认能越过早期 `EPMoE` placeholder / sharding 初始化问题。
- 已确认能完成：
  - `Scanning metadata for 145 model files`
  - `Loading Regular Weights: 536/536`
- 已确认能进入：
  - `Loading MoE Weights`
- 已确认能连续赋值多个 MoE group，例如：
  - `model.layers.1.mlp.experts.wi_0`
  - `model.layers.1.mlp.experts.wi_0_scale`
  - `model.layers.1.mlp.experts.wi_1`
  - `model.layers.1.mlp.experts.wi_1_scale`
  - `model.layers.1.mlp.experts.wo`
  - `model.layers.1.mlp.experts.wo_scale`
- 当时已经推进到至少 `Assigned MoE group = 24`。

结论：
- 当前已经不是“启动早期 shape/sharding 低级错误”。
- 已经进入真正的权重加载长流程，后续主要卡在 host 侧内存和文件缓存压力。

## 6. `/dev/shm` 到底是什么
- `/dev/shm` 不是“swap 分区”本身。
- `/dev/shm` 是 `tmpfs`，本质上是 host RAM 支撑的内存文件系统。
- 在这次环境里，`/models` 是 `gcsfuse` 挂载，且 `gcsfuse` 使用了：
  - `--cache-dir=/dev/shm`
- 这意味着模型文件从 `/models` 读取时，文件缓存会直接吃 host 内存。

所以更准确的说法是：
- 先是 `/dev/shm` 这个 RAM-backed cache 被打满。
- host 内存继续吃紧时，系统才可能进一步出现 swap 压力或整体变慢。
- 不是 `/dev/shm` 自己就是 swap。

## 7. 当前权重加载的数据流
以当前 `python/sgl_jax/srt/utils/weight_utils.py` 的实现看，数据流更接近下面这样：

1. 扫描阶段  
   - `WeightLoader._scan_weight_info()` 先扫 `/models/MiMo-V2-Flash/*.safetensors` 的 metadata。
   - 这里只拿 key / shape / dtype，不会先把整个模型完整读进 Python 内存。

2. 构造 lazy tensor 阶段  
   - regular weight 走 `_create_lazy_tensors()` / `_create_split_lazy_tensor()`。
   - MoE weight 走 `_create_stacked_moe_lazy_tensor()` / `_create_stacked_split_moe_lazy_tensor()`。
   - 这些路径底层都用 `jax.make_array_from_callback(...)`。

3. 实际 slice 读取阶段  
   - 当 JAX 需要某个 shard / 某个 slice 时，callback 才会通过 `safe_open(...).get_slice(...)` 去读对应 safetensors 内容。
   - 这不是“先把整个 300B 模型一次性全部读进 Python RAM，再整体搬上 TPU”。

4. host 侧 staging 与 sharding 阶段  
   - 读出的 slice 会先以 host 侧数组形式存在。
   - 然后再被 JAX 组装成 sharded array，并按目标 `NamedSharding` 放到 TPU 对应设备上。

5. 当前环境的额外放大项  
   - 因为 `/models` 是 `gcsfuse`，且 cache 落在 `/dev/shm`，所以读取 safetensors 时：
     - 远端对象存储 -> `gcsfuse` 文件缓存(`/dev/shm`) -> safetensors slice -> host array -> JAX sharded array -> TPU memory
   - 这就是为什么即使代码是 lazy loading，host 侧峰值依然可能很大。

## 8. 为什么峰值会高
主因不是单一一份 Python 对象，而是几个层次叠加：
- `gcsfuse` 把文件缓存落到 `/dev/shm`
- safetensors 读取时有 host 侧 slice / numpy buffer
- JAX 在构造 sharded array 时还会有 staging / callback 生命周期
- 多个权重 group 并行推进时，host 侧会出现重叠峰值

因此，表现上就会看到：
- `df -h /dev/shm` 很快接近 100%
- 日志还能继续往前推进一段
- 但后面越来越慢，尤其在 MoE 大权重阶段出现几十秒到分钟级长尾

## 9. 能不能释放一些，降低峰值
可以，但分成两类：

### 9.1 最有效的，不是代码里 `del` 对象
如果主要压力来自 `gcsfuse` 的 `/dev/shm` 文件缓存，那么：
- 释放 Python 变量，对 `gcsfuse` cache 基本没有帮助。
- 真正有效的是：
  - 不走 `/models` 的 `gcsfuse` 路径，改用本地盘放模型
  - 或把 `gcsfuse` 的 `--cache-dir` 改到磁盘目录，而不是 `/dev/shm`
  - 或收缩 `gcsfuse` 的 file cache 上限

### 9.2 代码侧能做但收益有限的
- 降低 MoE 权重加载并发
- 减少重复 reshape / repeat / host 侧中间副本
- 按层或按 group 更严格串行加载，再尽快赋值并释放临时对象

但当前这台机器上，第一优先级仍然是：
- 不让 `gcsfuse` 把大文件缓存顶进 `/dev/shm`

## 10. 实际建议
如果目标只是稳定复现并拉别人一起看：
- 直接复用上面的原始命令即可。
- 重点同步两条事实：
  - 代码已经推进到 `MoE Weights` 真正加载阶段
  - 当前主要瓶颈是 `/models` 的 `gcsfuse` file cache 落在 `/dev/shm`，导致 host RAM/tmpfs 峰值过高

如果目标是继续往 end-to-end 跑通：
1. 优先换成本地盘放模型，或者调整 `gcsfuse --cache-dir`
2. 其次再讨论 loader 并发和中间副本优化
