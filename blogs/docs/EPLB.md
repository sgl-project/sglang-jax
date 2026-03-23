# EPLB

# 背景

在测试Ring 1T性能过程中，发现了性能的主要瓶颈在fused moe kernel。经分析，kernel性能表现有以下的现象：

1.  随decode序列生成变长，kernel性能劣化

2.  同一次迭代，前几层kernel性能逊于后面的几十层平均值(2ms vs 1.1 ms)

3.  Padding Tokens对性能有较大的影响(通过skip padding tokens做了优化)


在此之后，我们做了expert均衡度分析[《Expert均衡度分析》](https://alidocs.dingtalk.com/i/nodes/1OQX0akWmxwn7OxaHQZnkkX68GlDd3mE?utm_scene=team_space&sideCollapsed=true&iframeQuery=utm_source%253Dportal%2526utm_medium%253Dportal_new_tab_open&corpId=dingf6b4fecb99df6d28f2c783f7214b6d69)，发现以下的现象：

1.  平均每一层，只有个别experts是热点。可能在10倍左右

2.  按device视角，不均衡的热点设备，不均衡度是平均值的2-2.5倍左右(10+7) = 2.1


可以确定，不均衡度和kernel耗时是正相关关系，为了解决这个问题，我们考虑使用DeepSeek EPLB方案

# 实现

代码：[https://github.com/sgl-project/sglang-jax/pull/781](https://github.com/sgl-project/sglang-jax/pull/781)

## 不均衡度测试

```Plain
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache   python3 -u -m sgl_jax.launch_server     --model-path /models/model_scope/Ring-1T    --trust-remote-code     --dist-init-addr=10.116.34.5:10011    --nnodes=8     --tp-size=64     --ep-size=64 --dp-size=16    --device=tpu     --moe-backend=epmoe     --random-seed=3     --mem-fraction-static=0.93     --chunked-prefill-size=1024 --precompile-token-paddings 16384  --download-dir=/dev/shm     --dtype=bfloat16     --max-running-requests 512     --skip-server-warmup     --page-size=256     --enable-expert-balance-debug  --expert-balance-segment-by decode_ste --expert-balance-segment-tokens 10 --node-rank 0
```

1.  启动服务(注意，当前必须使用EPMOE)

2.  使用数据集，测试

3.  sglang-jax server自动将experts topk相关信息，写入debug\_outputs目录中


使用 `scripts/analyze_hot_experts.py` `scripts/plot_expert_balance.py`分析

## 控制面集成+EPMOE验证正确性

```Plain
git pull --rebase && JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache  python3 -u -m sgl_jax.launch_server --model-path inclusionAI/Ling-mini-2.0 --trust-remote-code  --dist-init-addr=0.0.0.0:10011 --nnodes=1  --tp-size=8 --ep-size=8 --device=tpu --moe-backend=epmoe --random-seed=3 --mem-fraction-static=0.8 --chunked-prefill-size=2048 --download-dir=/dev/shm --dtype=bfloat16 --max-running-requests 128 --skip-server-warmup --page-size=256 --ep-num-redundant-experts 8 --ep-dispatch-algorith dynamic
```

#### 提取expert热点分布

1.  启动服务

    ```Plain
    JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache  python3 -u -m sgl_jax.launch_server --model-path inclusionAI/Ling-mini-2.0 --trust-remote-code  --dist-init-addr=0.0.0.0:10011 --nnodes=1  --tp-size=8  --dp-size 2 --ep-size=8 --device=tpu --moe-backend=epmoe --random-seed=3 --mem-fraction-static=0.9 --chunked-prefill-size=2048 --download-dir=/dev/shm --dtype=bfloat16 --max-running-requests 512 --skip-server-warmup --page-size=256 --enable-expert-balance-debug --enable-expert-distribution-recorder --expert-balance-segment-counter 20  --expert-distribution-recorder-output-file expert_dist.npy --expert-distribution-recorder-buffer-size 2000
    ```

2.  启动benching

    ```Plain
    python -m sgl_jax.bench_serving --dataset-path /orz/sharegpt_format.json --sharegpt-output-len 6000 --disable-ignore-eos  --model inclusionAI/Ling-mini-2.0 --num-prompts 512
    ```

    ![device_layer_019_heatmap.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJ5jw8WkNDq3p/img/dc3124a5-7716-4b7e-a427-6c9e6e9461f8.png)

    ![expert_layer_019_heatmap.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJ5jw8WkNDq3p/img/eabe9fa7-0368-4cf5-86af-9198bf04ab1c.png)

3.  再次启动带有eplb冗余专家的服务


```Plain
git pull --rebase && JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache  python3 -u -m sgl_jax.launch_server --model-path inclusionAI/Ling-mini-2.0 --trust-remote-code  --dist-init-addr=0.0.0.0:10011 --nnodes=1  --tp-size=8  --dp-size 2 --ep-size=8 --device=tpu --moe-backend=epmoe --random-seed=3 --mem-fraction-static=0.9 --chunked-prefill-size=2048 --download-dir=/dev/shm --dtype=bfloat16 --max-running-requests 512 --skip-server-warmup --page-size=256 --enable-expert-balance-debug --enable-expert-distribution-recorder --expert-balance-segment-counter 20  --expert-distribution-recorder-output-file expert_dist.npy --expert-distribution-recorder-buffer-size 2000 --init-expert-location expert_dist_20260208_110041.npy --ep-num-redundant-experts 32 --ep-dispatch-algorith dynamic
```
```Plain
python3 scripts/plot_expert_balance.py \
     ./debug_outputs/expert_balance_20260208_125258_487032.csv \
    --out-dir ./debug_outputs/plots2 --heatmap --heatmap-only --heatmap-cmap viridis --heatmap-normalize mean --heatmap-experts-per-device 36 --heatmap-discrete-step 0.1  --out-prefix device_
```

![device_layer_019_heatmap.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJ5jw8WkNDq3p/img/49ffc585-32c1-4db9-bfad-e338b4ed23bc.png)

```Plain
python3 scripts/plot_expert_balance.py \
     ./debug_outputs/expert_balance_20260208_125258_487032.csv \
    --out-dir ./debug_outputs/plots2 --heatmap --heatmap-only --heatmap-cmap viridis --heatmap-normalize mean  --heatmap-discrete-step 0.1  --out-prefix expert_
```

![expert_layer_019_heatmap.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJ5jw8WkNDq3p/img/d8375670-9e03-4a58-9915-f4325a7a6330.png)

## Fused Moe 

### TODO

*   [ ] 非2次幂experts fused moe kernel可用性验证，tuning

*   [ ] fused moe topk 修改(考虑使用jax实现)

*   [ ] EPLB 对fused moe性能提升的分析测试(kernel级别)


### 测试

正常server启动

```python
git pull --rebase && JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server --model-path /models/model_scope/Ring-1T-FP8/ --trust-remote-code --tp-size=32 --ep-size=32 --dp-size=16 --device=tpu --random-seed=3 --mem-fraction-static=0.94 --chunked-prefill-size=2048 --precompile-token-paddings 16384 --dtype=bfloat16 --max-running-requests=512 --skip-server-warmup --page-size=256 --dist-init-addr=10.116.20.5:10011 --moe-backend=fused --nnodes=4 --disable-radix-cache --enable-expert-balance-debug --enable-expert-distribution-recorder --expert-balance-segment-counter 20  --expert-distribution-recorder-output-file expert_dist.npy --expert-distribution-recorder-buffer-size 10000 --node-rank=0
```

```python
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server --model-path /models/model_scope/Ring-1T-FP8/ --trust-remote-code --tp-size=32 --ep-size=32 --dp-size=16 --device=tpu --random-seed=3 --mem-fraction-static=0.94 --chunked-prefill-size=1024 --precompile-token-paddings 16384 --dtype=bfloat16 --max-running-requests=512 --skip-server-warmup --page-size=256 --dist-init-addr=10.116.12.6:10011 --moe-backend=fused --nnodes=4 --disable-radix-cache --enable-expert-balance-debug --enable-expert-distribution-recorder --expert-balance-segment-counter 20  --expert-distribution-recorder-output-file expert_dist.npy --expert-distribution-recorder-buffer-size 10000 --init-expert-location expert_dist_20260210_175806.npy --ep-num-redundant-experts 32 --ep-dispatch-algorith dynamic --node-rank=0     重试  错误原因
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server --model-path /models/model_scope/Ring-1T-FP8/ --trust-remote-code --tp-size=32 --ep-size=32 --dp-size=16 --device=tpu --random-seed=3 --mem-fraction-static=0.94 --chunked-prefill-size=1024 --precompile-token-paddings 16384 --dtype=bfloat16 --max-running-requests=512 --skip-server-warmup --page-size=256 --dist-init-addr=10.116.12.6:10011 --moe-backend=fused --nnodes=4 --disable-radix-cache --enable-expert-balance-debug --enable-expert-distribution-recorder --expert-balance-segment-counter 20 --expert-distribution-recorder-output-file expert_dist.npy --expert-distribution-recorder-buffer-size 10000 --init-expert-location expert_dist_20260210_175806.npy --ep-num-redundant-experts 32 --ep-dispatch-algorith dynamic --node-rank=0
```

1.  ring-1t测试


# 分析

1.  expert均衡度 

    ```plaintext
    python3.12 scripts/plot_expert_balance.py \
         ./expert_balance_orz_288_experts_with_eplb_test1.csv \
        --out-dir ./plots/ --heatmap --heatmap-only --heatmap-cmap viridis --heatmap-normalize mean --heatmap-discrete-step 0.1 --skip-initial 5 --out-prefix expert_
    ```

2.  device均衡度


```plaintext
python3.12 ../scripts/plot_expert_balance.py \
     ./expert_balance_orz_288_experts_with_eplb_test1.csv \
    --out-dir ./plots/ --heatmap --heatmap-only --heatmap-cmap viridis --heatmap-normalize mean --heatmap-experts-per-device 9 --heatmap-discrete-step 0.1 --skip-initial 5 --out-prefix device_
```

# Reference

1.  [https://github.com/sgl-project/sglang/pull/5295/changes](https://github.com/sgl-project/sglang/pull/5295/changes)

2.  [https://github.com/deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)

3.  [https://qwen.ai/blog?id=global-load-balance](https://qwen.ai/blog?id=global-load-balance)


# 附录

[请至钉钉文档查看附件《expert\_balance\_20260213\_orz\_256experts\_without\_eplb.csv》](https://alidocs.dingtalk.com/i/nodes/3NwLYZXWynN5ADnOtQLO5dAOVkyEqBQm?iframeQuery=anchorId%3DX02mlkpuc3kw7dueewa7d)

random

[请至钉钉文档查看附件《expert\_balance\_20260213\_093822\_150361.csv》](https://alidocs.dingtalk.com/i/nodes/3NwLYZXWynN5ADnOtQLO5dAOVkyEqBQm?iframeQuery=anchorId%3DX02mlkqhikvq8ppzb5pxm)
