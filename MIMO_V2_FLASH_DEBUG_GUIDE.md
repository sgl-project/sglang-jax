# MiMo-V2-Flash 调试指南

更新时间：2026-03-03

## 集群信息
- **集群名**: `sky-c656-jiongxuan`（asia-northeast1-b，v6e-16，4 节点 × 4 chips）
- **服务器日志**: head 节点 `~/server_0.log`
- **代码路径**: `~/sky_workdir/sgl-jax/python/`（editable，PYTHONPATH 指向这里）
- **Python 环境**: `~/sky_workdir/sgl-jax/.venv/bin/python`

## 已知 Bug 修复历史

| Bug | 描述 | 修复 commit |
|-----|------|-------------|
| Bug 1 | KV head weight 按 tile 而非 repeat-per-head 复制 | 4bf1099 |
| Bug 2 | KV block-scale 应 tile-per-head 而非 repeat | 7dfa8c8 |
| Bug 3 | blockwise scale 应 P(None,None)（全局复制） | ed59ec7 |
| Bug 4 | KV reshape 用原始 k_head_num 而非 tensor shape | f140635 |
| Bug 5 | split KV 路径零填充 local kv head → 50% 输出为 0 | 8f3b9da |
| Bug 6 | FP8 k_proj scale 边界不对齐（head_dim=192 vs block=128） | 9d6f244 |
| Bug A | MoE routing 用 softmax 而非 sigmoid | a02485c |
| Bug B | noaux_tc 方法未加载 e_score_correction_bias | a02485c |
| Bug C | sliding_window_size 张冠李戴（SWA←→full 层反了） | dc354e8 |
| Bug D | Full 层错误使用 add_swa_attention_sink_bias | dc354e8 |

---

## 当前症状（截至 2026-03-03）

- 输出总是乱码，如 `"取得以及 Ağust Ağust Ağust Ağust"`
- 不同 prompt 给出相同输出
- **`MiMoV2Moe layer=1 hidden_absmax=1.7939`** 对所有 5 个 decode 步骤完全相同
  - 这意味着进入 MoE gate 的 hidden state 对每个 token 都一样 → attention 贡献为常数

---

## 诊断假设

### 假设 A：Attention 输出为零/常数（最可能）

如果 attention 每次都输出相同向量（或 0），则：
- 每层 `hidden = residual + 0` → hidden state 只被 MoE 处理
- MoE 是确定性的（同一输入→同一输出），hidden 收敛到固定吸引子
- 所有 decode 步骤 hidden_absmax 相同 ✓

**如何验证**：在 `MiMoMoeDecoderLayer.__call__` 中的 self_attn 调用前后加：
```python
jax.debug.print("layer={l} attn_out absmax={a}", l=self.layer_id, a=jnp.abs(hidden_states).max())
```
注意：`jax.debug.print` 不支持 `:.4f` 格式化，只用简单 `{}` 即可。

### 假设 B：KV Cache decode 路径错误

Prefill 可能正常，但 decode 使用了错误的 KV cache。
如果 decode 读到全零 cache，attention 输出错误。

**如何验证**：只生成 1 个 token（max_tokens=1），看第一个 token 是否正确。

### 假设 C：o_proj 权重加载错误

o_proj 是 BF16（不量化），映射用了 `head_dim_padding=True`。
需确认 o_proj weight 实际加载的形状和值范围。

---

## 服务器操作命令

### 检查服务器状态
```bash
cluster=$(cat .cluster_name_tpu)
sky exec $cluster -- "ps aux | grep sgl_jax | grep -v grep | head -3"
```

### 查看服务器日志
```bash
sky exec $cluster -- "tail -50 ~/server_0.log"
sky exec $cluster -- "grep 'MiMoV2Moe.*stats' ~/server_0.log | tail -20"
```

### 发送测试请求
```bash
sky exec $cluster -- "curl -s -X POST http://127.0.0.1:30271/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"/models/MiMo-V2-Flash\", \"prompt\": \"1+1=\", \"max_tokens\": 5, \"temperature\": 0}'"
```

### 拉取代码并重启服务器
```bash
cluster=$(cat .cluster_name_tpu)

# 1. 拉取代码（所有节点）
sky exec $cluster -- "cd ~/sky_workdir/sgl-jax && GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no' git pull origin feat/mimo-v2-flash 2>&1 | tail -3"

# 2. 杀掉旧服务器
sky exec $cluster -- "pkill -f 'sgl_jax.launch_server' 2>/dev/null; sleep 2; echo killed"

# 3. 重启（nohup 后台）
sky exec $cluster -- "
cd ~/sky_workdir/sgl-jax && source .venv/bin/activate
export PYTHONPATH=~/sky_workdir/sgl-jax/python XLA_PYTHON_CLIENT_PREALLOCATE=false
NUM_NODES=4; NODE_RANK=\${SKYPILOT_NODE_RANK:-0}
HEAD_IP=\$(echo \"\$SKYPILOT_NODE_IPS\" | head -1 | awk '{print \$1}'); HEAD_IP=\${HEAD_IP:-10.146.0.100}
nohup python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash --trust-remote-code \
  --tp-size 16 --ep-size 16 --moe-backend epmoe \
  --nnodes \${NUM_NODES} --node-rank \${NODE_RANK} \
  --dist-init-addr \${HEAD_IP}:10011 \
  --host 127.0.0.1 --port 30271 \
  --context-length 1024 --max-total-tokens 2048 --max-prefill-tokens 256 \
  --mem-fraction-static 0.7 --disable-precompile --skip-server-warmup --log-level info \
  > ~/server_\${NODE_RANK}.log 2>&1 &
echo started
"
```

---

## MiMo Config 关键参数（config.json 确认值）

```
hidden_size = 4096
num_attention_heads = 64    # Full 层 q heads
num_key_value_heads = 4     # Full 层 kv heads（原始，kv_head_padding 后扩到 16）
swa_num_attention_heads = 64
swa_num_key_value_heads = 8  （kv_head_padding 后扩到 16）
head_dim = 192, v_head_dim = 128
swa_head_dim = 192, swa_v_head_dim = 128
num_hidden_layers = 48
hybrid_layer_pattern = [0,1,1,1,1, 0,1,1,1,1,1, ...] (共 48 个元素)
  → Full 层（pattern=0）：layers 0,5,11,17,23,29,35,41,47（共9层）
  → SWA 层（pattern=1）：其余39层
moe_layer_freq = [0,1,1,...,1]  → layer 0 是 dense MLP，其余47层 MoE
sliding_window_size = 128  (没有 swa_sliding_window_size 属性！)
rope_theta = 5000000, swa_rope_theta = 10000, partial_rotary_factor = 0.334
scoring_func = sigmoid, topk_method = noaux_tc, norm_topk_prob = True
n_routed_experts = 256, num_experts_per_tok = 8
moe_intermediate_size = 2048，无 shared expert（checkpoint 无 shared keys）
add_swa_attention_sink_bias = True, add_full_attention_sink_bias = False
attention_value_scale = 0.707  (参考实现不使用，可忽略)
subc_quant_wsz = 128（FP8 block 大小）
```

---

## 关键代码位置

| 功能 | 文件 | 大概行数 |
|------|------|----------|
| Decoder Layer forward | `mimo_v2_flash.py` | ~653 |
| Attention forward (`__call__`) | `mimo_v2_flash.py` | ~478 |
| SWA/Full 层初始化 | `mimo_v2_flash.py` | ~548 |
| Weight mapping | `mimo_v2_flash.py` | ~820 |
| KV head padding | `weight_utils.py` | ~1717 |
| FP8 k_proj per-head pad | `weight_utils.py` | ~1734 |
| Split KV path (Bug5 fix) | `flashattention_backend.py` | ~568 |
| RadixAttention init | `radix_attention.py` | ~23 |

---

## 关于 jax.debug.print

- **可以用的地方**: `MiMoMoeDecoderLayer.__call__`（在 shard_map 外层）
- **不能用的地方**: 任何 `shard_map` 内部（会 SIGABRT）
- **正确格式**: `jax.debug.print("layer={l} val={v}", l=self.layer_id, v=tensor)` — 不要用 `:.4f`
- **输出去向**: stdout/stderr，均被 `~/server_0.log` 捕获
- `MiMoV2Moe` 中已有工作的 debug print：`"MiMoV2Moe layer={l} stats hidden_absmax={h} router_absmax={r}"`

---

## 关于 Attention 架构说明

### Split KV 路径（MiMo 使用）
因为 `v_head_dim=128 ≠ head_dim=192`，flashattention_backend 走 split 路径：
- `kv_dim_aligned = max(256, 128) = 256`（k 和 v 都 pad 到 256）
- v 被零填充：[seq, kv_heads, 128] → [seq, kv_heads, 256]
- 注意 sink：内部对 local kv heads 用 `jnp.tile`（Bug5 fix）而非 `jnp.pad`
- attn_output 在 mimo_v2_flash.py 中被 slice 到 v_head_dim：`[..., :head_dim]` 再 `[..., :v_head_dim]`

### KV Head 复制（kv_head_padding）
- Full 层：原始 4 kv heads → 16（tp=16 倍）
- SWA 层：原始 8 kv heads → 16（2 倍）
- 每设备 local kv heads = 1（16/16），BF16 packing 要求 2 → tile 到 2

### 权重形状（tp=16 后，per device）
- q_proj: [seq, 4096] → weight [768, 4096] (local) → output [seq, 768]
- k_proj FP8: weight [256, 4096] (local, 1 head × 256 padded) → output [seq, 256]，再 slice 到 192
- v_proj FP8: weight [128, 4096] (local, 1 head × 128) → output [seq, 128]
- o_proj BF16: input [seq, 512] (local, 4 heads × 128) → weight [512, 4096] → output [seq, 4096]
