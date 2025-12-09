import json
import os
import math

def format_num(num):
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    if num >= 1e3:
        return f"{num/1e3:.2f}K"
    return f"{num:.2f}"

def format_bytes(num):
    if num >= 1024**4:
        return f"{num/(1024**4):.2f} TiB"
    if num >= 1024**3:
        return f"{num/(1024**3):.2f} GiB"
    if num >= 1024**2:
        return f"{num/(1024**2):.2f} MiB"
    if num >= 1024:
        return f"{num/1024:.2f} KiB"
    return f"{num:.2f} B"

class BailingMoEAnalyzer:
    def __init__(self, config_path, batch_size=1, seq_len=4096):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.B = batch_size
        self.S = seq_len
        
        # Model Hyperparameters
        self.hidden_size = self.config['hidden_size']
        self.intermediate_size = self.config['intermediate_size']
        self.moe_intermediate_size = self.config['moe_intermediate_size']
        self.num_layers = self.config['num_hidden_layers']
        self.num_heads = self.config['num_attention_heads']
        self.num_kv_heads = self.config['num_key_value_heads']
        self.head_dim = self.config['head_dim']
        self.num_experts = self.config['num_experts']
        self.num_experts_per_tok = self.config['num_experts_per_tok']
        self.num_shared_experts = self.config.get('num_shared_experts', 0)
        self.vocab_size = self.config['vocab_size']
        self.max_position_embeddings = self.config['max_position_embeddings']
        self.first_k_dense_replace = self.config.get('first_k_dense_replace', 0)
        
        # Dtype size (bfloat16 = 2 bytes) 
        self.dtype_size = 2 
        
        # Derived
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_dim = self.num_heads * self.head_dim 
        
        # TPU v7x Specs
        self.tpu_peak_flops = 2307 * 1e12  # 2307 TFLOPs
        self.tpu_hbm_bw = 7380 * 1e9       # 7380 GB/s
        self.tpu_hbm_cap = 192 * 1024**3   # 192 GiB
        # ICI Bandwidth: 1200 GB/s is Bi-directional. Uni-directional is 600 GB/s.
        self.tpu_ici_bw_bidir = 1200 * 1e9       # 1200 GB/s (Inter-Chip Interconnect)
        self.tpu_ici_bw_uni = self.tpu_ici_bw_bidir / 2

    def calculate_latency(self, flops, mem_access):
        compute_time = flops / self.tpu_peak_flops
        mem_time = mem_access / self.tpu_hbm_bw
        return compute_time * 1e6, mem_time * 1e6 # in us
    
    def calculate_comm_latency(self, size_bytes, op_type="all_reduce", num_devices=1):
        # Simplified cost model
        if size_bytes == 0: return 0
        if num_devices <= 1 and op_type != "point_to_point": return 0
        
        # Use Uni-directional bandwidth for transmission time calculation
        bw = self.tpu_ici_bw_uni

        # Data sent per device based on collective algorithm (Ring/Torus)
        if op_type == "all_reduce":
            # 2 * (N-1)/N * S
            transfer_bytes = 2 * (num_devices - 1) / num_devices * size_bytes
        elif op_type == "all_gather":
            # (N-1) * S_local
            transfer_bytes = (num_devices - 1) * size_bytes
        elif op_type == "all_to_all":
            # (N-1)/N * S_local
            transfer_bytes = (num_devices - 1) / num_devices * size_bytes
        elif op_type == "point_to_point":
            transfer_bytes = size_bytes
        else:
            transfer_bytes = size_bytes
            
        return (transfer_bytes / bw) * 1e6 # us

    def analyze_elementwise(self, name, size_elements, rw_factor=2, ops_per_element=1, sharding_divisor=1):
        """
        Analyzes Element-wise operations (Norm, Act, etc.)
        sharding_divisor: How much this tensor is split across devices (e.g., TP size)
        """
        count = 1
        elements = size_elements / sharding_divisor
        
        mem_access_bytes = elements * self.dtype_size * rw_factor
        flops = elements * ops_per_element
        
        compute_us, memory_us = self.calculate_latency(flops, mem_access_bytes)
        
        return {
            "name": name,
            "count": count,
            "params": 0,
            "params_bytes": 0,
            "act_bytes": elements * self.dtype_size,
            "flops": flops,
            "mem_access": mem_access_bytes,
            "comm_bytes": 0,
            "compute_us": compute_us,
            "memory_us": memory_us,
            "comm_us": 0
        }

    def analyze_linear(self, name, in_features, out_features, count=1, tp_split_dim=None, tp_size=1):
        """
        Analyzes a Linear Layer with TP support.
        tp_split_dim: 'in' (ColumnParallel/RowParallel reduction), 'out' (ColumnParallel), or None
        """
        # Effective dimensions per device
        eff_in = in_features
        eff_out = out_features
        comm_bytes = 0
        comm_type = None
        
        if tp_split_dim == 'in':
            eff_in = in_features // tp_size
            # Reduction needed after matmul if splitting accumulation dimension
            comm_bytes = self.B * self.S * out_features * self.dtype_size
            comm_type = "all_reduce"
        elif tp_split_dim == 'out':
            eff_out = out_features // tp_size
            
        # Parameters
        weights = eff_in * eff_out
        params_bytes = weights * self.dtype_size
        
        # Activations
        # Input: [B, S, eff_in]
        # Output: [B, S, eff_out]
        input_bytes = self.B * self.S * eff_in * self.dtype_size
        output_bytes = self.B * self.S * eff_out * self.dtype_size
        act_mem = input_bytes + output_bytes
        
        flops = 2 * eff_in * eff_out * self.B * self.S
        memory_access_bytes = params_bytes + input_bytes + output_bytes
        
        arithmetic_intensity = flops / memory_access_bytes if memory_access_bytes > 0 else 0
        
        compute_us, memory_us = self.calculate_latency(flops * count, memory_access_bytes * count)
        comm_latency = self.calculate_comm_latency(comm_bytes, comm_type, num_devices=tp_size) * count if comm_bytes > 0 else 0
        
        return {
            "name": name,
            "count": count,
            "params": weights * count,
            "params_bytes": params_bytes * count,
            "act_bytes": act_mem * count,
            "flops": flops * count,
            "mem_access": memory_access_bytes * count,
            "comm_bytes": comm_bytes * count,
            "intensity": arithmetic_intensity,
            "compute_us": compute_us,
            "memory_us": memory_us,
            "comm_us": comm_latency
        }

    def analyze_attention(self, tp_size=1):
        ops = []
        # TP usually splits Heads.
        # Q, K, V Proj: Split Output (Column Parallel) -> No Comm
        ops.append(self.analyze_linear("Attn.Q_Proj", self.hidden_size, self.num_heads * self.head_dim, tp_split_dim='out', tp_size=tp_size))
        ops.append(self.analyze_linear("Attn.K_Proj", self.hidden_size, self.num_kv_heads * self.head_dim, tp_split_dim='out', tp_size=tp_size))
        ops.append(self.analyze_linear("Attn.V_Proj", self.hidden_size, self.num_kv_heads * self.head_dim, tp_split_dim='out', tp_size=tp_size))
        
        # Elementwise Ops (RoPE, Norm) split by TP
        norm_elements = self.B * self.S * (self.num_heads // tp_size) * self.head_dim
        if self.config.get("use_qk_norm", False):
            ops.append(self.analyze_elementwise("Attn.Q_Norm", norm_elements, rw_factor=2))
            ops.append(self.analyze_elementwise("Attn.K_Norm", self.B * self.S * (self.num_kv_heads // tp_size) * self.head_dim, rw_factor=2))
            
        rope_elements = self.B * self.S * ((self.num_heads + self.num_kv_heads) // tp_size) * self.head_dim
        ops.append(self.analyze_elementwise("Attn.RoPE", rope_elements, rw_factor=2, ops_per_element=5))

        # Attn Core (Per Device)
        eff_num_heads = self.num_heads // tp_size
        
        # QK MatMul
        qk_flops = 2 * self.B * eff_num_heads * (self.S ** 2) * self.head_dim
        logits_bytes = self.B * eff_num_heads * (self.S ** 2) * self.dtype_size
        q_bytes = self.B * self.S * eff_num_heads * self.head_dim * self.dtype_size
        kv_cache_bytes = 2 * self.B * self.S * (self.num_kv_heads // tp_size) * self.head_dim * self.dtype_size
        
        qk_compute, qk_memory = self.calculate_latency(qk_flops, q_bytes + kv_cache_bytes + logits_bytes)

        ops.append({
            "name": "Attn.QK_MatMul",
            "count": 1,
            "params": 0,
            "params_bytes": 0,
            "act_bytes": q_bytes + kv_cache_bytes + logits_bytes,
            "flops": qk_flops,
            "mem_access": q_bytes + kv_cache_bytes + logits_bytes,
            "comm_bytes": 0,
            "compute_us": qk_compute,
            "memory_us": qk_memory,
            "comm_us": 0
        })

        # Softmax
        ops.append(self.analyze_elementwise("Attn.Softmax", self.B * eff_num_heads * (self.S ** 2), rw_factor=2, ops_per_element=10))

        # AV MatMul
        av_flops = 2 * self.B * eff_num_heads * (self.S ** 2) * self.head_dim
        o_bytes = self.B * self.S * eff_num_heads * self.head_dim * self.dtype_size
        
        av_compute, av_memory = self.calculate_latency(av_flops, logits_bytes + kv_cache_bytes + o_bytes)
        
        ops.append({
            "name": "Attn.AV_MatMul",
            "count": 1,
            "params": 0,
            "params_bytes": 0,
            "act_bytes": logits_bytes + kv_cache_bytes + o_bytes,
            "flops": av_flops,
            "mem_access": logits_bytes + kv_cache_bytes + o_bytes,
            "comm_bytes": 0,
            "compute_us": av_compute,
            "memory_us": av_memory,
            "comm_us": 0
        })
        
        # Output Projection: Row Parallel (Split Input) -> AllReduce
        ops.append(self.analyze_linear("Attn.O_Proj", self.num_heads * self.head_dim, self.hidden_size, tp_split_dim='in', tp_size=tp_size))
        
        return ops

    def analyze_mlp_dense(self, tp_size=1):
        ops = []
        # Gate/Up: Column Parallel
        ops.append(self.analyze_linear("MLP.Gate_Proj", self.hidden_size, self.intermediate_size, tp_split_dim='out', tp_size=tp_size))
        ops.append(self.analyze_linear("MLP.Up_Proj", self.hidden_size, self.intermediate_size, tp_split_dim='out', tp_size=tp_size))
        
        # Act
        ops.append(self.analyze_elementwise("MLP.Act(SwiGLU)", self.B * self.S * self.intermediate_size, rw_factor=3, ops_per_element=2, sharding_divisor=tp_size))
        
        # Down: Row Parallel -> AllReduce
        ops.append(self.analyze_linear("MLP.Down_Proj", self.intermediate_size, self.hidden_size, tp_split_dim='in', tp_size=tp_size))
        return ops

    def analyze_moe(self, tp_size=1, ep_size=1):
        ops = []
        
        # Router is usually replicated or TP? Let's assume Replicated or TP. 
        # Typically Router output is small [B, S, Experts].
        ops.append(self.analyze_linear("MoE.Router", self.hidden_size, self.num_experts, tp_split_dim=None)) # Replicated
        ops.append(self.analyze_elementwise("MoE.TopK_Softmax", self.B * self.S * self.num_experts, rw_factor=2, ops_per_element=10))

        # Shared Experts (TP)
        if self.num_shared_experts > 0:
            shared_inter_dim = self.moe_intermediate_size * self.num_shared_experts
            ops.append(self.analyze_linear("MoE.Shared.Gate", self.hidden_size, shared_inter_dim, tp_split_dim='out', tp_size=tp_size))
            ops.append(self.analyze_linear("MoE.Shared.Up", self.hidden_size, shared_inter_dim, tp_split_dim='out', tp_size=tp_size))
            ops.append(self.analyze_elementwise("MoE.Shared.Act", self.B * self.S * shared_inter_dim, rw_factor=3, ops_per_element=2, sharding_divisor=tp_size))
            ops.append(self.analyze_linear("MoE.Shared.Down", shared_inter_dim, self.hidden_size, tp_split_dim='in', tp_size=tp_size))

        # Experts (EP)
        # EP splits Experts across devices.
        # Dispatch Strategy: All-Gather DP data to all EP devices.
        # Everyone sees full batch (B_local * EP) but only processes their experts.
        
        effective_inter_dim = self.num_experts_per_tok * self.moe_intermediate_size
        token_data_size = self.B * self.S * self.hidden_size * self.dtype_size
        
        # Dispatch Comm (AllGather)
        # We gather `B_local` tokens from all `EP` devices.
        # Transfer size is B_local * H * S (Dense).
        dispatch_comm_bytes = token_data_size
        # If EP=1, no comm.
        comm_bytes_dispatch = dispatch_comm_bytes if ep_size > 1 else 0
        
        ops.append({
            "name": "MoE.EP.AllGather_Dispatch",
            "count": 1,
            "params": 0,
            "params_bytes": 0,
            "act_bytes": 0,
            "flops": 0,
            "mem_access": 0,
            "comm_bytes": comm_bytes_dispatch,
            "compute_us": 0,
            "memory_us": 0,
            "comm_us": self.calculate_comm_latency(comm_bytes_dispatch, "all_gather", num_devices=ep_size)
        })

        # Expert Computation (Per Device)
        # Each device handles Total_Experts / EP_Size experts.
        # Workload Distribution:
        # - We start with `B_local` tokens on this device (from DP split).
        # - These tokens are routed to experts across the EP group.
        # - Assuming uniform load balancing, this device receives `B_local / EP_size` tokens worth of work (multiplied by k for TopK).
        # - Wait, strictly: Total Global Work = B_global * S * k.
        # - Total Devices = DP * EP.
        # - Work per Device = (B_global * S * k) / (DP * EP) = (B_local * S * k) / EP.
        # - So we must divide the local batch workload by EP_size.
        
        # Params per device
        total_experts_per_dev = self.num_experts // ep_size
        
        # FLOPs per device: 
        # Base FLOPs for full B_local batch -> divided by EP_size
        flops_gate = (2 * self.hidden_size * effective_inter_dim * self.B * self.S) / ep_size
        flops_up = (2 * self.hidden_size * effective_inter_dim * self.B * self.S) / ep_size
        flops_down = (2 * effective_inter_dim * self.hidden_size * self.B * self.S) / ep_size
        
        # Mem Access
        # Weights: Loaded experts per device (Already handled by total_experts_per_dev)
        local_params_gate = self.hidden_size * (total_experts_per_dev * self.moe_intermediate_size)
        local_params_up = self.hidden_size * (total_experts_per_dev * self.moe_intermediate_size)
        local_params_down = (total_experts_per_dev * self.moe_intermediate_size) * self.hidden_size
        
        # Inputs/Outputs per device: (Batch * S * TopK) / EP
        io_bytes_per_dev = ((self.B * self.S * self.num_experts_per_tok) * self.hidden_size * self.dtype_size) / ep_size
        
        # Gate
        gate_compute, gate_mem = self.calculate_latency(flops_gate, (local_params_gate * self.dtype_size) + 2 * io_bytes_per_dev)
        ops.append({
            "name": "MoE.Experts.Gate",
            "count": 1,
            "params": local_params_gate,
            "params_bytes": local_params_gate * self.dtype_size,
            "act_bytes": io_bytes_per_dev,
            "flops": flops_gate,
            "mem_access": (local_params_gate * self.dtype_size) + 2 * io_bytes_per_dev,
            "comm_bytes": 0,
            "compute_us": gate_compute,
            "memory_us": gate_mem,
            "comm_us": 0
        })
        
        # Up
        up_compute, up_mem = self.calculate_latency(flops_up, (local_params_up * self.dtype_size) + 2 * io_bytes_per_dev)
        ops.append({
            "name": "MoE.Experts.Up",
            "count": 1,
            "params": local_params_up,
            "params_bytes": local_params_up * self.dtype_size,
            "act_bytes": io_bytes_per_dev,
            "flops": flops_up,
            "mem_access": (local_params_up * self.dtype_size) + 2 * io_bytes_per_dev,
            "comm_bytes": 0,
            "compute_us": up_compute,
            "memory_us": up_mem,
            "comm_us": 0
        })
        
        # Act
        ops.append(self.analyze_elementwise("MoE.Experts.Act", (self.B * self.S * effective_inter_dim / ep_size), rw_factor=3, ops_per_element=2))

        # Down
        down_compute, down_mem = self.calculate_latency(flops_down, (local_params_down * self.dtype_size) + 2 * io_bytes_per_dev)
        ops.append({
            "name": "MoE.Experts.Down",
            "count": 1,
            "params": local_params_down,
            "params_bytes": local_params_down * self.dtype_size,
            "act_bytes": io_bytes_per_dev,
            "flops": flops_down,
            "mem_access": (local_params_down * self.dtype_size) + 2 * io_bytes_per_dev,
            "comm_bytes": 0,
            "compute_us": down_compute,
            "memory_us": down_mem,
            "comm_us": 0
        })
        
        # Combine Comm (AllToAll)
        # Dispatch was Dense, but Combine is usually Sparse (TopK results).
        # We send back k results per token.
        combine_comm_bytes = token_data_size * self.num_experts_per_tok
        comm_bytes_combine = combine_comm_bytes if ep_size > 1 else 0
        
        ops.append({
            "name": "MoE.EP.Combine",
            "count": 1,
            "params": 0,
            "params_bytes": 0,
            "act_bytes": 0,
            "flops": 0,
            "mem_access": 0,
            "comm_bytes": comm_bytes_combine,
            "compute_us": 0,
            "memory_us": 0,
            "comm_us": self.calculate_comm_latency(comm_bytes_combine, "all_to_all", num_devices=ep_size)
        })
        
        return ops

    def run(self, dp=1, tp=1, pp=1, ep=1):
        print(f"Analyzing BailingMoE Config: {self.config.get('architectures', ['Unknown'])[0]}")
        print(f"Parallelism: DP={dp}, TP={tp}, PP={pp}, EP={ep}")
        print(f"Global Batch: {self.B}, Seq: {self.S}. Effective Batch/Dev: {self.B/dp:.2f}")
        print(f"TPU v7x Specs: {self.tpu_peak_flops/1e12:.0f} TFLOPs, {self.tpu_hbm_bw/1e9:.0f} GB/s, {self.tpu_hbm_cap/1024**3:.0f} GiB Mem, {self.tpu_ici_bw_bidir/1e9:.0f} GB/s ICI")
        
        # Adjust Batch Size for Per-Device analysis (DP split)
        # DP splits the Global Batch.
        # MoE Dispatch is now modeled as All-Gathering this local batch across EP devices.
        original_B = self.B
        self.B = max(1, int(self.B / dp))
        
        # Adjust Layers for Per-Device analysis (PP split)
        original_layers = self.num_layers
        self.num_layers = max(1, int(self.num_layers / pp))
        
        print(f"Per-Device Analysis -> Layers: {self.num_layers}, Batch: {self.B}")
        print("-" * 185)
        print(f"{'Operator Name':<20} | {'Params (MB)':<12} | {'FLOPs':<10} | {'Mem (MB)':<10} | {'Comm (MB)':<10} | {'Comp (us)':<10} | {'Mem (us)':<10} | {'Comm (us)':<10} | {'Total (us)':<10}")
        print("-" * 185)
        
        total_stats = {
            "params_bytes": 0,
            "flops": 0,
            "mem_access": 0,
            "comm_bytes": 0,
            "latency_us": 0,
            "comm_us": 0
        }
        
        # Embeddings: Usually Replicated or TP? Assuming Replicated for now or TP-ed.
        # If PP > 1, Embeddings only on first/last stage.
        # Let's assume naive replication for calculation simplicity on "Average" device?
        # Or better, assume TP split for large vocab.
        vocab_params_per_dev = (self.vocab_size * self.hidden_size) / tp
        vocab_mem = vocab_params_per_dev * self.dtype_size
        
        # Add Embeddings only if PP stage 0 or last (averaged over PP stages for "average device latency"? No, latency is max stage).
        # Latency is determined by the slowest stage (bottleneck).
        # We analyze a "Middle" stage vs "First/Last". 
        # Let's analyze a "Representative Stage" which has Layers.
        
        dense_ops = self.analyze_mlp_dense(tp_size=tp)
        moe_ops = self.analyze_moe(tp_size=tp, ep_size=ep)
        attn_ops = self.analyze_attention(tp_size=tp)
        
        norm_elements = self.B * self.S * self.hidden_size
        input_norm_op = self.analyze_elementwise("Input RMSNorm", norm_elements, rw_factor=2, sharding_divisor=tp)
        post_norm_op = self.analyze_elementwise("Post-Attn RMSNorm", norm_elements, rw_factor=2, sharding_divisor=tp)
        
        # Per-layer breakdown (Representative)
        def print_op(op):
            total_us = max(op['compute_us'], op['memory_us'], op['comm_us'])
            print(f"{op['name']:<20} | {op['params_bytes']/1e6:<12.2f} | {format_num(op['flops']):<10} | {op['mem_access']/1e6:<10.2f} | {op['comm_bytes']/1e6:<10.2f} | {op['compute_us']:<10.2f} | {op['memory_us']:<10.2f} | {op['comm_us']:<10.2f} | {total_us:<10.2f}")
            return total_us

        print("--- Single Layer Breakdown (Per Device) ---")
        
        print_op(input_norm_op)
        for op in attn_ops: print_op(op)
        print_op(post_norm_op)
        for op in dense_ops: print_op(op)
        for op in moe_ops: print_op(op)

        print("-" * 185)
        print("--- Total Stage Summary (Per Device) ---")
        
        # Accumulate totals
        layer_latency_dense = 0
        layer_latency_moe = 0
        
        # Calc Dense Layer Latency
        l_dense = 0
        l_dense += max(input_norm_op['compute_us'], input_norm_op['memory_us'], input_norm_op['comm_us'])
        for op in attn_ops: l_dense += max(op['compute_us'], op['memory_us'], op['comm_us'])
        l_dense += max(post_norm_op['compute_us'], post_norm_op['memory_us'], post_norm_op['comm_us'])
        for op in dense_ops: l_dense += max(op['compute_us'], op['memory_us'], op['comm_us'])
        layer_latency_dense = l_dense

        # Calc MoE Layer Latency
        l_moe = 0
        l_moe += max(input_norm_op['compute_us'], input_norm_op['memory_us'], input_norm_op['comm_us'])
        for op in attn_ops: l_moe += max(op['compute_us'], op['memory_us'], op['comm_us'])
        l_moe += max(post_norm_op['compute_us'], post_norm_op['memory_us'], post_norm_op['comm_us'])
        for op in moe_ops: l_moe += max(op['compute_us'], op['memory_us'], op['comm_us'])
        layer_latency_moe = l_moe
        
        # Total Stages
        # We need to know how many Dense vs MoE layers are in THIS PP stage.
        # Uniform distribution assumption:
        # Total Dense = first_k_dense_replace
        # Total MoE = Total - first_k
        # Per Stage: Dense/PP, MoE/PP
        
        num_dense_local = max(0, self.first_k_dense_replace // pp) # Simplified
        num_moe_local = max(0, (original_layers - self.first_k_dense_replace) // pp)
        
        # If PP splits unevenly, this is approx.
        # Correct approach: Calculate for specific stage range.
        # Assuming balanced stages.
        
        total_stage_latency = (num_dense_local * layer_latency_dense) + (num_moe_local * layer_latency_moe)
        
        # PP Communication (Inter-Stage)
        # Activation Size: [B, S, Hidden]
        # In a middle stage: Recv [B,S,H] from Prev, Send [B,S,H] to Next.
        # PP=1: No comm.
        pp_comm_us = 0
        pp_comm_bytes = 0
        if pp > 1:
            act_size_bytes = self.B * self.S * self.hidden_size * self.dtype_size
            # Assume Send + Recv cost (serialized)
            # Send to Next + Recv from Prev
            pp_comm_bytes = act_size_bytes * 2 
            pp_comm_us = self.calculate_comm_latency(pp_comm_bytes, "point_to_point") # P2P over ICI
            
            total_stage_latency += pp_comm_us

        # Memory Usage Per Device
        # Weights
        total_params_bytes = 0
        # Attention Params
        for op in attn_ops: total_params_bytes += op['params_bytes']
        # Dense/MoE Params
        total_params_bytes = (total_params_bytes * (num_dense_local + num_moe_local)) # Attn in all layers
        
        # MLP Params
        dense_params = sum(op['params_bytes'] for op in dense_ops)
        total_params_bytes += (dense_params * num_dense_local)
        
        moe_params = sum(op['params_bytes'] for op in moe_ops)
        total_params_bytes += (moe_params * num_moe_local)
        
        # KV Cache
        # Total KV / (TP * PP)
        # Note: We calculated per-layer per-TP-shard KV size in `analyze_attention` -> `act_bytes` includes it? 
        # No, `act_bytes` is buffer size.
        # KV Cache persistent size:
        kv_layer_size = 2 * self.B * self.S * (self.num_kv_heads // tp) * self.head_dim * self.dtype_size
        total_kv_bytes = kv_layer_size * (num_dense_local + num_moe_local)
        
        print(f"Stage Configuration: {num_dense_local} Dense Layers, {num_moe_local} MoE Layers")
        print(f"Total Params (Dev):     {format_bytes(total_params_bytes)}")
        print(f"KV Cache (Dev):         {format_bytes(total_kv_bytes)}")
        if pp > 1:
            print(f"PP Comm (Stage):        {pp_comm_us:.2f} us ({format_bytes(pp_comm_bytes)} transfer)")
        print(f"Total Est. Latency:     {total_stage_latency/1000:.2f} ms")
        
        # Check HBM
        total_mem_usage = total_params_bytes + total_kv_bytes # + Activation Buffer (approx max of single layer)

        
        max_act_buffer = max([op['act_bytes'] for op in attn_ops + dense_ops + moe_ops])
        total_mem_usage += max_act_buffer
        
        print(f"Total HBM Usage (Est):  {format_bytes(total_mem_usage)} / {format_bytes(self.tpu_hbm_cap)}")
        if total_mem_usage > self.tpu_hbm_cap:
            print("!!! WARNING: OOM Risk !!!")
        
        # Restore original for next run if any (not needed for script)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4096)
    args = parser.parse_args()
    
    analyzer = BailingMoEAnalyzer("config.json", batch_size=args.bs, seq_len=args.seq)
    analyzer.run(dp=args.dp, tp=args.tp, pp=args.pp, ep=args.ep)
