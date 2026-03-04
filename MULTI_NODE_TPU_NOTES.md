# 多机多卡 TPU 运行机制总结

基于 v6e-16 (4 nodes × 4 chips) 实际观察，记录从申请机器到执行任务的全流程。

## 1. 机器拓扑

以 `tpu-v6e-16` 为例：
- **物理结构**：4 个 TPU VM（4 台机器），每台 4 个 TPU 芯片，合计 16 chips
- **网络**：同一 VPC，IP 段连续（如 10.146.0.25/26/33 + head）
- **每 chip HBM**：31.25 GiB（v6e lite）
- **SkyPilot 视角**：`1x[spot](gpus=tpu-v6e-16:1, TPU-VM)` ——对 sky 是"1 个资源单元"，但实际拉起 4 台 VM

```
sky status 显示:
sky-b537-jiongxuan  GCP (asia-northeast1-b)  1x[spot](tpu-v6e-16:1, TPU-VM)  UP
                                                ↑ 实际是 4 台 VM
```

## 2. SkyPilot 如何拉起多节点

### 申请命令
```bash
sky launch scripts/tpu_v6e16_mimo.sky.yaml -n my-cluster -y --use-spot -i 120
```

### 内部流程
1. **资源协商**：sky 轮询各 region/zone，找到有 v6e-16 配额的 zone（这次是 asia-northeast1-b）
2. **同时创建 4 台 VM**：GCP TPU API 原子性地创建整个 pod，4 台 VM 一起拉起
3. **文件同步**：`file_mounts` 内容同步到所有 4 台机器
4. **setup 并行执行**：4 台机器同时跑 setup 脚本（git clone、uv venv、pip install），彼此独立
5. **run 并行执行**：setup 完成后，4 台机器同时执行 run 脚本

### SkyPilot 注入的环境变量（run 阶段）
```bash
SKYPILOT_NUM_NODES=4          # 总节点数
SKYPILOT_NODE_RANK=0/1/2/3   # 当前节点编号（head=0，workers=1/2/3）
SKYPILOT_NODE_IPS="10.x.x.x 10.x.x.x ..."  # 所有节点 IP，空格分隔，第一个是 head
```

### 日志格式
sky 在 log streaming 时自动给每台机器加前缀：
```
(head, rank=0, pid=14394)          → 主节点
(worker1, rank=1, pid=13566, ip=10.146.0.26)  → worker
(worker2, rank=2, pid=13680, ip=10.146.0.25)
(worker3, rank=3, pid=13783, ip=10.146.0.33)
```

## 3. JAX 在 TPU Pod 上的多节点行为

### 关键机制：JAX 自动发现所有 chip
在 GCP TPU Pod（多 VM 拓扑）上，**JAX 不需要显式调用 `jax.distributed.initialize()`**，
TPU runtime 通过 GCP metadata service 自动完成跨节点 rendezvous：
- 每台 VM 上的 `jax.devices()` 返回全部 16 个 TPU 设备
- 4 个 JAX 进程自动形成一个协同的分布式运行时
- `jax.sharding.reshard()` 等集合操作跨节点正常工作

### 实际表现
```
(head, rank=0)    jax.devices() → 16 个 TpuDevice
(worker1, rank=1) jax.devices() → 同样的 16 个 TpuDevice
```

每个进程都能"看到"全部 16 张卡并参与计算。

### 对比单节点 TPU
| 类型 | VM 数量 | 需要 distributed.init | chips |
|------|---------|----------------------|-------|
| v5p-8 | 1 | 否 | 8 |
| v6e-8 | 1 | 否 | 8 |
| v6e-16 | 4 | **否（自动）** | 16 |
| v6e-32 | 8 | **否（自动）** | 32 |

## 4. sglang-jax launch_server 多节点行为

### 当前命令（--nnodes 1 的情况）
即使写的是 `--nnodes 1 --node-rank 0`，由于 JAX 自动发现了全部 16 chips：
- 每台 VM 上的 launch_server 都能使用全部 16 张卡
- 4 个进程协同加载模型权重（通过 JAX 分布式 reshard）
- **实际效果**：等价于 4 个进程共同运行一个 16 卡的 server

### 权重加载的并行性
每台 VM 独立从磁盘（GCS）读取权重，各自调用 `jax.sharding.reshard()` 将 tensor 分发到正确的 chip 上。由于 JAX 多节点下 reshard 是集合操作，4 个进程会互相同步。

### 正确的多节点启动方式（推荐）
使用 `SKYPILOT_NODE_RANK` 正确区分 head 和 worker，仅 head 启动 HTTP server：
```bash
NUM_NODES=${SKYPILOT_NUM_NODES:-1}
NODE_RANK=${SKYPILOT_NODE_RANK:-0}
HEAD_IP=$(echo "$SKYPILOT_NODE_IPS" | awk '{print $1}')
HEAD_IP=${HEAD_IP:-$(hostname -I | awk '{print $1}')}

python -u -m sgl_jax.launch_server \
  --nnodes ${NUM_NODES} \
  --node-rank ${NODE_RANK} \
  --dist-init-addr ${HEAD_IP}:10011 \
  ...
```

## 5. 内存规划

以 MiMo-V2-Flash（256 experts, fp8, 47 MoE layers）为例：

### v6e-8（失败）：ep=8，每 chip 32 experts
```
MoE 权重/chip = 32 experts × 47 layers × 3 matrices × 2048 × 4096 fp8
             = 34.5 GiB > 31.25 GiB (HBM 上限) → OOM
```

### v6e-16（成功）：ep=16，每 chip 16 experts
```
MoE 权重/chip = 16 experts × 47 layers × 3 matrices × 2048 × 4096 fp8
             = 17.25 GiB < 31.25 GiB (HBM 上限) → OK
```

## 6. sky exec vs sky launch

| 操作 | 说明 |
|------|------|
| `sky launch yaml -n name` | 申请机器 + 执行 setup + 执行 run |
| `sky exec name yaml` | 仅重新执行 run 部分（机器已存在，跳过 setup） |
| `sky exec name --cmd "..."` | 在所有节点执行单条命令 |

`sky exec` 适合机器已准备好、只需重新跑任务的场景，比 launch 快很多（跳过 setup）。

## 7. 常见坑

1. **yaml 提交时已固化**：`sky launch` 时 run 脚本已上传到 sky API server，本地修改 yaml 不影响正在运行的 job。需用 `sky exec` 重新提交。
2. **disk_tier: ultra 费用高**：GCP extreme persistent disk 约 $2/h 额外费用，注意预算。
3. **spot 资源稀缺**：v6e-16 spot 资源紧张，可能需要轮询多个 zone/region 才能拿到。
4. **autostop (-i 120)**：120 分钟无活动自动关机，调试时注意。
5. **`jax.debug.print` 在 shard_map 内部会触发 SIGABRT**，多节点下尤其危险。
