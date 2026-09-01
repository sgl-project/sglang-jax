# SGLang-Jax 本地开发环境（Apple Silicon · macOS · CPU）

面向 **Apple Silicon Mac（M 系列，arm64）** 的 sglang-jax 开发/调试环境搭建、测试、以及 Cursor / VS Code 配置。

> **结论先行：Apple Silicon 上不需要 Docker，本地原生安装即可。**

---

## 0. 关键前提：必须用「原生 arm64」的 Python

这是最容易踩的坑，且报错信息具有误导性。

`jaxlib` 为 macOS 只发布 **`macosx_11_0_arm64`** 一种 wheel（**没有 x86_64 版**）。所以：

- 用**原生 arm64 Python** → 正常安装 ✅
- 用 **Rosetta 下的 x86_64 Python**（如 `/usr/local` 的 Intel Homebrew Python）→ 报错 ❌：

```
jaxlib==X.Y.Z has no wheels with a matching platform tag (e.g., `macosx_14_0_x86_64`)
hint: Wheels are available for `jaxlib` on: manylinux_2_27_x86_64, macosx_11_0_arm64, ...
```

**这个报错不代表"Mac 不支持"，只代表"你的 Python 是 x86_64 的"。**

### 先确认你的 shell 是原生 arm64

```bash
uname -m                     # 期望 arm64（若是 x86_64 说明在 Rosetta 下）
sysctl -n sysctl.proc_translated   # 期望 0（1 = 正在 Rosetta 转译）
```

> ⚠️ Rosetta 会把 CPU 型号**伪造**成 `Intel(R) Core(TM) i7-9750H`，`sysctl machdep.cpu.brand_string` 在 Rosetta 下不可信。判断真实硬件请用 `system_profiler SPHardwareDataType | grep Chip`。

### 认准 arm64 的工具链

| 工具 | ✅ 原生 arm64 | ❌ Rosetta/Intel |
|---|---|---|
| Homebrew | `/opt/homebrew/bin/brew` | `/usr/local/bin/brew` |
| uv | `/opt/homebrew/bin/uv` | `/usr/local/bin/uv` |
| Python | `/opt/homebrew/bin/python3` | `/usr/local/opt/python@3.12/...` |

```bash
uv --version        # 期望结尾是 aarch64-apple-darwin
```

---

## 1. 安装（一次性）

```bash
cd /Users/pengyuqing/workspace/tpu-jax/sglang-jax

# 用原生 arm64 uv 建 venv
/opt/homebrew/bin/uv venv --python 3.12 .venv

# 可编辑安装 + CPU 版 jax（走阿里云镜像，国内快很多）
VIRTUAL_ENV="$PWD/.venv" /opt/homebrew/bin/uv pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  -e "python[cpu]"
```

**验证：**

```bash
.venv/bin/python -c "
import platform, jax, sgl_jax
print('arch   :', platform.machine())     # arm64
print('jax    :', jax.__version__)
print('devices:', jax.devices())          # [CpuDevice(id=0)]
print('sgl_jax:', sgl_jax.__file__)
"
```

### 把镜像设成默认（可选，省得每次带 `--index-url`）

```bash
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml <<'TOML'
[[index]]
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
TOML
```

### 关于 extra：`cpu` / `tpu` / `gpu` / `all`

`python/pyproject.toml` 里这几个 extra **只决定装哪个 JAX 后端**，包本体相同：

| extra | 安装 | 适用 |
|---|---|---|
| `cpu` | `jax[cpu]` | **本机开发/读代码/跑单测** |
| `tpu` | `jax[tpu]`（libtpu，**仅 Linux**） | TPU 机器 |
| `gpu` | `jax[cuda12]` | NVIDIA |
| `all` | `jax[tpu]` + `fastokens` | TPU 机器（注意是 tpu 版） |

> macOS 上**只能**用 `cpu`：`jax[tpu]` 依赖的 `libtpu` 是 Linux-only。

---

## 2. 日常使用

```bash
cd /Users/pengyuqing/workspace/tpu-jax/sglang-jax
source .venv/bin/activate
```

`-e`（editable）安装意味着 **改源码立即生效**，不用重装。只有改了 `pyproject.toml` 的依赖才需要重跑安装命令。

### 跑一段 JAX（验证 JIT/XLA 编译链）

```bash
python - <<'PY'
import jax, jax.numpy as jnp

@jax.jit                       # 首次调用触发 XLA 编译（面向 CPU 后端）
def f(x):
    return jnp.sin(x).sum()

print(f(jnp.arange(1_000_000, dtype=jnp.float32)))
print(jax.devices())
PY
```

启动时打印 “No TPU/GPU found, falling back to CPU” 之类是**正常警告**。

---

## 3. 跑测试：CPU 上能跑哪些

项目用标准库 `unittest`（见 `test/README.md`），也可用 pytest 跑。

```bash
uv pip install pytest        # 默认不装，需手动安装
```

### ⚠️ 大部分 `test/srt/` 测试在 CPU 上跑不了

它们是**端到端集成测试**，会 `Engine(..., device="tpu")` 起真实引擎并从 HuggingFace 下载模型（如 `Qwen/Qwen3-8B`）。在 CPU 上必然失败：

- 连不上 `huggingface.co` → `ConnectTimeout` / `LocalEntryNotFoundError`；**且**
- 参数写死 `device="tpu"` → 本机没有 TPU 设备。

典型如 `test_srt_engine.py`，两条都占，**别拿它当冒烟测试**。

> 附带现象：失败后常见 `AttributeError: 'Engine' object has no attribute 'server_args'` —— 那是 `Engine.__init__` 失败后 `atexit` 清理钩子的连带报错，**不是根因**。

### 筛出「可能能在 CPU 上跑」的测试

```bash
cd test/srt
grep -LE 'Engine\(|device="tpu"|from_pretrained|QWEN|popen_launch' test_*.py
```

### 逐个跑（zsh 安全写法）

```bash
cd test/srt
grep -LE 'Engine\(|device="tpu"|from_pretrained|QWEN|popen_launch' test_*.py | while IFS= read -r f; do
  printf '%-48s' "$f"
  ../../.venv/bin/python -m pytest "$f" -q --no-header -p no:cacheprovider 2>&1 | tail -1
done
```

> 两个 macOS 脚本陷阱（在 Linux 容器里不会遇到）：
> - **zsh 不对未加引号的变量分词**，`for f in $LIST` 会把整个列表当成一个参数 → 用 `while IFS= read -r` 逐行读。
> - **macOS 没有 `timeout` 命令**（那是 GNU coreutils）→ 用 `perl -e 'alarm shift; exec @ARGV' 150 <cmd>`，或 `brew install coreutils` 后用 `gtimeout`。

### 实测结果（Apple M1 Pro 原生 arm64）

26 个候选中 **25 个通过，共 199 passed**（+40 subtests，2 skipped）：

| 测试文件 | 结果 |
|---|---|
| test_recurrent_state_sizing.py | 45 passed |
| test_dp_schedule_policy.py | 22 passed |
| test_native_attention_paged_decode.py | 17 passed (+40 subtests) |
| test_recurrent_track_metadata.py | 13 passed |
| test_recurrent_track_scatter.py | 12 passed |
| test_recurrent_boundary_split.py | 10 passed |
| test_dp_rank_assignment.py | 9 passed |
| test_dp_schedule_shape_aware.py | 8 passed |
| test_recurrent_split_equivalence.py | 8 passed, 2 skipped |
| test_bench_recurrent_reuse_sweep.py | 7 passed |
| test_reasoning_parser.py | 7 passed |
| test_jax_utils.py / test_parallel_utils.py | 6 passed each |
| test_merge_cache_loc.py / test_recurrent_cow_metadata.py | 4 passed each |
| test_eplb.py / test_moe_block_quant_e2e.py / test_prepare_for_extend_protected_len.py / test_server_info.py | 3 passed each |
| test_dtype_config_llama.py / test_radix_input_ids.py / test_tokenizer_manager_event.py | 2 passed each |
| test_bench_one_batch.py / test_bench_serving_prompt_types.py / test_dtype_config_consistency.py | 1 passed each |
| **test_schedule_batch_dp.py** | ❌ **collection error（仓库自身 bug，见下）** |

#### 已知仓库缺陷：test_schedule_batch_dp.py 无法收集

```
ImportError: cannot import name 'ForwardMode' from 'sgl_jax.srt.configs'
```

`test/srt/test_schedule_batch_dp.py:8` 写的是：

```python
from sgl_jax.srt.configs import ForwardMode          # ❌ 该模块 __all__ = []
```

而 `ForwardMode` 实际定义在 `python/sgl_jax/srt/model_executor/forward_batch_info.py:48`。仓库里其余 4 处都用的是正确路径：

```python
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode   # ✅
```

该文件最后一次改动是 `de5287ed dp support (#939)`；自那以后这个测试一直无法被 pytest 收集。**与平台无关**（Linux 容器里同样失败）。

### 需要 HuggingFace 时走国内镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> 镜像只解决**下载**；写死 `device="tpu"` 的测试即使下到模型，CPU 上仍跑不通。

---

## 4. Cursor / VS Code 配置

因为是**本地原生环境**，配置极简 —— 不需要 Dev Containers、不需要 Remote-SSH。

1. 打开文件夹 `/Users/pengyuqing/workspace/tpu-jax/sglang-jax`
2. `Cmd+Shift+P` → **Python: Select Interpreter** → 选 `./.venv/bin/python`
3. 完成。补全、跳转、调试、集成终端全部原生可用。

可选，把解释器固定进项目配置：

```bash
mkdir -p .vscode
cat > .vscode/settings.json <<'JSON'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["test/srt"]
}
JSON
```

> `.venv` 已在仓库 `.gitignore` 中（`.gitignore:130`），不会被提交。

---

## 5. CPU 环境的能力边界

| 能 ✅ | 不能 ❌ |
|---|---|
| 读/改源码、lint | 完整 LLM serving（`launch_server` 端到端） |
| `import jax` / `import sgl_jax` | **TPU 专用 Pallas kernel**（attention 等） |
| JIT/XLA 编译（CPU 后端） | `device="tpu"` 的测试 |
| 纯逻辑单元测试 | 有实用价值的推理性能 |

**这条限制与 Docker 无关** —— 用容器也一样跑不了 TPU kernel。真正跑推理仍需云端 TPU（SkyPilot / GCP TPU VM）。

---

## 附录 A：什么时候才需要 Docker

原生环境已足够日常开发。**只有需要「与 CI/TPU 一致的 Linux 环境」时才值得上 Docker**，例如：

- 复现只在 Linux 出现的行为（多进程 fork/spawn、`uvloop`、文件系统语义差异）
- 验证 CI 脚本

Colima 方案（Docker Desktop 的轻量替代）：

```bash
brew install colima docker
colima start --cpu 4 --memory 8 --disk 40      # Apple Silicon 默认 aarch64，原生不模拟

docker run -d --name sglang-jax-dev \
  -v "$PWD":/workspace -w /workspace \
  python:3.12 sleep infinity

docker exec sglang-jax-dev pip install -q uv
docker exec -e VIRTUAL_ENV=/opt/venv sglang-jax-dev sh -c \
  'uv venv /opt/venv --python 3.12 && uv pip install --index-url https://mirrors.aliyun.com/pypi/simple/ -e "python[cpu]"'

docker exec -it sglang-jax-dev bash -c 'source /opt/venv/bin/activate; cd /workspace; exec bash'
```

清理（释放内存/磁盘）：

```bash
docker rm -f sglang-jax-dev
colima stop && colima delete
```

> Cursor 连容器：Cursor 用的是 Anysphere 自家的 Dev Containers / Remote-SSH 扩展（微软官方版对 fork 有授权限制）。装上后 `Cmd+Shift+P` → **Dev Containers: Attach to Running Container**，再 Open Folder `/workspace`、选解释器 `/opt/venv/bin/python`。

---

## 附录 B：故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `jaxlib has no wheels ... macosx_*_x86_64` | Python 是 x86_64（Rosetta）。用 `/opt/homebrew` 的原生 arm64 Python 重建 venv |
| `uname -m` 显示 x86_64 | 当前 shell 在 Rosetta 下。开一个原生终端，或 `arch -arm64 zsh` |
| CPU 型号显示 Intel i7-9750H | Rosetta 伪造的，不可信。用 `system_profiler SPHardwareDataType \| grep Chip` |
| 安装慢 | 用阿里云镜像 `--index-url https://mirrors.aliyun.com/pypi/simple/` |
| `Failed to read metadata ... METADATA: No such file` | venv 被中断的安装弄坏：`rm -rf .venv` 后重建 |
| `Connection to huggingface.co timed out` | 设 `HF_ENDPOINT=https://hf-mirror.com`；若测试同时写死 `device="tpu"` 则换纯逻辑测试 |
| `No module named 'pytest'` | `uv pip install pytest` |
| `command not found: timeout` | macOS 无此命令，见第 3 节 |

---

*环境：Apple M1 Pro / macOS 26.6.2 / Python 3.12.14 (arm64) / jax 0.10.2*
