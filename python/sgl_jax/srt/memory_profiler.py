import functools
import glob
import json
import logging
import os
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.profiler

logger = logging.getLogger(__name__)


class MemoryProfilerConfig:
    def __init__(self):
        self.enabled = False
        self.output_dir = "memory_profiles"
        self.layer_filter = None
        self.generate_prof = True
        self.generate_reports = True
        self.log_to_console = True

    def from_env(self):
        self.enabled = os.getenv("ENABLE_MEMORY_PROFILING", "0") == "1"
        self.output_dir = os.getenv("SGL_MEMORY_OUTPUT_DIR", "memory_profiles")
        self.generate_prof = os.getenv("DISABLE_PROF_GENERATION", "0") != "1"
        self.generate_reports = os.getenv("DISABLE_MEMORY_REPORTS", "0") != "1"
        self.log_to_console = os.getenv("DISABLE_MEMORY_CONSOLE_LOG", "0") != "1"

        layer_filter_str = os.getenv("MEMORY_PROFILING_LAYERS", "4")
        if layer_filter_str == "all":
            self.layer_filter = None
        elif "," in layer_filter_str:
            self.layer_filter = [int(x.strip()) for x in layer_filter_str.split(",")]
        else:
            try:
                mod_value = int(layer_filter_str)
                self.layer_filter = lambda layer_id: layer_id % mod_value == 0
            except ValueError:
                self.layer_filter = None

        return self


_config = MemoryProfilerConfig().from_env()


def configure_memory_profiler(
    enabled: bool = None,
    output_dir: str = None,
    layer_filter: Union[List[int], Callable, None] = None,
    generate_prof: bool = None,
    generate_reports: bool = None,
    log_to_console: bool = None,
):
    global _config

    if enabled is not None:
        _config.enabled = enabled
    if output_dir is not None:
        _config.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    if layer_filter is not None:
        _config.layer_filter = layer_filter
    if generate_prof is not None:
        _config.generate_prof = generate_prof
    if generate_reports is not None:
        _config.generate_reports = generate_reports
    if log_to_console is not None:
        _config.log_to_console = log_to_console


def _should_profile_layer(layer_id: Optional[int]) -> bool:
    if not _config.enabled or layer_id is None:
        return False

    if _config.layer_filter is None:
        return True
    elif isinstance(_config.layer_filter, list):
        return layer_id in _config.layer_filter
    elif callable(_config.layer_filter):
        return _config.layer_filter(layer_id)
    else:
        return False


def _calculate_tensor_memory_mb(*tensors) -> float:
    total_bytes = 0
    for tensor in tensors:
        if tensor is not None:
            total_bytes += tensor.size * tensor.dtype.itemsize
    return total_bytes / (1024 * 1024)


def _ensure_ready(*tensors):
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, "block_until_ready"):
            tensor.block_until_ready()


def _save_memory_snapshot(filename: str, condition: bool = True):
    if not condition or not _config.enabled or not _config.generate_prof:
        return

    try:
        output_path = (
            os.path.join(_config.output_dir, filename)
            if _config.output_dir
            else filename
        )
        jax.profiler.save_device_memory_profile(output_path)
    except Exception as e:
        logger.warning(f"Failed to save memory snapshot {filename}: {e}")


def _log_tensor_memory(stage: str, layer_id: Optional[int] = None, **tensors):
    if not _config.enabled or not _config.log_to_console:
        return

    if layer_id is not None and not _should_profile_layer(layer_id):
        return

    layer_info = f"Layer {layer_id}" if layer_id is not None else "Global"
    logger.info(f"  [Memory] {layer_info} - {stage}:")

    total_memory = 0
    tensor_info = []

    for name, tensor in tensors.items():
        if tensor is not None:
            memory_mb = _calculate_tensor_memory_mb(tensor)
            total_memory += memory_mb
            tensor_info.append((name, memory_mb, tensor.shape, tensor.dtype))
            logger.info(
                f"    {name:<20}: {memory_mb:>8.2f} MB - {tensor.shape} {tensor.dtype}"
            )

    logger.info(f"    Total Memory: {total_memory:.2f} MB")
    logger.info("-" * 80)

    return tensor_info, total_memory


def _create_memory_report(
    stage: str,
    tensor_dict: Dict[str, jax.Array],
    layer_id: Optional[int] = None,
    report_type: str = "general",
):
    if not _config.enabled or not _config.generate_reports:
        return

    if layer_id is not None and not _should_profile_layer(layer_id):
        return

    try:
        tensor_memory = {}
        total_memory = 0
        largest_tensor = None
        largest_memory = 0

        for name, tensor in tensor_dict.items():
            if tensor is not None:
                memory_mb = _calculate_tensor_memory_mb(tensor)
                tensor_memory[name] = {
                    "memory_mb": memory_mb,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                total_memory += memory_mb

                if memory_mb > largest_memory:
                    largest_memory = memory_mb
                    largest_tensor = name

        if layer_id is not None:
            base_name = f"memory_report_{report_type}_layer_{layer_id}_{stage}"
        else:
            base_name = f"memory_report_{report_type}_{stage}"

        report_path = os.path.join(_config.output_dir, f"{base_name}.txt")
        with open(report_path, "w") as f:
            title = f"{report_type.upper()} MEMORY ANALYSIS"
            if layer_id is not None:
                title += f" - Layer {layer_id}"
            title += f" - {stage}"

            f.write(f"  {title}\n")
            f.write("=" * 70 + "\n\n")

            f.write("  TENSOR BREAKDOWN:\n")
            f.write("-" * 50 + "\n")

            sorted_tensors = sorted(
                tensor_memory.items(), key=lambda x: x[1]["memory_mb"], reverse=True
            )

            for name, info in sorted_tensors:
                percentage = (
                    (info["memory_mb"] / total_memory) * 100 if total_memory > 0 else 0
                )
                shape_str = "x".join(map(str, info["shape"]))
                f.write(
                    f"{name:<25}: {info['memory_mb']:>8.2f} MB ({percentage:>5.1f}%) "
                    f"[{shape_str}] {info['dtype']}\n"
                )

            f.write("\n  SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  TOTAL MEMORY: {total_memory:>8.2f} MB\n")
            f.write(f"  LARGEST TENSOR: {largest_tensor} ({largest_memory:.2f} MB)\n")
            f.write(f"  STAGE: {stage}\n")
            f.write("=" * 70 + "\n")

        json_report_path = os.path.join(_config.output_dir, f"{base_name}.json")
        json_report = {
            "report_type": report_type,
            "layer_id": layer_id,
            "stage": stage,
            "total_memory_mb": total_memory,
            "largest_tensor": largest_tensor,
            "largest_memory_mb": largest_memory,
            "tensors": tensor_memory,
        }

        with open(json_report_path, "w") as f:
            json.dump(json_report, f, indent=2)

        logger.debug(f"  Generated memory reports: {report_path}, {json_report_path}")

    except Exception as e:
        logger.warning(f"Failed to create memory report for {stage}: {e}")


class MemoryProfiler:
    def __init__(
        self,
        stage: str,
        layer_id: Optional[int] = None,
        report_type: str = "general",
        auto_snapshot: bool = True,
    ):
        self.stage = stage
        self.layer_id = layer_id
        self.report_type = report_type
        self.auto_snapshot = auto_snapshot
        self.tensors_to_profile = {}

    def add_tensor(self, name: str, tensor: jax.Array):
        self.tensors_to_profile[name] = tensor
        return self

    def add_tensors(self, **tensors):
        self.tensors_to_profile.update(tensors)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tensors_to_profile:
            _ensure_ready(*self.tensors_to_profile.values())

            if self.auto_snapshot:
                filename = (
                    f"{self.report_type}_layer_{self.layer_id}_{self.stage}.prof"
                    if self.layer_id
                    else f"{self.report_type}_{self.stage}.prof"
                )
                _save_memory_snapshot(filename)

            _log_tensor_memory(self.stage, self.layer_id, **self.tensors_to_profile)

            _create_memory_report(
                self.stage, self.tensors_to_profile, self.layer_id, self.report_type
            )


def memory_profile(
    stage: str,
    layer_id: Optional[int] = None,
    report_type: str = "general",
    include_args: bool = False,
    include_result: bool = True,
):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _config.enabled or (
                layer_id is not None and not _should_profile_layer(layer_id)
            ):
                return func(*args, **kwargs)

            result = func(*args, **kwargs)

            tensors_to_profile = {}

            if include_args:
                for i, arg in enumerate(args):
                    if isinstance(arg, jax.Array):
                        tensors_to_profile[f"arg_{i}"] = arg

                for name, value in kwargs.items():
                    if isinstance(value, jax.Array):
                        tensors_to_profile[name] = value

            if include_result:
                if isinstance(result, jax.Array):
                    tensors_to_profile["result"] = result
                elif isinstance(result, (tuple, list)):
                    for i, item in enumerate(result):
                        if isinstance(item, jax.Array):
                            tensors_to_profile[f"result_{i}"] = item
                elif isinstance(result, dict):
                    for name, value in result.items():
                        if isinstance(value, jax.Array):
                            tensors_to_profile[f"result_{name}"] = value

            if tensors_to_profile:
                with MemoryProfiler(stage, layer_id, report_type) as prof:
                    prof.add_tensors(**tensors_to_profile)

            return result

        return wrapper

    return decorator


@contextmanager
def profile_memory_scope(stage: str, **tensors):

    if not _config.enabled:
        yield
        return

    _ensure_ready(*tensors.values())
    _save_memory_snapshot(f"{stage}.prof")
    _log_tensor_memory(stage, **tensors)
    _create_memory_report(stage, tensors)
    yield


def move_reports_to_output_dir():
    if not _config.output_dir:
        return []

    moved_files = []

    for prof_file in glob.glob("*.prof"):
        if not prof_file.startswith(_config.output_dir):
            dest_path = os.path.join(_config.output_dir, prof_file)
            os.rename(prof_file, dest_path)
            moved_files.append(dest_path)

    for report_file in glob.glob("memory_report_*.txt") + glob.glob(
        "memory_report_*.json"
    ):
        if not report_file.startswith(_config.output_dir):
            dest_path = os.path.join(_config.output_dir, report_file)
            os.rename(report_file, dest_path)
            moved_files.append(dest_path)

    return moved_files


def generate_summary_report(output_dir: Optional[str] = None):
    if output_dir is None:
        output_dir = _config.output_dir

    json_files = glob.glob(os.path.join(output_dir, "memory_report_*.json"))

    if not json_files:
        logger.info("No memory reports found for summary")
        return

    summary = {
        "total_reports": len(json_files),
        "peak_memory_by_stage": {},
        "memory_breakdown_by_type": {},
        "layer_analysis": {},
    }

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                report = json.load(f)

            stage = report.get("stage", "unknown")
            total_memory = report.get("total_memory_mb", 0)
            report_type = report.get("report_type", "general")
            layer_id = report.get("layer_id")

            if (
                stage not in summary["peak_memory_by_stage"]
                or total_memory > summary["peak_memory_by_stage"][stage]
            ):
                summary["peak_memory_by_stage"][stage] = total_memory

            if report_type not in summary["memory_breakdown_by_type"]:
                summary["memory_breakdown_by_type"][report_type] = []
            summary["memory_breakdown_by_type"][report_type].append(
                {"stage": stage, "memory_mb": total_memory, "layer_id": layer_id}
            )

            if layer_id is not None:
                if layer_id not in summary["layer_analysis"]:
                    summary["layer_analysis"][layer_id] = {}
                summary["layer_analysis"][layer_id][stage] = total_memory

        except Exception as e:
            logger.warning(f"Failed to process {json_file}: {e}")

    summary_path = os.path.join(output_dir, "memory_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Generated memory summary report: {summary_path}")
    return summary


def profile_attention(stage: str, layer_id: Optional[int] = None):
    return memory_profile(stage, layer_id, report_type="attention", include_result=True)


def profile_mlp(stage: str, layer_id: Optional[int] = None):
    return memory_profile(stage, layer_id, report_type="mlp", include_result=True)


def profile_model(stage: str):
    return memory_profile(stage, report_type="model", include_result=True)
