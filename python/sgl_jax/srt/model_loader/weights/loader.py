import contextlib
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import fields
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig

from .reader import JaxShardReader, WeightReader
from .recipes import TensorLayout
from .source import (
    _SAFETENSORS_DTYPE_TO_JAX,
    LocalSource,
    WeightSource,
    coordinate_error,
)
from .specs import WeightSpec

logger = logging.getLogger(__name__)

_PD_WEIGHT_CACHE: dict[tuple, jax.Array] = {}


class WeightLoader:
    def __init__(
        self,
        model: nnx.Module,
        model_config: ModelConfig,
        mesh: Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        source: WeightSource | None = None,
        reader: WeightReader | None = None,
    ):
        self.model = model
        self.model_config = model_config
        self.mesh = mesh
        self.dtype = dtype
        self.dummy_mode = getattr(model_config, "_dummy_mode", False)
        if source is None:
            source = getattr(model_config, "_weight_source", None)
        self._owns_source = source is None
        self.source: WeightSource = source if source is not None else LocalSource(model_config)
        self.reader: WeightReader = reader if reader is not None else JaxShardReader(mesh)
        self.layout = TensorLayout(model_config, mesh)

    @property
    def is_static_quant(self) -> bool:
        """Check if the model uses a static FP8 checkpoint."""
        quant_cfg = getattr(self.model_config, "quantization_config", None)
        return quant_cfg is not None and quant_cfg.is_static_checkpoint

    def is_quant_ignored(self, hf_path: str) -> bool:
        """Check if a HuggingFace weight path is in the quantization ignored_layers list."""
        quant_cfg = getattr(self.model_config, "quantization_config", None)
        if quant_cfg is None or not quant_cfg.is_static_checkpoint:
            return True
        ignored = quant_cfg.ignored_layers or []
        return any(hf_path == ig or hf_path.endswith(f".{ig}") for ig in ignored)

    def has_weight_on_disk(self, hf_key: str) -> bool:
        """Return whether a concrete HF weight key exists in the safetensors files."""
        if self.dummy_mode:
            return False
        return hf_key in self.metadata

    def _pd_remap_sharding(self, src_shd):
        """Rebuild a NamedSharding whose mesh devices are positionally remapped
        from the cached (P-slice) mesh to self.mesh's slice. Keeps mesh shape,
        axis_names, axis_types and spec — only swaps devices."""
        if not isinstance(src_shd, jax.sharding.NamedSharding):
            return jax.sharding.NamedSharding(self.mesh, P())
        cache: dict[int, jax.sharding.Mesh] | None = getattr(self, "_pd_mesh_cache", None)
        if cache is None:
            cache = self._pd_mesh_cache = {}
            src_devs = sorted({d for d in src_shd.mesh.devices.flatten()}, key=lambda d: d.id)
            dst_devs = sorted(self.mesh.devices.flatten(), key=lambda d: d.id)
            self._pd_dev_map = {p.id: d for p, d in zip(src_devs, dst_devs)}
        pm = src_shd.mesh
        key = id(pm)
        if key not in cache:
            new_devs = np.empty(pm.devices.shape, dtype=object)
            for idx in np.ndindex(pm.devices.shape):
                new_devs[idx] = self._pd_dev_map[pm.devices[idx].id]  # type: ignore[union-attr]
            cache[key] = jax.sharding.Mesh(
                new_devs, axis_names=pm.axis_names, axis_types=pm.axis_types
            )
        return jax.sharding.NamedSharding(cache[key], src_shd.spec)

    def _dummy_array(self, shape, dtype, sharding):
        if getattr(self.model_config, "_abstract_mode", False):
            # Used only inside eval_shape. Specify the aval's sharding directly:
            # nested jit out_shardings are not propagated through shape tracing.
            # This also lets post-load reshapes/splits infer their true layout.
            with jax.sharding.use_abstract_mesh(sharding.mesh.abstract_mesh):
                return jnp.zeros(shape, dtype, out_sharding=sharding)
        return jax.jit(lambda: jnp.zeros(shape, dtype), out_shardings=sharding)()

    def _is_excluded_layer_weight(self, hf_key: str) -> bool:
        if not hf_key.startswith("model.layers."):
            return False

        parts = hf_key.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            return False

        layer_num = int(parts[2])
        return layer_num >= getattr(self.model_config, "num_hidden_layers", float("inf"))

    @property
    def metadata(self):
        return self.source.metadata

    def _sharding(self, spec):
        axes = spec.sharding or ()
        mesh = self.mesh
        if (
            any(axis == "expert" or isinstance(axis, tuple) and "expert" in axis for axis in axes)
            and "expert" not in mesh.axis_names
        ):
            ep = getattr(
                getattr(self.model_config, "hf_config", None),
                "ep_size",
                getattr(self.model_config, "ep_size", 1),
            )
            mesh = Mesh(
                mesh.devices.reshape(ep, -1),
                ("expert", "tensor"),
                axis_types=(jax.sharding.AxisType.Explicit,) * 2,
            )
        return jax.sharding.NamedSharding(mesh, P(*axes))

    def _expand(self, mappings):
        from dataclasses import replace

        for name, spec in mappings.items():
            if not isinstance(spec, WeightSpec):
                spec = WeightSpec(spec)
            if "*" not in name or spec.sources:
                yield name, spec
                continue
            pattern = re.compile(re.escape(name).replace(r"\*", "(.*?)"))
            for source in sorted(self.metadata):
                match = pattern.fullmatch(source)
                if match:

                    def expand(path, parts=match.groups()):
                        return path.replace("*", "{}").format(*parts)

                    path = spec.target_path
                    yield source, replace(
                        spec,
                        target_path=(
                            expand(path) if isinstance(path, str) else [expand(p) for p in path]
                        ),
                    )

    def load(
        self,
        mappings,
        safetensors_partition=1,
        dummy=False,
        validate_checkpoint_coverage=False,
    ):
        """Load a mapping of source names to targets, or explicit source groups.

        A recipe consumes its declared inputs together. Preparation runs before
        binding NNX state; subsequent execution can only assign parameter values.
        """
        start = time.monotonic()
        error = None
        try:
            prepare = getattr(self.model, "prepare_weight_loading", None)
            if prepare is not None:
                mappings = prepare(self, mappings)
            for path, module in list(nnx.iter_graph(self.model)):
                if module is self.model:
                    continue
                prepare = getattr(module, "prepare_weight_loading", None)
                if prepare is not None:
                    mappings = prepare(self, mappings, ".".join(map(str, path)))
        except Exception as exc:
            error = exc
        params = self.model
        if dummy or self.dummy_mode:
            if error is not None:
                raise error
            self._load_dummy(nnx.state(self.model), mappings)
            return
        coordinate_error(error, "preparation")
        error = None
        try:
            planned = self._plan(
                params, mappings, safetensors_partition, validate_checkpoint_coverage
            )
        except Exception as exc:
            error = exc
        coordinate_error(error, "planning")
        active, writers, skipped, unexpected, schemas = planned
        self._check_plan(active, schemas)
        error = None
        try:
            self.source.prefetch()
        except Exception as exc:
            error = exc
        coordinate_error(error, "prefetch")
        for name, spec in active:
            paths = (spec.target_path,) if isinstance(spec.target_path, str) else spec.target_path
            for path, schema in zip(paths, schemas[name]):
                param = self._get_param(params, path)
                if isinstance(param.value, jax.ShapeDtypeStruct):
                    self._assign(param, schema, schema, path)
        # One global order, independent of local I/O completion. Budget waits
        # only drain already-submitted work, never insert an ad-hoc collective.
        pending, pending_owners, pending_bytes = [], [], 0
        group_files = {
            name: {
                info["file"] for source in spec.sources or (name,) for info in self.metadata[source]
            }
            for name, spec in active
        }
        remaining_files = Counter(filename for files in group_files.values() for filename in files)
        budget = int(os.environ.get("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", str(4 << 30)))
        if budget <= 0:
            raise ValueError("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES must be positive")
        try:
            for name, spec in tqdm(active, desc="Loading weights"):
                if spec.host_recipe is not None:
                    outputs = self.reader.read_host_group(
                        self.source, spec, schemas[name], self._sharding(spec)
                    )
                elif spec.recipe is not None:
                    source_bytes = sum(
                        info.get("byte_size", 0)
                        for source in spec.sources
                        for info in self.metadata[source]
                    )
                    # The declared full-group recipes (dequantization, QKV and
                    # fused MLP) are bounded independently of model layer count.
                    required = 8 * source_bytes
                    if required > budget:
                        raise ValueError(
                            f"Recipe {name} needs up to {required} host bytes, budget={budget}"
                        )
                    if pending_bytes + required > budget:
                        jax.block_until_ready(pending)
                        pending, pending_owners, pending_bytes = [], [], 0
                    error, inputs = None, None
                    try:
                        inputs = [self._read_group_input(source) for source in spec.sources]
                    except Exception as exc:
                        error = exc
                    coordinate_error(error, "recipe read")
                    outputs = spec.recipe(inputs)
                    pending_owners.extend(inputs)
                    inputs = None
                    pending_bytes += source_bytes
                elif spec.sources:
                    outputs = self._load_experts(params, name, spec)
                else:
                    outputs = self._load_tensor(params, name, spec)
                targets = (
                    (spec.target_path,) if isinstance(spec.target_path, str) else spec.target_path
                )
                values = list(outputs)
                pending.extend(values)
                if len(values) != len(targets):
                    raise ValueError(f"Recipe output count mismatch: {name}")
                for path, value, expected in zip(targets, values, schemas[name]):
                    self._assign(self._get_param(params, path), value, expected, path)
                pending_bytes += sum(
                    sum(s.data.nbytes for s in v.addressable_shards) for v in values
                )
                if pending_bytes >= budget:
                    jax.block_until_ready(pending)
                    pending, pending_owners, pending_bytes = [], [], 0
                completed = []
                for filename in group_files[name]:
                    remaining_files[filename] -= 1
                    if not remaining_files[filename]:
                        completed.append(filename)
                if completed and self.source.retains_views:
                    jax.block_until_ready(pending)
                    pending, pending_owners, pending_bytes = [], [], 0
                    self.source.release(completed)
            jax.block_until_ready(pending)
            pending_owners.clear()
        finally:
            # A Source may own mmap or SDK memory needed by an outstanding H2D.
            jax.block_until_ready(pending)
            if self._owns_source:
                self.source.close()
        report = {
            "seconds": time.monotonic() - start,
            "loaded": tuple(writers),
            "skipped": tuple(skipped),
            "unexpected": tuple(unexpected),
        }
        logger.info("Loaded %d parameters in %.2fs", len(writers), report["seconds"])
        return report

    @staticmethod
    def _spec_identity(spec):
        def encode(value):
            if isinstance(value, np.ndarray):
                return (
                    value.shape,
                    str(value.dtype),
                    hashlib.sha256(value.tobytes()).hexdigest(),
                )
            if callable(value):
                fn = value.func if isinstance(value, partial) else value
                return f"{fn.__module__}.{fn.__qualname__}"
            return value

        return json.dumps(
            {f.name: encode(getattr(spec, f.name)) for f in fields(spec)},
            sort_keys=True,
        )

    def _check_plan(self, active, schemas):
        if jax.process_count() == 1:
            return
        from jax.experimental import multihost_utils

        description = [
            (
                name,
                self._spec_identity(spec),
                [(s.shape, str(s.dtype)) for s in schemas[name]],
            )
            for name, spec in active
        ]
        digest = np.frombuffer(
            hashlib.sha256(
                json.dumps((description, os.environ.get("SGLANG_PD_WEIGHT_CACHE"))).encode()
            ).digest(),
            np.uint8,
        )
        digests = multihost_utils.process_allgather(digest).reshape(-1, len(digest))
        if not np.all(digests == digests[0]):
            raise ValueError("Weight plans differ across processes; no tensor reads were submitted")

    def _read_group_input(self, source, index=slice(None)):
        infos = self.metadata[source]
        if len(infos) != 1:
            raise ValueError(f"Recipe must explicitly handle split input: {source}")
        info = infos[0]
        return self.source.read_tensor(info["file"], source, index)

    def _load_tensor(self, params, name, spec):
        direct = (
            isinstance(spec.target_path, str)
            and all(x is None for x in (spec.pad_width, spec.reshape, spec.repeat))
            and not (spec.kv_head_padding or spec.head_dim_padding)
        )
        sharding = None
        if direct:
            axes = tuple(spec.sharding or ())
            # PartitionSpec may omit trailing replicated axes.
            if spec.transpose_axes is not None:
                axes += (None,) * (len(spec.transpose_axes) - len(axes))
                axes = tuple(axes[i] for i in np.argsort(spec.transpose_axes))
            elif spec.transpose:
                axes += (None,) * (2 - len(axes))
                axes = axes[::-1]
            sharding = jax.sharding.NamedSharding(self.mesh, P(*axes))
        value = self.reader.read(self.source, name, spec, sharding)
        paths = (spec.target_path,) if isinstance(spec.target_path, str) else spec.target_path
        targets = {path: self._get_param(params, path).value for path in paths}
        return self.layout.transform(name, value, spec, targets)

    def _load_experts(self, params, name, spec):
        target = spec.target_path
        sharding = self._sharding(spec)
        cache_enabled = os.environ.get("SGLANG_PD_WEIGHT_CACHE") == "1"
        # Checkpoint identity and layout prevent accidental reuse between models.
        if cache_enabled and not hasattr(self, "_source_identity"):
            error = None
            try:
                self._source_identity = self.source.identity
            except Exception as exc:
                error = exc
            coordinate_error(error, "cache identity")
        cache_key = (
            (
                self._source_identity,
                target,
                self._spec_identity(spec),
                str(self.dtype),
                tuple(sharding.mesh.shape.items()),
            )
            if cache_enabled
            else None
        )
        param = self._get_param(params, target)
        cache_hit = cache_enabled and cache_key in _PD_WEIGHT_CACHE
        if cache_enabled and jax.process_count() > 1:
            from jax.experimental import multihost_utils

            cache_hit = bool(
                np.all(multihost_utils.process_allgather(np.array(cache_hit, np.int32)))
            )
        if cache_hit:
            value = _PD_WEIGHT_CACHE[cache_key]
            return (jax.device_put(value, self._pd_remap_sharding(value.sharding)),)
        value = self.reader.read(self.source, name, spec, sharding)
        with jax.set_mesh(sharding.mesh):
            value = self.layout.transform_experts(value, spec, param.value)
        if cache_enabled:
            _PD_WEIGHT_CACHE[cache_key] = value
        return (value,)

    def _load_dummy(self, params, mappings):
        # Final parameter schema is sufficient: no checkpoint scan or fake I/O.
        overrides = {}
        for spec in mappings.values():
            if isinstance(spec, WeightSpec) and spec.sharding is not None:
                paths = (
                    (spec.target_path,) if isinstance(spec.target_path, str) else spec.target_path
                )
                for path in paths:
                    if not (self.is_static_quant and path.endswith(("weight_q", "weight_scale"))):
                        with contextlib.suppress(ValueError):
                            overrides[id(self._get_param(params, path))] = P(*spec.sharding)
        for _, leaf in jax.tree_util.tree_flatten_with_path(
            params, is_leaf=lambda x: isinstance(x, nnx.VariableState)
        )[0]:
            if not isinstance(leaf, nnx.VariableState):
                continue
            value = leaf.value
            if not isinstance(value, (jax.Array, jax.ShapeDtypeStruct, jax.core.Tracer)):
                continue
            sharding = getattr(value, "sharding", None)
            spec = overrides.get(id(leaf), getattr(sharding, "spec", P()))
            axes = {
                a for entry in spec for a in ((entry,) if isinstance(entry, str) else entry or ())
            }
            mesh = self.mesh
            if (
                any(
                    axis == "expert" or isinstance(axis, tuple) and "expert" in axis
                    for axis in axes
                )
                and "expert" not in mesh.axis_names
            ):
                ep = getattr(
                    getattr(self.model_config, "hf_config", None),
                    "ep_size",
                    getattr(self.model_config, "ep_size", 1),
                )
                mesh = Mesh(
                    mesh.devices.reshape(ep, -1),
                    ("expert", "tensor"),
                    axis_types=(jax.sharding.AxisType.Explicit,) * 2,
                )
            self._assign(
                leaf,
                self._dummy_array(value.shape, value.dtype, jax.sharding.NamedSharding(mesh, spec)),
                value,
                "dummy parameter",
            )
        nnx.update(self.model, params)

    def _target_schema(self, params, name, spec, cache):
        paths = (spec.target_path,) if isinstance(spec.target_path, str) else spec.target_path
        targets = {path: self._get_param(params, path).value for path in paths}
        if spec.recipe is not None or spec.host_recipe is not None:
            return tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in targets.values())
        source = spec.sources[0] if spec.sources else name
        infos = self.metadata[source]
        shape = list(infos[0]["shape"])
        if spec.concat_axis is not None:
            shape[spec.concat_axis] = sum(info["shape"][spec.concat_axis] for info in infos)
        dtype = _SAFETENSORS_DTYPE_TO_JAX[infos[0]["dtype"]]
        if spec.sources:
            if spec.transpose and len(shape) > 1:
                shape[-2:] = shape[-2:][::-1]
            count = (
                len(spec.physical_to_logical_map)
                if spec.physical_to_logical_map is not None
                else len(spec.sources)
            )
            shape = [count, *shape]
        # Tracing is cached by layout/shape, never once per layer. Recipes with
        # model-specific NumPy transforms bind their prepared final schema.
        key = (
            tuple(shape),
            str(dtype),
            tuple(
                (re.sub(r"\.\d+(?=\.|$)", ".*", p), v.shape, str(v.dtype))
                for p, v in targets.items()
            ),
            spec.transpose,
            spec.transpose_axes,
            spec.reshape,
            spec.repeat,
            spec.pad_width,
            spec.split_sizes,
            spec.split_axis,
            spec.head_dim_padding,
            spec.kv_head_padding,
            spec.sharding,
            bool(spec.sources),
            name.rsplit(".", 1)[-1],
        )
        if key not in cache:

            def infer(value):
                if spec.sources:
                    return (self.layout.transform_experts(value, spec, targets[paths[0]]),)
                return self.layout.transform(name, value, spec, targets)

            with jax.set_mesh(self.mesh):
                cache[key] = jax.eval_shape(infer, jax.ShapeDtypeStruct(tuple(shape), dtype))
        return cache[key]

    def _plan(self, params, mappings, safetensors_partition, validate_checkpoint_coverage):
        budget = int(os.environ.get("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", str(4 << 30)))
        if budget <= 0:
            raise ValueError("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES must be positive")
        entries = list(self._expand(mappings))
        used, writers, skipped = set(), {}, []
        active = []
        identities = {}
        shape_cache = {}
        schemas = {}
        # Fail before reads. A partial model may intentionally provide only a
        # subset of targets, but each declared group must be complete.
        for name, spec in entries:
            sources = spec.sources or (name,)
            missing = [s for s in sources if s not in self.metadata]
            if missing:
                if spec.optional or all(self._is_excluded_layer_weight(s) for s in missing):
                    skipped.extend(missing)
                    continue
                raise ValueError(f"Missing checkpoint inputs for {name}: {missing}")
            if spec.concat_axis is not None:
                for source in sources:
                    if len(self.metadata[source]) < safetensors_partition:
                        raise ValueError(f"Incomplete checkpoint partitions: {source}")
            if spec.recipe is not None:
                required = 8 * sum(
                    info["byte_size"] for source in sources for info in self.metadata[source]
                )
                if required > budget:
                    raise ValueError(
                        f"Recipe {name} needs up to {required} host bytes, budget={budget}"
                    )
            targets = (spec.target_path,) if isinstance(spec.target_path, str) else spec.target_path
            if spec.sharding is None:
                from dataclasses import replace

                value = self._get_param(params, targets[0]).value
                axes = getattr(getattr(value, "sharding", None), "spec", P())
                spec = replace(spec, sharding=tuple(axes))
            for target in targets:
                variable = self._get_param(params, target)
                if id(variable) in identities:
                    raise ValueError(
                        f"Duplicate writer for shared parameter {target}: {identities[id(variable)]} and {name}"
                    )
                identities[id(variable)] = name
                if target in writers:
                    raise ValueError(f"Duplicate writer for {target}: {writers[target]} and {name}")
                writers[target] = name
            schemas[name] = self._target_schema(params, name, spec, shape_cache)
            for target, schema in zip(targets, schemas[name]):
                expected = self._get_param(params, target).value
                if schema.shape != expected.shape:
                    raise ValueError(
                        f"Target shape mismatch for {target}: checkpoint layout produces "
                        f"{schema.shape}, model declares {expected.shape}"
                    )
            used.update(sources)
            active.append((name, spec))
        # Recipes only depend on declared checkpoint inputs. Keep groups near
        # their files so a late appended GDN/MLA recipe cannot pin all previously
        # read mmap pages until the end of the model. Basenames are rank-neutral.
        active.sort(
            key=lambda entry: min(
                (os.path.basename(info["file"]), info.get("byte_offset", 0), entry[0])
                for source in entry[1].sources or (entry[0],)
                for info in self.metadata[source]
            )
        )
        unexpected = sorted(
            s for s in self.metadata if s not in used and not self._is_excluded_layer_weight(s)
        )
        if validate_checkpoint_coverage and unexpected:
            raise ValueError(f"Unmapped checkpoint tensors: {unexpected[:10]}")
        return active, writers, skipped, unexpected, schemas

    @staticmethod
    def _get_param(params, path: str):
        if isinstance(params, dict) and path in params:
            return params[path]
        keys = path.split(".")
        current_level = params

        for key in keys:
            if key.isdigit():
                current_level = current_level[int(key)]
            else:
                if hasattr(current_level, "__contains__") and key in current_level:
                    current_level = current_level[key]
                elif hasattr(current_level, key):
                    current_level = getattr(current_level, key)
                else:
                    raise ValueError(f"{path} is not a valid param path")

        return current_level

    @staticmethod
    def _assign(param, value, expected, path):
        """The only parameter write: preserve identity and enforce the planned schema."""
        if (value.shape, value.dtype) != (expected.shape, expected.dtype):
            raise ValueError(
                f"Loaded target {path}: got {value.shape}/{value.dtype}, "
                f"expected {expected.shape}/{expected.dtype}"
            )
        param.value = value
