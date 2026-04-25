"""
Hidden state dumper for sgl-jax (TPU) inference.

Port of sglang's GPU hidden_state_dump_hook for JAX/TPU.
Dumps per-layer hidden states, logits, and sampled token IDs during inference.
Uses jax.debug.callback for saving tensors from inside JIT-compiled forward passes.

Controlled by environment variables:

    SGLANG_DUMP_HIDDEN_STATES_DIR=/tmp/dump/sgl-jax-tpu
        Directory to save dumps. Enables the dumper when set.

    SGLANG_DUMP_MAX_PASSES=4
        Max number of forward passes to dump (default: 4, i.e. 1 prefill + 3 decode).

Usage:
    1. Start the server with the env vars set:
         SGLANG_DUMP_HIDDEN_STATES_DIR=/tmp/dump/sgl-jax-tpu \
         SGLANG_DUMP_MAX_PASSES=4 \
         python3 -m sgl_jax.launch_server ...

    2. After startup, create a trigger file to start dumping:
         touch /tmp/dump/sgl-jax-tpu/.trigger

    3. Send ONE request with a short prompt and max_tokens=3.

    4. Remove trigger to stop:
         rm /tmp/dump/sgl-jax-tpu/.trigger

Note on TP:
    JAX uses SPMD sharding — tensors inside JIT are logically global.
    All dumps represent the full (replicated/gathered) tensor values.
    after_self_attn and after_mlp are POST all-reduce (full hidden_size),
    unlike GPU dumps which are PRE all-reduce (hidden_size/tp_size).

Output format:
    Tensors are saved as .npy (numpy) files. Use numpy.load() to read.
    GPU dumps use .pt (PyTorch) format — use torch.load() to read those.
"""

import json
import logging
import os
from pathlib import Path

import jax
import numpy as np

logger = logging.getLogger(__name__)


class HiddenStateDumper:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.dump_dir = os.environ.get("SGLANG_DUMP_HIDDEN_STATES_DIR", "")
        self.max_passes = int(os.environ.get("SGLANG_DUMP_MAX_PASSES", "4"))
        self.dumped_count = 0
        self._active = False
        self._current_dir = None
        self._last_completed_dir = None

        print(
            f"[DUMP_DEBUG] HiddenStateDumper.__init__: "
            f"dump_dir='{self.dump_dir}', "
            f"max_passes={self.max_passes}, "
            f"enabled={self.enabled}, "
            f"pid={os.getpid()}",
            flush=True,
        )

        if self.dump_dir:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            logger.info(
                f"HiddenStateDumper: enabled, dir={self.dump_dir}, "
                f"max_passes={self.max_passes}. "
                f"Touch {self.dump_dir}/.trigger to start dumping."
            )

    @property
    def enabled(self):
        return bool(self.dump_dir)

    @property
    def is_active(self):
        return self._active

    def _should_start_dump(self):
        if not self.enabled:
            return False
        if self.dumped_count >= self.max_passes:
            return False
        trigger = Path(self.dump_dir) / ".trigger"
        return trigger.exists()

    def begin_forward(self, forward_batch=None):
        """Called OUTSIDE JIT before each forward pass."""
        if not self.enabled:
            return

        should = self._should_start_dump()
        trigger_path = Path(self.dump_dir) / ".trigger"
        print(
            f"[DUMP_DEBUG] begin_forward called: "
            f"dumped_count={self.dumped_count}, "
            f"max_passes={self.max_passes}, "
            f"trigger_exists={trigger_path.exists()}, "
            f"should_dump={should}, "
            f"pid={os.getpid()}",
            flush=True,
        )

        if not should:
            return

        self._active = True
        self._current_dir = (
            Path(self.dump_dir) / f"pass_{self.dumped_count:04d}" / "rank_0"
        )
        self._current_dir.mkdir(parents=True, exist_ok=True)

        meta = {"pass_id": self.dumped_count, "tp_rank": 0}

        if forward_batch is not None:
            if hasattr(forward_batch, "input_ids") and forward_batch.input_ids is not None:
                input_ids_np = np.asarray(forward_batch.input_ids)
                np.save(self._current_dir / "input_ids.npy", input_ids_np)
                meta["input_ids"] = input_ids_np.tolist()
                meta["num_tokens"] = int(input_ids_np.shape[0])
            if hasattr(forward_batch, "forward_mode"):
                meta["forward_mode"] = str(forward_batch.forward_mode)
            if hasattr(forward_batch, "seq_lens") and forward_batch.seq_lens is not None:
                meta["seq_lens"] = np.asarray(forward_batch.seq_lens).tolist()
            if hasattr(forward_batch, "positions") and forward_batch.positions is not None:
                meta["positions"] = np.asarray(forward_batch.positions).tolist()

        with open(self._current_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        fm = meta.get("forward_mode", "?")
        nt = meta.get("num_tokens", "?")
        logger.info(
            f"HiddenStateDumper: dumping pass {self.dumped_count} "
            f"(mode={fm}, tokens={nt})"
        )

    def end_forward(self, logits=None):
        """Called OUTSIDE JIT after each forward pass."""
        if not self._active:
            return

        print(
            f"[DUMP_DEBUG] end_forward: pass={self.dumped_count}, "
            f"has_logits={logits is not None}, "
            f"dir={self._current_dir}",
            flush=True,
        )

        if logits is not None:
            logits_np = np.asarray(logits)
            np.save(self._current_dir / "logits.npy", logits_np)
            argmax = np.argmax(logits_np, axis=-1)
            np.save(self._current_dir / "argmax_token_ids.npy", argmax)

        self._last_completed_dir = self._current_dir
        self._active = False
        self._current_dir = None
        self.dumped_count += 1

        if self.dumped_count >= self.max_passes:
            logger.info(
                f"HiddenStateDumper: reached max_passes={self.max_passes}, "
                f"no more dumps."
            )

    def dump(self, name, tensor, layer_id=None):
        """Call from inside JIT — uses jax.debug.callback to save tensor to disk.

        The enabled check gates at trace time: if env var is not set, no
        callbacks are compiled into the XLA program (zero overhead).
        When enabled, the callback checks _active at runtime to decide
        whether to actually write.
        """
        if not self.enabled:
            return

        def _cb(t):
            self._save_tensor(name, t, layer_id)

        jax.debug.callback(_cb, tensor)

    def dump_sampled_tokens(self, token_ids):
        """Called OUTSIDE JIT after sampling."""
        if self._last_completed_dir is None:
            return
        token_ids_np = np.asarray(token_ids)
        np.save(self._last_completed_dir / "sampled_token_ids.npy", token_ids_np)
        logger.info(
            f"HiddenStateDumper: sampled tokens saved to "
            f"{self._last_completed_dir.name}/"
        )
        self._last_completed_dir = None

    def _save_tensor(self, name, tensor, layer_id=None):
        if not self._active or self._current_dir is None:
            return
        save_dir = self._current_dir
        if layer_id is not None:
            save_dir = save_dir / f"layer_{int(layer_id):02d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{name}.npy", np.asarray(tensor))


def get_hidden_state_dumper():
    return HiddenStateDumper.get()
