"""MiMo-V2.5 host-side audio codec helpers.

The first integration slice only normalizes and validates RVQ audio codes that
are already provided by a request or returned by an HF processor. Raw audio
encoding is intentionally left behind this model-specific helper instead of
being embedded in the shared multimodal tokenizer.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib.util
import io
import json
import math
import os
import types
import urllib.request

import numpy as np


@dataclasses.dataclass
class MiMoV25AudioPayload:
    """Structured MiMo-V2.5 audio-code payload for omni requests.

    ``codes`` is always stored time-major as ``[T, num_channels]`` (one row per
    RVQ timestep). ``codes_layout`` records that contract so the embed stage does
    not have to re-guess the axis order from the shape (which is ambiguous for the
    square ``T == num_channels`` case). ``codebook_sizes`` optionally carries the
    per-quantizer codebook size so out-of-range ids can be rejected per channel
    instead of against the loose scalar ``codebook_size`` upper bound.
    """

    codes: np.ndarray | list
    token_lengths: list[int]
    offsets: list[tuple[int, int]] | None = None
    audio_token_id: int | None = None
    num_channels: int = 20
    codebook_size: int = 1280
    codebook_sizes: list[int] | None = None
    group_size: int = 4
    codes_layout: str = "time_major"
    source: str = "unknown"
    is_tokenized: bool = True

    @classmethod
    def from_obj(cls, obj) -> MiMoV25AudioPayload | None:
        """Build a payload from dataclass/dict transport forms."""
        if obj is None:
            return None
        if isinstance(obj, cls):
            return obj
        if not isinstance(obj, dict):
            raise TypeError(f"MiMo-V2.5 audio payload must be a dict or payload, got {type(obj)!r}")

        data = dict(obj)
        if "codes" in data and not isinstance(data["codes"], np.ndarray):
            data["codes"] = np.asarray(data["codes"], dtype=np.int32)
        if "token_lengths" in data and data["token_lengths"] is not None:
            data["token_lengths"] = [int(length) for length in data["token_lengths"]]
        if "offsets" in data and data["offsets"] is not None:
            offsets = []
            for offset in data["offsets"]:
                if len(offset) != 2:
                    raise ValueError(
                        f"MiMo-V2.5 audio payload offsets must be [start, end] pairs, got {offset}"
                    )
                offsets.append(tuple(map(int, offset)))
            data["offsets"] = offsets
        for key in ("audio_token_id", "num_channels", "codebook_size", "group_size"):
            if key in data and data[key] is not None:
                data[key] = int(data[key])
        if data.get("codebook_sizes") is not None:
            data["codebook_sizes"] = [int(size) for size in data["codebook_sizes"]]
        if data.get("codes_layout") is not None:
            data["codes_layout"] = str(data["codes_layout"])
        if isinstance(data.get("is_tokenized"), str):
            value = data["is_tokenized"].strip().lower()
            if value in {"true", "1", "yes"}:
                data["is_tokenized"] = True
            elif value in {"false", "0", "no"}:
                data["is_tokenized"] = False
            else:
                raise ValueError(
                    "MiMo-V2.5 audio payload is_tokenized must be a boolean-like value, "
                    f"got {data['is_tokenized']!r}"
                )
        return cls(**data)

    def to_transport_dict(self) -> dict:
        """Return a JSON/pickle-friendly representation for mm_inputs transport."""
        codes = np.asarray(self.codes, dtype=np.int32)
        return {
            "codes": codes.tolist(),
            "token_lengths": [int(length) for length in self.token_lengths],
            "offsets": (
                [[int(start), int(end)] for start, end in self.offsets]
                if self.offsets is not None
                else None
            ),
            "audio_token_id": (
                int(self.audio_token_id) if self.audio_token_id is not None else None
            ),
            "num_channels": int(self.num_channels),
            "codebook_size": int(self.codebook_size),
            "codebook_sizes": (
                [int(size) for size in self.codebook_sizes]
                if self.codebook_sizes is not None
                else None
            ),
            "group_size": int(self.group_size),
            "codes_layout": str(self.codes_layout),
            "source": self.source,
            "is_tokenized": bool(self.is_tokenized),
        }


class MiMoV25AudioCodecProcessor:
    """Host-side MiMo-V2.5 audio-code processor boundary."""

    DEFAULT_NUM_CHANNELS = 20
    DEFAULT_CODEBOOK_SIZE = 1280
    DEFAULT_GROUP_SIZE = 4

    def __init__(
        self,
        model_path: str | None = None,
        *,
        audio_token_id: int | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.audio_token_id = audio_token_id
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._remote_module: types.ModuleType | None = None
        self._audio_tokenizer = None
        self._codebook_sizes: list[int] | None | bool = False  # False = not yet probed

    def get_codebook_sizes(self) -> list[int] | None:
        """Per-quantizer codebook sizes from audio_tokenizer/config.json.

        Returns ``None`` only when there is no audio_tokenizer config to read (the
        loose scalar ``codebook_size`` check then applies). If the config file *exists*
        but is unreadable / missing / malformed ``codebook_size``, raise rather than
        silently downgrade to the uniform-1280 check (review R2-8): a real MiMo-V2.5
        omni checkpoint always ships this file, so a present-but-broken one is a real
        misconfiguration, not a reason to skip per-channel validation.
        """
        if self._codebook_sizes is not False:
            return self._codebook_sizes
        self._codebook_sizes = None
        if not self.model_path:
            return None
        config_path = os.path.join(self.model_path, "audio_tokenizer", "config.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path) as f:
                tok_config = json.load(f)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"MiMo-V2.5 audio_tokenizer config exists but is unreadable: {config_path} "
                f"({exc}). Cannot derive per-quantizer codebook sizes for code validation."
            ) from exc
        sizes = tok_config.get("codebook_size")
        if not isinstance(sizes, (list, tuple)) or not sizes:
            raise ValueError(
                f"MiMo-V2.5 audio_tokenizer config {config_path} has no per-quantizer "
                f"'codebook_size' list (got {sizes!r}); cannot validate code ranges."
            )
        self._codebook_sizes = [int(size) for size in sizes]
        return self._codebook_sizes

    @classmethod
    def build_payload_from_codes(
        cls,
        audio_codes,
        *,
        audio_token_id: int | None = None,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        codebook_size: int = DEFAULT_CODEBOOK_SIZE,
        codebook_sizes: list[int] | None = None,
        group_size: int = DEFAULT_GROUP_SIZE,
        source: str,
    ) -> MiMoV25AudioPayload:
        if group_size <= 0:
            raise ValueError(f"MiMo-V2.5 audio group_size must be positive, got {group_size}")
        # A payload only exists when audio is present, and audio is useless without a
        # scatter target. Fail here rather than let placeholder expansion / count
        # validation / the embed scatter all silently no-op on a None id (review R2-15).
        if audio_token_id is None:
            raise ValueError(
                "MiMo-V2.5 audio payload built without an audio_token_id; cannot expand "
                "placeholders or scatter audio embeddings. Resolve audio_token_id "
                "(expected 151669) from the model config / processor."
            )
        code_segments = cls.normalize_code_segments(
            audio_codes,
            num_channels=num_channels,
            codebook_size=codebook_size,
            codebook_sizes=codebook_sizes,
        )
        token_lengths = [math.ceil(codes.shape[0] / group_size) for codes in code_segments]
        padded_codes = [
            cls.pad_codes_to_group_size(codes, group_size=group_size) for codes in code_segments
        ]
        codes = np.concatenate(padded_codes, axis=0)
        return MiMoV25AudioPayload(
            codes=codes,
            token_lengths=token_lengths,
            audio_token_id=audio_token_id,
            num_channels=num_channels,
            codebook_size=codebook_size,
            codebook_sizes=list(codebook_sizes) if codebook_sizes is not None else None,
            group_size=group_size,
            source=source,
        )

    @classmethod
    def normalize_payload(
        cls,
        payload: MiMoV25AudioPayload,
        *,
        audio_token_id: int | None = None,
        num_channels: int | None = None,
        codebook_size: int | None = None,
        codebook_sizes: list[int] | None = None,
        group_size: int | None = None,
        source: str | None = None,
    ) -> MiMoV25AudioPayload:
        """Normalize a user-provided payload into the stage0-ready contract."""
        num_channels = int(num_channels or payload.num_channels or cls.DEFAULT_NUM_CHANNELS)
        codebook_size = int(codebook_size or payload.codebook_size or cls.DEFAULT_CODEBOOK_SIZE)
        if codebook_sizes is None:
            codebook_sizes = payload.codebook_sizes
        codebook_sizes = list(codebook_sizes) if codebook_sizes is not None else None
        group_size = int(group_size or payload.group_size or cls.DEFAULT_GROUP_SIZE)
        if group_size <= 0:
            raise ValueError(f"MiMo-V2.5 audio group_size must be positive, got {group_size}")
        token_lengths = [int(length) for length in payload.token_lengths]
        if not token_lengths or any(length <= 0 for length in token_lengths):
            raise ValueError(f"MiMo-V2.5 audio token_lengths must be positive, got {token_lengths}")
        offsets = cls.normalize_offsets(payload.offsets, token_lengths)

        codes = cls.normalize_codes(
            payload.codes,
            num_channels=num_channels,
            codebook_size=codebook_size,
            codebook_sizes=codebook_sizes,
        )
        expected_rows = sum(token_lengths) * group_size
        if codes.shape[0] != expected_rows:
            if len(token_lengths) == 1:
                raw_token_length = math.ceil(codes.shape[0] / group_size)
                if token_lengths[0] != raw_token_length:
                    raise ValueError(
                        "MiMo-V2.5 audio payload token_lengths mismatch: "
                        f"codes rows={codes.shape[0]} imply token_lengths=[{raw_token_length}] "
                        f"with group_size={group_size}, got {token_lengths}"
                    )
                return MiMoV25AudioPayload(
                    codes=cls.pad_codes_to_group_size(codes, group_size=group_size),
                    token_lengths=token_lengths,
                    offsets=offsets,
                    audio_token_id=(
                        audio_token_id if audio_token_id is not None else payload.audio_token_id
                    ),
                    num_channels=num_channels,
                    codebook_size=codebook_size,
                    codebook_sizes=codebook_sizes,
                    group_size=group_size,
                    source=source or payload.source,
                    is_tokenized=payload.is_tokenized,
                )
            raise ValueError(
                "MiMo-V2.5 audio payload codes must be stage0-ready for multi-audio payloads: "
                f"codes rows={codes.shape[0]}, expected={expected_rows} from "
                f"token_lengths={token_lengths}, group_size={group_size}. "
                "Pass multi-audio codes as a list/3D audio_codes input so each segment can be padded "
                "independently, or provide already padded payload.codes."
            )

        return MiMoV25AudioPayload(
            codes=codes,
            token_lengths=token_lengths,
            offsets=offsets,
            audio_token_id=audio_token_id if audio_token_id is not None else payload.audio_token_id,
            num_channels=num_channels,
            codebook_size=codebook_size,
            codebook_sizes=codebook_sizes,
            group_size=group_size,
            source=source or payload.source,
            is_tokenized=payload.is_tokenized,
        )

    @staticmethod
    def normalize_offsets(
        offsets: list[tuple[int, int]] | None,
        token_lengths: list[int],
    ) -> list[tuple[int, int]] | None:
        if offsets is None:
            return None
        normalized_offsets = []
        if len(offsets) != len(token_lengths):
            raise ValueError(
                "MiMo-V2.5 audio payload offset count mismatch: "
                f"found {len(offsets)} offsets, expected {len(token_lengths)} "
                f"from token_lengths={token_lengths}"
            )
        for idx, (offset, expected_len) in enumerate(zip(offsets, token_lengths)):
            if len(offset) != 2:
                raise ValueError(
                    f"MiMo-V2.5 audio payload offsets must be [start, end] pairs, got {offset}"
                )
            start, end = (int(offset[0]), int(offset[1]))
            if start < 0 or end <= start:
                raise ValueError(
                    "MiMo-V2.5 audio payload offsets must be positive-length spans, "
                    f"got offset=({start}, {end})"
                )
            actual_len = end - start
            if actual_len != int(expected_len):
                raise ValueError(
                    "MiMo-V2.5 audio payload offset length mismatch: "
                    f"span={idx}, offset=({start},{end}), found={actual_len}, "
                    f"expected={expected_len}"
                )
            normalized_offsets.append((start, end))
        return normalized_offsets

    @classmethod
    def normalize_code_segments(
        cls,
        audio_codes,
        *,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        codebook_size: int = DEFAULT_CODEBOOK_SIZE,
        codebook_sizes: list[int] | None = None,
    ) -> list[np.ndarray]:
        if audio_codes is None:
            raise ValueError("MiMo-V2.5 audio_codes cannot be None")
        if hasattr(audio_codes, "detach"):
            audio_codes = audio_codes.detach().cpu().numpy()
        if (
            isinstance(audio_codes, np.ndarray)
            and audio_codes.ndim == 3
            or cls._looks_like_code_segment_list(audio_codes)
        ):
            segments = [
                cls.normalize_codes(
                    item,
                    num_channels=num_channels,
                    codebook_size=codebook_size,
                    codebook_sizes=codebook_sizes,
                )
                for item in audio_codes
            ]
        else:
            segments = [
                cls.normalize_codes(
                    audio_codes,
                    num_channels=num_channels,
                    codebook_size=codebook_size,
                    codebook_sizes=codebook_sizes,
                )
            ]
        if not segments:
            raise ValueError("MiMo-V2.5 audio_codes cannot be empty")
        for idx, codes in enumerate(segments):
            if codes.shape[0] <= 0:
                raise ValueError(f"MiMo-V2.5 audio_codes segment {idx} cannot be empty")
        return segments

    @staticmethod
    def _looks_like_code_segment_list(audio_codes) -> bool:
        if hasattr(audio_codes, "detach") or isinstance(audio_codes, np.ndarray):
            return False
        if not isinstance(audio_codes, (list, tuple)) or not audio_codes:
            return False
        try:
            arr = np.asarray(audio_codes)
        except ValueError:
            return True
        if arr.dtype == object:
            return True
        if arr.ndim <= 2:
            return False
        return arr.ndim == 3

    @staticmethod
    def pad_codes_to_group_size(codes: np.ndarray, *, group_size: int) -> np.ndarray:
        if group_size <= 0:
            raise ValueError(f"MiMo-V2.5 audio group_size must be positive, got {group_size}")
        remainder = codes.shape[0] % group_size
        if remainder == 0:
            return codes
        pad_len = group_size - remainder
        return np.concatenate([codes, np.repeat(codes[-1:], pad_len, axis=0)], axis=0)

    @classmethod
    def normalize_codes(
        cls,
        audio_codes,
        *,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        codebook_size: int = DEFAULT_CODEBOOK_SIZE,
        codebook_sizes: list[int] | None = None,
    ) -> np.ndarray:
        if audio_codes is None:
            raise ValueError("MiMo-V2.5 audio_codes cannot be None")
        if hasattr(audio_codes, "detach"):
            audio_codes = audio_codes.detach().cpu().numpy()
        codes = np.asarray(audio_codes)
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        if codes.ndim != 2:
            raise ValueError(
                "MiMo-V2.5 audio_codes must be 2D [T, C] or [C, T], " f"got shape={codes.shape}"
            )
        if codes.shape[-1] == num_channels and codes.shape[0] == num_channels:
            # Square [num_channels, num_channels] codes are layout-ambiguous; the
            # codec contract is time-major [T, C], so trust the last axis as C.
            normalized = codes
        elif codes.shape[-1] == num_channels:
            normalized = codes
        elif codes.shape[0] == num_channels:
            normalized = codes.T
        else:
            raise ValueError(
                f"MiMo-V2.5 audio_codes must have {num_channels} channels, "
                f"got shape={codes.shape}"
            )
        normalized = normalized.astype(np.int32, copy=False)
        if normalized.shape[0] <= 0:
            raise ValueError("MiMo-V2.5 audio_codes cannot be empty")
        cls._validate_code_range(
            normalized, codebook_size=codebook_size, codebook_sizes=codebook_sizes
        )
        return normalized

    @staticmethod
    def _validate_code_range(
        normalized: np.ndarray,
        *,
        codebook_size: int,
        codebook_sizes: list[int] | None,
    ) -> None:
        """Reject out-of-range RVQ ids, per-channel when codebook_sizes is known."""
        if not normalized.size:
            return
        if normalized.min() < 0:
            raise ValueError(
                "MiMo-V2.5 audio_codes contain negative ids: "
                f"min={int(normalized.min())}, expected >= 0"
            )
        num_channels = normalized.shape[1]
        if codebook_sizes is not None:
            if len(codebook_sizes) != num_channels:
                raise ValueError(
                    "MiMo-V2.5 codebook_sizes length mismatch: "
                    f"got {len(codebook_sizes)} entries for {num_channels} channels"
                )
            channel_max = normalized.max(axis=0)
            for channel, (observed, limit) in enumerate(zip(channel_max, codebook_sizes)):
                if int(observed) >= int(limit):
                    raise ValueError(
                        "MiMo-V2.5 audio_codes contain out-of-range ids on channel "
                        f"{channel}: max={int(observed)}, expected [0, {int(limit)})"
                    )
            return
        max_code = int(normalized.max())
        if max_code >= codebook_size:
            raise ValueError(
                "MiMo-V2.5 audio_codes contain out-of-range ids: "
                f"max={max_code}, expected [0, {codebook_size})"
            )

    @staticmethod
    def validate_placeholder_count(
        input_ids: list[int] | np.ndarray | None,
        payload: MiMoV25AudioPayload | None,
    ) -> None:
        MiMoV25AudioCodecProcessor.attach_offsets_from_input_ids(input_ids, payload)

    @staticmethod
    def attach_offsets_from_input_ids(
        input_ids: list[int] | np.ndarray | None,
        payload: MiMoV25AudioPayload | None,
    ) -> None:
        """Attach per-audio placeholder spans and validate them against token lengths."""
        if input_ids is None or payload is None or payload.audio_token_id is None:
            return
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        offsets = MiMoV25AudioCodecProcessor._find_token_runs(ids, payload.audio_token_id)
        actual = sum(end - start for start, end in offsets)
        expected_lengths = [int(length) for length in payload.token_lengths]
        expected = sum(expected_lengths)
        if actual != expected:
            raise ValueError(
                "MiMo-V2.5 audio placeholder count mismatch: "
                f"found {actual} audio pad tokens, expected {expected} from "
                f"audio_codes shape={np.asarray(payload.codes).shape}, "
                f"group_size={payload.group_size}"
            )
        if len(offsets) != len(expected_lengths):
            run_lengths = [end - start for start, end in offsets]
            raise ValueError(
                "MiMo-V2.5 audio span count mismatch: "
                f"found {len(offsets)} audio spans with lengths={run_lengths}, "
                f"expected {len(expected_lengths)} spans with token_lengths={expected_lengths}"
            )
        for idx, ((start, end), expected_len) in enumerate(zip(offsets, expected_lengths)):
            actual_len = end - start
            if actual_len != expected_len:
                raise ValueError(
                    "MiMo-V2.5 audio span length mismatch: "
                    f"span={idx}, offset=({start},{end}), found={actual_len}, "
                    f"expected={expected_len}"
                )
        payload.offsets = offsets

    @staticmethod
    def expand_single_audio_placeholders(
        input_ids: list[int] | np.ndarray | None,
        payload: MiMoV25AudioPayload | None,
    ) -> list[int] | np.ndarray | None:
        """Expand one-pad-per-audio template placeholders to codec token lengths."""
        if input_ids is None or payload is None or payload.audio_token_id is None:
            return input_ids
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        offsets = MiMoV25AudioCodecProcessor._find_token_runs(ids, payload.audio_token_id)
        expected_lengths = [int(length) for length in payload.token_lengths]
        if not offsets or len(offsets) != len(expected_lengths):
            return input_ids

        run_lengths = [end - start for start, end in offsets]
        if run_lengths == expected_lengths:
            payload.offsets = offsets
            return input_ids
        if any(length != 1 for length in run_lengths):
            return input_ids

        expanded = []
        cursor = 0
        for (start, end), expected_len in zip(offsets, expected_lengths):
            if expected_len <= 0:
                raise ValueError(
                    f"MiMo-V2.5 audio token length must be positive, got {expected_len}"
                )
            expanded.extend(ids[cursor:start])
            expanded.extend([payload.audio_token_id] * expected_len)
            cursor = end
        expanded.extend(ids[cursor:])
        return expanded

    @staticmethod
    def _find_token_runs(input_ids: list[int], token_id: int) -> list[tuple[int, int]]:
        offsets = []
        pos = 0
        while pos < len(input_ids):
            if input_ids[pos] != token_id:
                pos += 1
                continue
            start = pos
            while pos < len(input_ids) and input_ids[pos] == token_id:
                pos += 1
            offsets.append((start, pos))
        return offsets

    def encode(self, audio_data) -> MiMoV25AudioPayload:
        """Encode raw audio or mel inputs into a MiMo-V2.5 audio payload."""
        mels = [self._to_mel_tensor(item) for item in self._normalize_audio_items(audio_data)]
        return self.encode_mels(mels, source="host_codec")

    def encode_mels(self, mels, *, source: str = "host_codec_mel") -> MiMoV25AudioPayload:
        """Encode one or more mel tensors shaped [T, 128] into RVQ codes."""
        if mels is None:
            raise ValueError("MiMo-V2.5 mel inputs cannot be None")
        mel_items = self._normalize_mel_items(mels)
        if not mel_items:
            raise ValueError("MiMo-V2.5 mel inputs cannot be empty")

        tokenizer = self._load_audio_tokenizer()
        code_tensors = self._tokenize_audio_batch(mel_items, tokenizer.encoder)
        if not code_tensors:
            raise ValueError("MiMo-V2.5 audio tokenizer produced no codes")

        codes_np = [self._tensor_to_numpy(codes) for codes in code_tensors]
        payload = self.build_payload_from_codes(
            codes_np,
            audio_token_id=self.audio_token_id,
            num_channels=self.DEFAULT_NUM_CHANNELS,
            codebook_size=self.DEFAULT_CODEBOOK_SIZE,
            codebook_sizes=self.get_codebook_sizes(),
            group_size=self.DEFAULT_GROUP_SIZE,
            source=source,
        )
        return payload

    def _normalize_audio_items(self, audio_data) -> list:
        if audio_data is None:
            raise ValueError("MiMo-V2.5 audio_data cannot be None")
        if self._is_waveform_tuple(audio_data):
            return [audio_data]
        if isinstance(audio_data, list):
            return audio_data
        return [audio_data]

    def _normalize_mel_items(self, mels) -> list:
        if self._is_torch_tensor(mels):
            return self._split_mel_tensor(mels)
        arr = np.asarray(mels) if not isinstance(mels, list) else None
        if arr is not None and arr.ndim in (2, 3):
            return self._split_mel_tensor(arr)
        items = []
        for mel in mels:
            items.extend(self._split_mel_tensor(mel))
        return items

    def _split_mel_tensor(self, mel) -> list:
        torch = self._require_torch()
        tensor = mel if self._is_torch_tensor(mel) else torch.as_tensor(mel)
        if tensor.ndim == 2:
            tensor = self._ensure_mel_time_major(tensor)
            return [tensor.to(dtype=torch.float32)]
        if tensor.ndim == 3:
            return [self._ensure_mel_time_major(item).to(dtype=torch.float32) for item in tensor]
        raise ValueError(f"MiMo-V2.5 mel must be 2D or 3D, got shape={tuple(tensor.shape)}")

    def _ensure_mel_time_major(self, mel):
        if mel.shape[-1] == 128:
            time_major = mel
        elif mel.shape[0] == 128:
            time_major = (
                mel.transpose(0, 1) if self._is_torch_tensor(mel) else np.swapaxes(mel, 0, 1)
            )
        else:
            raise ValueError(f"MiMo-V2.5 mel must have 128 bins, got shape={tuple(mel.shape)}")
        if time_major.shape[0] <= 0:
            raise ValueError(
                f"MiMo-V2.5 mel time dimension cannot be empty, got shape={tuple(mel.shape)}"
            )
        return time_major

    def _to_mel_tensor(self, item):
        if self._looks_like_mel(item):
            return self._split_mel_tensor(item)[0]
        waveform, sample_rate = self._load_waveform(item)
        return self._waveform_to_mel(waveform, sample_rate)

    def _looks_like_mel(self, item) -> bool:
        if self._is_waveform_tuple(item):
            return False
        if self._is_torch_tensor(item):
            return item.ndim in (2, 3) and (item.shape[-1] == 128 or item.shape[-2] == 128)
        if isinstance(item, np.ndarray):
            return item.ndim in (2, 3) and (item.shape[-1] == 128 or item.shape[-2] == 128)
        return False

    @staticmethod
    def _is_waveform_tuple(item) -> bool:
        return isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, np.integer))

    @staticmethod
    def _is_torch_tensor(value) -> bool:
        return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")

    def _load_waveform(self, item):
        torch = self._require_torch()
        torchaudio = self._require_torchaudio()

        if self._is_waveform_tuple(item):
            waveform, sample_rate = item
            return torch.as_tensor(waveform, dtype=torch.float32), int(sample_rate)
        if self._is_torch_tensor(item) or isinstance(item, np.ndarray):
            return torch.as_tensor(item, dtype=torch.float32), 24000
        if isinstance(item, dict) and "url" in item:
            item = item["url"]
        if hasattr(item, "url"):
            item = item.url
        if isinstance(item, bytes):
            return self._load_waveform_bytes(item)
        if isinstance(item, str):
            if os.path.exists(item):
                waveform, sample_rate = torchaudio.load(item)
                return waveform, int(sample_rate)
            if item.startswith(("http://", "https://")):
                with urllib.request.urlopen(item, timeout=10) as resp:
                    return self._load_waveform_bytes(resp.read())
            if item.startswith("data:") and "base64," in item:
                item = item.split("base64,", 1)[1]
            try:
                return self._load_waveform_bytes(base64.b64decode(item, validate=True))
            except Exception as exc:
                raise ValueError("Unsupported MiMo-V2.5 audio source format") from exc
        raise ValueError(f"Unsupported MiMo-V2.5 audio source type: {type(item)!r}")

    def _load_waveform_bytes(self, audio_bytes: bytes):
        torch = self._require_torch()
        torchaudio = self._require_torchaudio()
        try:
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
            return waveform, int(sample_rate)
        except Exception:
            import soundfile as sf

            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            return torch.as_tensor(audio_array, dtype=torch.float32), int(sample_rate)

    def _waveform_to_mel(self, waveform, sample_rate: int):
        torch = self._require_torch()
        torchaudio = self._require_torchaudio()
        audio = torch.as_tensor(waveform, dtype=torch.float32)
        if audio.ndim == 2:
            if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
                audio = audio.mean(dim=0)
            else:
                audio = audio.mean(dim=-1)
        if audio.ndim != 1:
            raise ValueError(f"MiMo-V2.5 waveform must be 1D or 2D, got shape={tuple(audio.shape)}")
        if sample_rate != 24000:
            audio = torchaudio.functional.resample(audio, sample_rate, 24000)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=960,
            win_length=960,
            hop_length=240,
            n_mels=128,
            power=1.0,
            center=True,
            f_min=0,
            f_max=None,
        )
        mel = mel_transform(audio)
        return torch.log(torch.clamp(mel, min=1e-7)).transpose(0, 1).to(dtype=torch.float32)

    def _load_audio_tokenizer(self):
        if self._audio_tokenizer is not None:
            return self._audio_tokenizer
        if not self.model_path:
            raise ValueError("MiMo-V2.5 audio tokenizer requires model_path")

        torch = self._require_torch()
        config_cls = self._load_remote_symbol("MiMoAudioTokenizerConfig")
        tokenizer_cls = self._load_remote_symbol("MiMoAudioTokenizer")
        tokenizer_dir = os.path.join(self.model_path, "audio_tokenizer")
        config_path = os.path.join(tokenizer_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"MiMo-V2.5 audio tokenizer config not found: {config_path}")
        with open(config_path) as f:
            config = config_cls(**json.load(f))
        tokenizer = tokenizer_cls(config)

        safetensors_path = os.path.join(tokenizer_dir, "model.safetensors")
        bin_path = os.path.join(tokenizer_dir, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No MiMo-V2.5 audio tokenizer weights found in {tokenizer_dir}"
            )

        load_result = tokenizer.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        if missing_keys:
            preview = ", ".join(missing_keys[:8])
            suffix = "..." if len(missing_keys) > 8 else ""
            raise ValueError(
                "MiMo-V2.5 audio tokenizer weights are incomplete: "
                f"missing {len(missing_keys)} keys ({preview}{suffix})"
            )
        dtype = getattr(torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
        tokenizer = tokenizer.to(device=self.device, dtype=dtype)
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        self._audio_tokenizer = tokenizer
        return tokenizer

    def _load_remote_symbol(self, symbol: str):
        module = self._load_remote_module()
        if module is not None:
            return getattr(module, symbol)
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        return get_class_from_dynamic_module(
            f"modeling_mimo_v2.{symbol}",
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

    def _load_remote_module(self):
        if self._remote_module is not None:
            return self._remote_module
        if not self.model_path:
            return None
        modeling_path = os.path.join(self.model_path, "modeling_mimo_v2.py")
        if not os.path.exists(modeling_path):
            return None
        module_name = f"_sgl_jax_mimo_v25_{abs(hash(os.path.abspath(modeling_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, modeling_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load MiMo-V2.5 remote module from {modeling_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._remote_module = module
        return module

    def _tokenize_audio_batch(self, mels, audio_tokenizer_encoder, segment_size=6000):
        torch = self._require_torch()
        if not mels:
            return []
        device = next(audio_tokenizer_encoder.parameters()).device
        input_len_seg_per_mel = []
        for mel in mels:
            input_len = int(mel.size(0))
            segs = [segment_size] * (input_len // segment_size)
            if input_len % segment_size > 0:
                segs.append(input_len % segment_size)
            input_len_seg_per_mel.append(segs)
        input_lens_flat = [s for segs in input_len_seg_per_mel for s in segs]
        input_features = torch.cat([mel.to(device) for mel in mels], dim=0)
        input_lens_t = torch.tensor(input_lens_flat, dtype=torch.long, device=device)
        feature_groups, len_groups = self._group_by_length(input_features, input_lens_t, 256000)
        encoded_parts = []
        with torch.no_grad():
            for features, lengths in zip(feature_groups, len_groups):
                codes, _ = audio_tokenizer_encoder.encode(
                    input_features=features,
                    input_lens=lengths,
                    return_codes_only=True,
                )
                encoded_parts.append(codes)
            codes = torch.cat(encoded_parts, dim=-1).transpose(0, 1).detach()
            code_lengths = []
            for segs in input_len_seg_per_mel:
                out_len = audio_tokenizer_encoder.get_output_length(
                    torch.tensor(segs, dtype=torch.long, device=device)
                )
                if getattr(audio_tokenizer_encoder, "down_sample_layer", None) is not None:
                    avg = audio_tokenizer_encoder.config.avg_pooler
                    out_len = out_len // avg + (out_len % avg != 0).long()
                code_lengths.append(int(out_len.sum().item()))
            return list(torch.split(codes, code_lengths))

    @staticmethod
    def _group_by_length(features, lengths, max_length):
        split_points, current_sum = [], 0
        for i, seq_len in enumerate(lengths):
            seq_len = int(seq_len.item())
            if current_sum + seq_len > max_length and current_sum > 0:
                split_points.append(i)
                current_sum = seq_len
            else:
                current_sum += seq_len
        group_sizes, prev = [], 0
        for point in split_points:
            group_sizes.append(point - prev)
            prev = point
        if prev < len(lengths):
            group_sizes.append(len(lengths) - prev)
        len_groups = lengths.split(group_sizes)
        feature_groups = features.split([int(group.sum().item()) for group in len_groups])
        return feature_groups, len_groups

    @staticmethod
    def _tensor_to_numpy(tensor) -> np.ndarray:
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    @staticmethod
    def _require_torch():
        try:
            import torch
        except ImportError as exc:
            raise ImportError("MiMo-V2.5 audio codec encode requires torch") from exc
        return torch

    @staticmethod
    def _require_torchaudio():
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("MiMo-V2.5 raw audio preprocessing requires torchaudio") from exc
        return torchaudio
