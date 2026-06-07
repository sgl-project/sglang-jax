import dataclasses
import sys
import types
import unittest

try:
    import zmq  # noqa: F401
except ModuleNotFoundError:
    zmq_stub = types.ModuleType("zmq")
    zmq_stub.Context = object
    zmq_stub.Socket = object
    zmq_stub.SocketType = object
    sys.modules["zmq"] = zmq_stub

try:
    from fastapi.responses import ORJSONResponse  # noqa: F401
except ModuleNotFoundError:
    fastapi_stub = types.ModuleType("fastapi")
    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.ORJSONResponse = object
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.responses"] = responses_stub

from sgl_jax.srt.utils.common_utils import dataclass_to_string_truncated


@dataclasses.dataclass
class _Payload:
    keep: str
    audio_codes: list[int]


@dataclasses.dataclass
class _Request:
    name: str
    payload: _Payload
    nested: dict


class TestDataclassToStringTruncated(unittest.TestCase):
    def test_skip_names_apply_recursively(self):
        obj = _Request(
            name="req",
            payload=_Payload(keep="top", audio_codes=[1, 2, 3]),
            nested={
                "audio_payload": {"codes": [4, 5, 6]},
                "items": [_Payload(keep="nested", audio_codes=[7, 8, 9])],
            },
        )

        text = dataclass_to_string_truncated(
            obj,
            skip_names={"audio_codes", "audio_payload"},
        )

        self.assertIn("keep='top'", text)
        self.assertIn("keep='nested'", text)
        self.assertNotIn("audio_codes", text)
        self.assertNotIn("audio_payload", text)
        self.assertNotIn("[1, 2, 3]", text)
        self.assertNotIn("[7, 8, 9]", text)

    def test_tuple_format_is_preserved_while_recursing(self):
        obj = (_Payload(keep="tuple", audio_codes=[1, 2, 3]),)

        text = dataclass_to_string_truncated(
            obj,
            skip_names={"audio_codes"},
        )

        self.assertEqual(text, "(_Payload(keep='tuple'),)")


if __name__ == "__main__":
    unittest.main()
