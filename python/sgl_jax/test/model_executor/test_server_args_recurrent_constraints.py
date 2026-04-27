"""ServerArgs additions for hybrid recurrent state: --state-to-kv-ratio."""

import unittest


class TestStateToKvRatioCli(unittest.TestCase):
    def test_default_value_is_zero_point_nine(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(model_path="dummy")
        self.assertEqual(args.state_to_kv_ratio, 0.9)

    def test_can_be_overridden(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(model_path="dummy", state_to_kv_ratio=0.5)
        self.assertEqual(args.state_to_kv_ratio, 0.5)

    def test_argparse_registers_state_to_kv_ratio(self):
        import argparse

        from sgl_jax.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        # Parse a minimal arg set + the new flag.
        ns = parser.parse_args(["--model-path", "dummy", "--state-to-kv-ratio", "0.75"])
        self.assertEqual(ns.state_to_kv_ratio, 0.75)


if __name__ == "__main__":
    unittest.main()
