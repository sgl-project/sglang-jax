import argparse

from sgl_jax.srt.server_args import ServerArgs


def test_enable_gc_freeze_flag_parsing():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model", "--enable-gc-freeze"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.enable_gc_freeze is True


def test_enable_gc_freeze_default_false():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.enable_gc_freeze is False


def test_gc_freeze_rollback_flag_parsing():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model", "--gc-freeze-rollback"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.gc_freeze_rollback is True


def test_gc_freeze_rollback_default_false():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.gc_freeze_rollback is False


def test_enable_tokenizer_batch_send_flag_parsing():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model", "--enable-tokenizer-batch-send"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.enable_tokenizer_batch_send is True


def test_enable_tokenizer_batch_send_default_false():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.enable_tokenizer_batch_send is False
