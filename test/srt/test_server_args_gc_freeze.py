# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
