"""Tests for RouterArgs CLI parsing."""

import pytest

from sgl_jax.srt.disaggregation.router_args import RouterArgs


def _parse(argv: list[str]) -> RouterArgs:
    import argparse

    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return RouterArgs.from_cli_args(parser.parse_args(argv))


class TestPrefillUrlParsing:
    def test_single_prefill_with_bootstrap_port_space_separated(self):
        args = _parse(["--prefill", "http://p1:30100", "8998"])
        assert args.prefill_urls == [("http://p1:30100", 8998)]

    def test_single_prefill_with_bootstrap_port_comma_separated(self):
        args = _parse(["--prefill", "http://p1:30100,8998"])
        assert args.prefill_urls == [("http://p1:30100", 8998)]

    def test_single_prefill_without_bootstrap_port(self):
        args = _parse(["--prefill", "http://p1:30100"])
        assert args.prefill_urls == [("http://p1:30100", None)]

    def test_prefill_bootstrap_port_none_keyword(self):
        args = _parse(["--prefill", "http://p1:30100", "none"])
        assert args.prefill_urls == [("http://p1:30100", None)]

    def test_prefill_bootstrap_port_none_keyword_comma(self):
        args = _parse(["--prefill", "http://p1:30100,none"])
        assert args.prefill_urls == [("http://p1:30100", None)]

    def test_multiple_prefill_urls(self):
        args = _parse([
            "--prefill", "http://p1:30100", "8998",
            "--prefill", "http://p2:30100", "9998",
        ])
        assert args.prefill_urls == [
            ("http://p1:30100", 8998),
            ("http://p2:30100", 9998),
        ]


class TestDecodeUrlParsing:
    def test_single_decode_url(self):
        args = _parse(["--decode", "http://d1:30200"])
        assert args.decode_urls == ["http://d1:30200"]

    def test_multiple_decode_urls(self):
        args = _parse([
            "--decode", "http://d1:30200",
            "--decode", "http://d2:30200",
        ])
        assert args.decode_urls == ["http://d1:30200", "http://d2:30200"]

    def test_no_decode_urls(self):
        args = _parse([])
        assert args.decode_urls == []


class TestOtherArgs:
    def test_defaults(self):
        args = _parse([])
        assert args.host == "0.0.0.0"
        assert args.port == 30000
        assert args.mini_lb is False
        assert args.pd_disaggregation is False
        assert args.policy == "random"
        assert args.request_timeout_secs == 1800
        assert args.prefill_bootstrap_host is None

    def test_override_host_port(self):
        args = _parse(["--host", "127.0.0.1", "--port", "8080"])
        assert args.host == "127.0.0.1"
        assert args.port == 8080

    def test_flags(self):
        args = _parse(["--mini-lb", "--pd-disaggregation", "--test-external-dp-routing"])
        assert args.mini_lb is True
        assert args.pd_disaggregation is True
        assert args.test_external_dp_routing is True

    def test_prefill_bootstrap_host_override(self):
        args = _parse(["--prefill-bootstrap-host", "10.0.0.1"])
        assert args.prefill_bootstrap_host == "10.0.0.1"
