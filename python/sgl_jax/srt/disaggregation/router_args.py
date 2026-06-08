from __future__ import annotations

import argparse
import dataclasses


@dataclasses.dataclass
class RouterArgs:
    host: str = "0.0.0.0"
    port: int = 30000

    mini_lb: bool = False
    test_external_dp_routing: bool = False
    pd_disaggregation: bool = False
    request_timeout_secs: int = 1800
    policy: str = "random"

    prefill_urls: list[tuple[str, int | None]] = dataclasses.field(
        default_factory=list
    )
    decode_urls: list[str] = dataclasses.field(default_factory=list)

    prefill_bootstrap_host: str | None = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--host", default=RouterArgs.host)
        parser.add_argument("--port", type=int, default=RouterArgs.port)
        parser.add_argument("--mini-lb", action="store_true")
        parser.add_argument(
            "--test-external-dp-routing",
            action="store_true",
        )
        parser.add_argument("--pd-disaggregation", action="store_true")
        parser.add_argument(
            "--policy",
            default=RouterArgs.policy,
            choices=["random"],
        )
        parser.add_argument(
            "--prefill",
            nargs="+",
            action="append",
            help="Prefill URL and optional bootstrap port. "
            "Format: --prefill URL [BOOTSTRAP_PORT].",
        )
        parser.add_argument(
            "--decode",
            nargs=1,
            action="append",
            metavar=("URL",),
            help="Decode URL. Can be specified multiple times.",
        )
        parser.add_argument(
            "--request-timeout-secs",
            type=int,
            default=RouterArgs.request_timeout_secs,
        )
        parser.add_argument(
            "--prefill-bootstrap-host",
            default=RouterArgs.prefill_bootstrap_host,
            help="Override bootstrap_host injected into forwarded requests. "
            "Useful when the router talks to prefill via localhost but "
            "decode must see the pod IP.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> RouterArgs:
        return cls(
            host=args.host,
            port=args.port,
            mini_lb=args.mini_lb,
            test_external_dp_routing=args.test_external_dp_routing,
            pd_disaggregation=args.pd_disaggregation,
            request_timeout_secs=args.request_timeout_secs,
            policy=args.policy,
            prefill_urls=cls._parse_prefill_urls(
                getattr(args, "prefill", None)
            ),
            decode_urls=cls._parse_decode_urls(
                getattr(args, "decode", None)
            ),
            prefill_bootstrap_host=getattr(
                args, "prefill_bootstrap_host", None
            ),
        )

    @staticmethod
    def _parse_prefill_urls(
        prefill_list: list[list[str]] | None,
    ) -> list[tuple[str, int | None]]:
        if not prefill_list:
            return []

        parsed: list[tuple[str, int | None]] = []
        for prefill_args in prefill_list:
            if len(prefill_args) == 1 and "," in prefill_args[0]:
                url, bootstrap_port = prefill_args[0].rsplit(",", 1)
                parsed.append(
                    (
                        url,
                        None
                        if bootstrap_port.lower() == "none"
                        else int(bootstrap_port),
                    )
                )
                continue

            url = prefill_args[0]
            if len(prefill_args) >= 2:
                bootstrap_port_str = prefill_args[1]
                bootstrap_port = (
                    None
                    if bootstrap_port_str.lower() == "none"
                    else int(bootstrap_port_str)
                )
            else:
                bootstrap_port = None
            parsed.append((url, bootstrap_port))

        return parsed

    @staticmethod
    def _parse_decode_urls(
        decode_list: list[list[str]] | None,
    ) -> list[str]:
        if not decode_list:
            return []
        return [url[0] for url in decode_list]
