# PD transfer manual tests

These scripts exercise real transfer behavior and are intentionally not wired
into CI yet.

## Files

- `pd_transfer_smoke.py`
  - one request, one dtype (bf16), one byte-equality check
  - with `--hybrid`, also transfers one request's recurrent/conv state
  - intended as the default bring-up command when validating a new cluster or build
- `pd_transfer_matrix.py`
  - broader manual matrix over dtypes (bf16, fp16, fp8_e4m3fn), page counts,
    and path-A (D2H staging) / path-B (direct from HBM)
- `pd_transfer_probe.py`
  - targeted diagnostic for the transfer readback issue seen with JAX 0.8.1
- `pd_hybrid_generation_parity.py`
  - end-to-end generation parity between a PD deployment and a monolithic server

## Prerequisites

- `sgl_jax` installed on every host (e.g. `pip install -e python` from the repo root).
- Run the commands from the repo root and invoke the scripts by path (not
  `python -m`): `pd_transfer_smoke.py` imports helpers from
  `pd_transfer_matrix.py` in the same directory.
- Decode must reach prefill on the control, transfer and side-channel ports
  (the side channel carries decode's pull-done ack to prefill).
- Set `MY_IP` on each host to an address the other host can reach, IPv4 or
  IPv6, and use the prefill host's `MY_IP` as decode's `--remote`:

  ```bash
  MY_IP=$(hostname -I | awk '{print $1}')   # first non-loopback address
  ```

  Do not use `$(hostname -i)`: on many images it resolves to the loopback
  alias `127.0.1.1`, so prefill only listens on loopback and decode times out
  in `_connect`. The smoke script prints a `[warn]` if `--my-host` is loopback.
  `pd_transfer_smoke.py` and `pd_transfer_matrix.py` accept IPv6 addresses
  (e.g. `--remote fb12:3456:789a:8002::4`); `pd_transfer_probe.py` is IPv4-only.

## Smoke test (`pd_transfer_smoke.py`)

1. Start prefill. It prints `[ctl] waiting for D on <host>:<ctl-port>`.

   ```bash
   python test/srt/multi_host/disaggregation/pd_transfer_smoke.py \
     --role prefill --my-host $MY_IP --ctl-port 31000 \
     --transfer-port 31001 --side-channel-port 31002
   ```

2. Start decode, pointing `--remote` at the prefill host. It retries the
   control connection for up to 120s.

   ```bash
   python test/srt/multi_host/disaggregation/pd_transfer_smoke.py \
     --role decode --my-host $MY_IP --remote <prefill-host-ip> \
     --ctl-port 31000 --transfer-port 31001 --side-channel-port 31002
   ```

3. Check the result:
   - decode prints `[D] PASS byte-equal transfer (kv) and ack completed`
     (`(kv + recurrent/conv state)` with `--hybrid`)
   - prefill prints `[P] PASS sender reached SUCCESS and cleanup completed`
   - both processes exit `0`

Options:

- `--use-d2h-staging`: pass on **prefill only** to exercise path A (D2H staging
  through `QueueHostKVPool`). Decode learns the path from the control-channel
  handshake.
- `--hybrid`: pass on **both** sides. See [Hybrid models](#hybrid-linear-recurrent-models).
- Defaults: `--ctl-port 31000 --transfer-port 31001 --side-channel-port 31002`.
  `--my-host` is required on both roles; `--remote` is required on decode.

## Matrix (`pd_transfer_matrix.py`)

Same prefill/decode flow as the smoke test:

```bash
# prefill (add --use-d2h-staging for path A)
python test/srt/multi_host/disaggregation/pd_transfer_matrix.py \
  --role prefill --my-host $MY_IP --ctl-port 31000 \
  --transfer-port 31001 --side-channel-port 31002

# decode
python test/srt/multi_host/disaggregation/pd_transfer_matrix.py \
  --role decode --my-host $MY_IP --remote <prefill-host-ip> \
  --ctl-port 31000 --transfer-port 31001 --side-channel-port 31002
```

Note that `--side-channel-port` defaults to `9600` here. Both sides print a
final `[P] done: ...` / `[D] done: ...` summary and exit `0` only if every cell
is byte-equal and (path A) no host-pool buffers leaked. Sizes and iteration
counts can be tuned through environment variables, e.g. `PD_PAGE_COUNTS=1,16`,
`PD_ROUNDTRIP_ITERS=8`, `PD_PAGE_SIZE_TOKENS=128`, `PD_POOL_SIZE=16`,
`PD_KV_LAYER_NUM=36` (see the top of the script for the full list).

## Probe (`pd_transfer_probe.py`)

```bash
# producer
python test/srt/multi_host/disaggregation/pd_transfer_probe.py \
  --role producer --my-host $MY_IP --ctl-port 31000 --transfer-port 31001

# consumer
python test/srt/multi_host/disaggregation/pd_transfer_probe.py \
  --role consumer --my-host $MY_IP --remote <producer-host-ip> \
  --ctl-port 31000 --transfer-port 31001
```

The consumer prints which readback strategies succeed and which raise.

## Hybrid (linear-recurrent) models

Dual-payload smoke: add `--hybrid` to **both** smoke commands above (optionally
with `--use-d2h-staging` on prefill). The payload then also carries one
request's `recurrent_state` / `conv_state` gathered from a real
`RecurrentStatePool` (Qwen3.5 GDN shapes by default: `--recurrent-layers 2
--recurrent-num-heads 16 --recurrent-head-dim 128 --conv-kernel-size 4`; if you
override any of these, use the same values on both sides). Decode scatters it
into a different slot of its own pool and checks that slot byte-for-byte, and
that no other slot changed.

End-to-end generation parity (`pd_hybrid_generation_parity.py`): start a
monolithic server and a PD deployment (1 router + 1 prefill + 1 decode) with
the same hybrid model and parallel layout (e.g. `--dp-size 2`), then run:

```bash
python test/srt/multi_host/disaggregation/pd_hybrid_generation_parity.py \
  --baseline-url http://<mono-host>:30000 --pd-url http://<router-host>:8000 \
  --model-path <hf-model-path> --page-size <page-size>
```

It sends greedy requests to both. With `--model-path` set, the prompts also
include lengths where `L-1` is a page multiple (`--page-size`, default `64`).
It requires identical output tokens and per-token logprobs within `1e-5`
(`--logprob-atol`), and exits `0` on full parity.
