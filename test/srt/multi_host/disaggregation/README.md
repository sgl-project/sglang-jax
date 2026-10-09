# PD transfer manual tests

These scripts exercise real transfer behavior and are intentionally not wired
into CI yet.

## Files

- `pd_transfer_smoke.py`
  - one request, one dtype (bf16), one byte-equality check
  - `--engine jax_transfer` (default) or `--engine raiden`
  - with `--hybrid`, also transfers one request's recurrent/conv state
  - intended as the default bring-up command when validating a new cluster or build
- `pd_transfer_matrix.py`
  - broader manual matrix over dtypes (bf16, fp16, fp8_e4m3fn), page counts,
    and path-A (D2H staging) / path-B (direct from HBM)
  - jax_transfer only
- `pd_transfer_probe.py`
  - targeted diagnostic for the transfer readback issue seen with JAX 0.8.1
  - jax_transfer only
- `pd_hybrid_generation_parity.py`
  - end-to-end generation parity between a PD deployment and a monolithic server
  - works with either engine; `--expect-engine` checks which one the PD servers run

## Prerequisites

- `sgl_jax` installed on every host (e.g. `pip install -e python` from the repo root).
- Run the commands from the repo root and invoke the scripts by path (not
  `python -m`): `pd_transfer_smoke.py` imports helpers from
  `pd_transfer_matrix.py` in the same directory.
- Decode must reach prefill on the control, transfer and side-channel ports
  (the side channel carries decode's pull-done ack to prefill). With
  `--engine raiden`, it must reach the control, transfer and bootstrap ports.
- Set `MY_IP` on each host to an address the other host can reach, IPv4 or
  IPv6, and use the prefill host's `MY_IP` as decode's `--remote`:

  ```bash
  MY_IP=$(hostname -I | awk '{print $1}')   # first non-loopback address
  ```

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
  handshake. jax_transfer only.
- `--hybrid`: pass on **both** sides. See [Hybrid models](#hybrid-linear-recurrent-models).
- Defaults: `--ctl-port 31000 --transfer-port 31001 --side-channel-port 31002`.
  `--my-host` is required on both roles; `--remote` is required on decode.

### Raiden engine (`--engine raiden`)

Pass `--engine raiden` on **both** sides. The script preloads the Raiden
library before importing JAX, registers a small paged KV cache with a
`RaidenTransferWrapper` on each side, and drives the real
`RaidenTransferKVManager` sender/receiver through a `BootstrapServer` that
prefill starts.

```bash
# prefill
python test/srt/multi_host/disaggregation/pd_transfer_smoke.py \
  --role prefill --engine raiden --my-host $MY_IP \
  --ctl-port 31000 --transfer-port 31001 --bootstrap-port 8998

# decode
python test/srt/multi_host/disaggregation/pd_transfer_smoke.py \
  --role decode --engine raiden --my-host $MY_IP --remote <prefill-host-ip> \
  --ctl-port 31000 --transfer-port 31001 --bootstrap-port 8998
```

- Ports:
  - `--ctl-port`: the script's own control channel.
  - `--transfer-port`: the Raiden control port on each host.
  - `--bootstrap-port`: the bootstrap server on prefill. It listens on `::`;
    decode only learns the port from the handshake.
  - `--side-channel-port` is ignored.
- `--use-d2h-staging` is rejected: Raiden reads directly from HBM.
- Prefill pages `1,2,3` are read into decode pages `4,5,6`. Decode checks those
  pages byte-for-byte and checks that every other page is still zero.
- Success: decode prints `[D] PASS byte-equal transfer (kv) via raiden`
  (`(kv + recurrent/conv state)` with `--hybrid`), prefill prints
  `[P] PASS sender reached SUCCESS (raiden)`, and both exit `0`.

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
that no other slot changed. `--hybrid` works with `--engine raiden` too. There,
the recurrent/conv buffers are registered through
`RaidenTransferWrapper.start_recurrent`, and the slot is read as part of the
same request.

For end-to-end generation with a real hybrid model, see
[Generation parity](#generation-parity-pd_hybrid_generation_paritypy).

## Generation parity (`pd_hybrid_generation_parity.py`)

Checks that PD disaggregation does not change model output. It sends the same
greedy (`temperature=0`) requests to a monolithic server (`--baseline-url`) and
to a PD deployment behind a load balancer (`--pd-url`), then requires identical
output token ids and per-token logprobs within `2e-2` (`--logprob-atol`).
The returned logprobs are bf16-quantized (steps of ~`7.8e-3` near 0), and PD
re-runs the last prompt token as a separate 1-token extend, so a logprob on a
rounding boundary can flip one step without any token changing. The default
allows two steps; pass a smaller `--logprob-atol` to demand closer agreement.
With `--model-path`, the prompts also include token lengths where `L-1` is a
page multiple, which exercises the "prefill skips the last prompt token" path
for hybrid models.

Before sending prompts, it asks the load balancer's `/get_server_info` which
mode and engine each PD server runs. It fails if a server's mode does not match
its role or, with `--expect-engine`, if a server runs a different engine.

### Topology

Four components, all serving the **same model** with the **same** `--tp-size`,
`--dp-size` and `--page-size`:

| Component | Host | Port |
| --- | --- | --- |
| Monolithic server (baseline) | its own TPU host | 30000 |
| Bootstrap server + prefill server | prefill host | 8998, 10000 |
| Decode server | decode host | 10001 |
| Load balancer (router) | any host; decode host below | 8000 |

The baseline and the PD deployment must be up at the same time, and each
server needs its own TPU chips. That means three TPU hosts, or a baseline with
the same layout on spare chips.

### Bring up the servers

Set these on every host (the IPs below are examples; use each host's routable
address, see [Prerequisites](#prerequisites)):

```bash
MODEL=<hf-model-path>               # same path on every host
TP=8; DP=1; PAGE=128
P_IP=1234:1234:1234:1234::4         # prefill host
D_IP=1234:1234:1234:1234::8         # decode host (also runs the router)
COMMON=(--model-path "$MODEL" --tp-size "$TP" --dp-size "$DP"
        --page-size "$PAGE" --disable-radix-cache)
PD=(--disaggregation-use-raiden
    --disaggregation-bootstrap-url "http://[$P_IP]:8998"
    --disaggregation-max-inflight-transfers 2)
```

1. Monolithic server (baseline host):

   ```bash
   python -m sgl_jax.launch_server "${COMMON[@]}" --host :: --port 30000
   ```

2. Bootstrap server and prefill server (prefill host):

   ```bash
   python -m sgl_jax.srt.disaggregation.run_bootstrap --host :: --port 8998 &
   python -m sgl_jax.launch_server "${COMMON[@]}" "${PD[@]}" \
     --disaggregation-mode prefill --disaggregation-host-ip "$P_IP" \
     --host :: --port 10000
   ```

3. Decode server (decode host):

   ```bash
   python -m sgl_jax.launch_server "${COMMON[@]}" "${PD[@]}" \
     --disaggregation-mode decode --disaggregation-host-ip "$D_IP" \
     --host :: --port 10001
   ```

4. Load balancer (decode host, once prefill and decode report healthy, e.g.
   `curl -g "http://[$P_IP]:10000/health"`):

   ```bash
   python -m sgl_jax.srt.disaggregation.launch_router \
     --pd-disaggregation --mini-lb \
     --prefill "http://[$P_IP]:10000" 8998 \
     --decode "http://[$D_IP]:10001" \
     --host :: --port 8000
   ```

Notes:

- `--disaggregation-host-ip` is the address each PD server publishes to its
  peer. Without it, the server resolves `$HOSTNAME` with an IPv4-only lookup,
  which fails on IPv6-only hosts or yields a rejected loopback address.
- On IPv4 hosts, use `--host 0.0.0.0` and unbracketed IPs in URLs.
- Raiden requires TPU and `--disable-radix-cache`, and cannot be combined with
  D2H staging. Hybrid models cannot use
  `--disaggregation-enable-chunk-prefill-transfer` with Raiden.
- For jax_transfer, drop `--disaggregation-use-raiden`; decode must then also
  reach prefill on `--disaggregation-transfer-port` and
  `--disaggregation-side-channel-port` (default `9600`).
- `scripts/disaggregation/run_raiden_smoke.sh` automates steps 2–4
  (`ROLE=prefill` / `ROLE=decode`, router on port 30000). It brackets IPv6 and
  binds to `::` when `PREFILL_HOST` is IPv6, but it does not pass
  `--disaggregation-host-ip`.

### Run the parity check

From the repo root on any host that can reach both deployments (`--model-path`
needs the tokenizer locally, via `transformers`):

```bash
M_IP=<baseline-host-ip>
python test/srt/multi_host/disaggregation/pd_hybrid_generation_parity.py \
  --baseline-url "http://[$M_IP]:30000" --pd-url "http://[$D_IP]:8000" \
  --model-path "$MODEL" --page-size "$PAGE" --expect-engine raiden
```

Expected output:

```text
[engine] prefill: disaggregation_mode=prefill engine=raiden
[engine] decode: disaggregation_mode=decode engine=raiden
[PASS] #0 len=24 chars: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
...
[PASS] #4 len=128 ids: 32 tokens, max |logprob diff|=7.721e-03, 1/32 positions differ
...
11/11 prompts at parity
```

It exits `0` only if every prompt is at parity. A token mismatch prints the
first differing position and both id lists. A logprob failure prints the
output index (and, for token-id prompts, the absolute position) of the largest
difference with the token and both logprobs.
