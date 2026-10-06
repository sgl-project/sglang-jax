# PD transfer manual tests

These scripts exercise real transfer behavior and are intentionally not wired
into CI yet.

## Files

- `pd_transfer_smoke.py`
  - one request, one dtype, one byte-equality check
  - intended as the default bring-up command when validating a new cluster or build
- `pd_transfer_matrix.py`
  - broader manual matrix over multiple dtypes, sizes, and path-A/path-B behavior
- `pd_transfer_probe.py`
  - targeted diagnostic for the transfer readback issue seen with JAX 0.8.1

## Usage

Smoke, prefill side:

```bash
python test/srt/multi_host/disaggregation/pd_transfer_smoke.py \
  --role prefill --my-host $(hostname -i) --ctl-port 31000 \
  --transfer-port 31001 --side-channel-port 31002
```

Smoke, decode side:

```bash
python test/srt/multi_host/disaggregation/pd_transfer_smoke.py \
  --role decode --my-host $(hostname -i) --remote <prefill-host-ip> \
  --ctl-port 31000 --transfer-port 31001 --side-channel-port 31002
```

Expected result:

- decode prints `PASS byte-equal transfer and ack completed`
- prefill prints `PASS sender reached SUCCESS and cleanup completed`
- both processes exit `0`

## Hybrid (linear-recurrent) models

Dual-payload smoke: add `--hybrid` to **both** commands above (optionally with
`--use-d2h-staging` on prefill). The payload then also carries one request's
`recurrent_state` / `conv_state` gathered from a real `RecurrentStatePool`
(Qwen3.5 GDN shapes by default: `--recurrent-num-heads 16
--recurrent-head-dim 128 --conv-kernel-size 4`). Decode scatters it into a
different slot of its own pool and checks that slot byte-for-byte, and that
no other slot changed.

End-to-end generation parity (`pd_hybrid_generation_parity.py`): start a
monolithic server and a PD deployment (1 router + 1 prefill + 1 decode) with
the same hybrid model and parallel layout (e.g. `--dp-size 2`), then run:

```bash
python test/srt/multi_host/disaggregation/pd_hybrid_generation_parity.py \
  --baseline-url http://<mono-host>:30000 --pd-url http://<router-host>:8000 \
  --model-path <hf-model-path> --page-size <page-size>
```

It sends greedy requests to both, including prompt lengths where `L-1` is a
page multiple. It requires identical output tokens and per-token logprobs
within `1e-5`.
