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
