"""Render a self-contained Falcon manifest from a pinned base plus local patch."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = "24ef4d3aa5de85246d0c1ee09572784e879000e4"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", required=True, help="Unique Falcon experiment name")
parser.add_argument("--output", type=Path, required=True, help="Destination YAML")
args = parser.parse_args()
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
patch = subprocess.check_output(
    ["git", "diff", "--binary", BASE, "--", "python", "test", "docs/features"], cwd=ROOT
)
files = {
    name: (HERE / name).read_bytes()
    for name in [
        "setup_single_pod.sh",
        "single_pod_pd.py",
        "split_chip_probe.py",
        "lifecycle_checks.py",
        "session_supervisor.py",
        "fault_suite.py",
        "fault_sitecustomize.py",
        "performance_suite.py",
        "stream_regression.py",
        "eos_checks.py",
        "mixed_requests.py",
    ]
}
files["source.patch"] = patch
manifest = {
    "base_commit": BASE,
    "implementation_commit": head,
    "files_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
}
files["manifest.json"] = json.dumps(manifest, indent=2).encode()
payload = gzip.compress(
    json.dumps({k: base64.b64encode(v).decode() for k, v in files.items()}).encode()
)
encoded = base64.b64encode(payload).decode()
command = (
    """set -eu
mkdir -p /tmp/pd-payload /workspace/sglang-jax
python3 - <<'PY_PAYLOAD'
import base64, gzip, hashlib, json
from pathlib import Path
payload = base64.b64decode('__PAYLOAD__')
assert hashlib.sha256(payload).hexdigest() == '__HASH__'
root = Path('/tmp/pd-payload')
for name, content in json.loads(gzip.decompress(payload)).items():
    (root / name).write_bytes(base64.b64decode(content))
manifest = json.loads((root / 'manifest.json').read_text())
for name, digest in manifest['files_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
PY_PAYLOAD
cd /workspace/sglang-jax
git init
git remote add origin https://github.com/sgl-project/sglang-jax.git
git fetch --depth 1 origin __BASE__
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = __BASE__
git apply --check /tmp/pd-payload/source.patch
git apply /tmp/pd-payload/source.patch
mkdir -p scripts/disaggregation/falcon
cp /tmp/pd-payload/*.py /tmp/pd-payload/*.sh scripts/disaggregation/falcon/
timeout --signal=TERM --kill-after=90s 21600 bash scripts/disaggregation/falcon/setup_single_pod.sh
""".replace(
        "__PAYLOAD__", encoded
    )
    .replace("__HASH__", hashlib.sha256(payload).hexdigest())
    .replace("__BASE__", BASE)
)
spec = {
    "name": args.name,
    "exp_type": "TRAINING",
    "artifact_type": "GCS",
    "cluster_name": "tpu-training-antgroup",
    "priority": 0,
    "config": json.dumps(manifest),
    "role_to_task_spec": {
        "worker": {
            "replica": 1,
            "image": "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1",
            "device_count": 8,
            "device_type": "v7x",
            "device_topo": "2x2x1",
            "command": command,
            "mounts": [
                {
                    "name": "models",
                    "type": "gcs",
                    "bucket": "inference-model-storage-poc-tpu-hns",
                    "mount_path": "/models",
                    "read_only": True,
                    "mount_options": ["implicit-dirs"],
                },
                {
                    "name": "raiden-wheel",
                    "type": "gcs",
                    "bucket": "tpu-for-training-falcon-logs",
                    "prefix": "experiments/exp-r4qe9t4zsh/artifacts/art-4i4xt8n9k5",
                    "mount_path": "/raiden-wheel",
                    "read_only": True,
                    "mount_options": ["implicit-dirs"],
                },
            ],
        }
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(yaml.safe_dump(spec, sort_keys=False))
args.output.with_suffix(".source.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"command_bytes": len(command), "source": manifest}, indent=2))
