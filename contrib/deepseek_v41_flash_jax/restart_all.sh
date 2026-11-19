#!/bin/bash
# Two-phase restart of the V4.1 server on every host (avoids new workers registering with a still-alive old coordinator).
# Usage (from a workstation): ENVS="ROWS=16 SPARSE_MOE=1" bash restart_all.sh
Z=${Z:?set Z (TPU zone)}; P=${P:?set P (GCP project)}; N=${N:?set N (TPU node name)}
gcloud compute tpus tpu-vm ssh $N --zone=$Z --project=$P --worker=all --command='pkill -f "[s]erver.py"; for i in $(seq 1 20); do pgrep -f "[s]erver.py" >/dev/null || break; sleep 1; done; pkill -9 -f "[s]erver.py"; sleep 3; echo "$(hostname | sed "s/.*-w-/w-/") left=$(pgrep -f "[s]erver.py" | wc -l)"' 2>&1 | grep "^w-" | sort | tr '\n' ' '; echo
sleep 5
gcloud compute tpus tpu-vm ssh $N --zone=$Z --project=$P --worker=all --command="[ -f ~/v41_server.log ] && mv -f ~/v41_server.log ~/v41_server.prev.log; (env $ENVS setsid nohup bash ~/v41_jax/run_server.sh > ~/v41_server.log 2>&1 < /dev/null &); echo started" 2>&1 | grep -c started
