#!/bin/bash
# Run on EVERY host: replace the V4 Flash RAM disk with one big enough for V4.1 (510 GB) and copy the checkpoint from the
# same-region bucket. V4 Flash is also in the bucket, so nothing is lost. ~520 GB of the host's 708 GB RAM.
set -e
SRC=gs://YOUR_BUCKET/DeepSeek-V4.1-Flash-1017/original
DST=/mnt/models/DeepSeek-V4.1-Flash-1017
sudo umount /mnt/models 2>/dev/null || true
sudo mkdir -p /mnt/models
sudo mount -t tmpfs -o size=560G tmpfs /mnt/models
sudo chown $USER /mnt/models
mkdir -p $DST
export CLOUDSDK_STORAGE_PROCESS_COUNT=8 CLOUDSDK_STORAGE_THREAD_COUNT=8
gcloud storage rsync -r --exclude='^\.cache/.*' $SRC $DST
echo "STAGED $(du -sb $DST | cut -f1) bytes"
