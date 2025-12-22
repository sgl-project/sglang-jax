set -e

target_cluster_file=$1

if [ -z "$target_cluster_file" ]; then
    echo "Warning: target cluster file if not set will always be .cluster_name file"
    target_cluster_file=.cluster_name
fi


target_cluster=$(cat $target_cluster_file)
echo "Target cluster: ${target_cluster}"

NUM_WORKERS=$2

if [ -z "$NUM_WORKERS" ]; then
    echo "Warning: NUM_WORKERS is not set, using default NUM_WORKERS 8"
    NUM_WORKERS=8
fi

# first copy to the head node
echo "Copying python directory to ${target_cluster}"
rsync -Pavz --ignore-times test/ ${target_cluster}:/home/gcpuser/sky_workdir/sglang-jax/test/

if [ "$NUM_WORKERS" -gt 1 ]; then
    for i in $(seq 1 $((NUM_WORKERS - 1))); do
        echo "Copying python directory to ${target_cluster}-worker${i}"
        rsync -Pavz --ignore-times python/ ${target_cluster}-worker${i}:/home/gcpuser/sky_workdir/sglang-jax/python/ &
    done
fi