# first copy python directory using rsync

target_cluster_file=$1
if [ -z "$target_cluster_file" ]; then
    echo "Error: target cluster file is not set"
    exit 1
fi
target_cluster=$(cat $target_cluster_file)

# require argument command to run
task_file="$2" #yaml file

if [ -z "$task_file" ]; then
    echo "Error: task_file is not set"
    exit 1
fi

# run the command on the cluster
sky exec ${target_cluster} ${task_file}