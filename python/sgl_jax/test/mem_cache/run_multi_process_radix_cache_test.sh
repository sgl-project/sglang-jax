#!/bin/bash

echo "Starting multi-process RadixCache test..."
echo "Number of processes: 2"
echo "Coordinator address: localhost:12345"
echo "Tensor parallel size: 4"

# Set coordinator address
COORDINATOR_ADDRESS="localhost:12345"
NUM_PROCESSES=2
TP_SIZE=4

# Start process 0 (run in background)
echo "Starting process 0..."
PROCESS_ID=0 NUM_PROCESSES=$NUM_PROCESSES COORDINATOR_ADDRESS=$COORDINATOR_ADDRESS TP_SIZE=$TP_SIZE \
python -m sgl_jax.test.test_multi_process_radix_cache &
PROC0_PID=$!

# Wait a moment for process 0 to start
sleep 2

# Start process 1
echo "Starting process 1..."
PROCESS_ID=1 NUM_PROCESSES=$NUM_PROCESSES COORDINATOR_ADDRESS=$COORDINATOR_ADDRESS TP_SIZE=$TP_SIZE \
python -m sgl_jax.test.test_multi_process_radix_cache

# Wait for process 0 to complete
wait $PROC0_PID

echo "Multi-process test completed!"
