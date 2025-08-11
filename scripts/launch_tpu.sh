#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <accelerator> <ref>"
    echo "Example: $0 tpu-v6e-1 main"
    exit 1
fi

# Get arguments
ACCELERATOR="$1"
REF="$2"

# Validate arguments
if [ -z "$ACCELERATOR" ]; then
    echo "Error: Accelerator type cannot be empty"
    exit 1
fi

if [ -z "$REF" ]; then
    echo "Error: ref name cannot be empty"
    exit 1
fi

# Check environment variables
if [ -z "$USERNAME" ]; then
    echo "Error: USERNAME environment variable is not set"
    exit 1
fi

if [ -z "$GIT_TOKEN" ]; then
    echo "Error: GIT_TOKEN environment variable is not set"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a temporary rendered yaml file
TEMP_YAML="${SCRIPT_DIR}/tpu_resource_rendered.yaml"

# Read the template and replace variables
# Use | as delimiter to handle slashes in branch names
sed -e "s|\$ACCELERATOR|${ACCELERATOR}|g" \
    -e "s|\$REF|${REF}|g" \
    "${SCRIPT_DIR}/tpu_resource.yaml" > "$TEMP_YAML"

# Execute sky launch command
echo ""
echo "Executing command with:"
echo "  Accelerator: ${ACCELERATOR}"
echo "  Ref: ${REF}"
echo ""

sky launch "$TEMP_YAML" \
    --cluster=sgl-jax-ci-$ACCELERATOR \
    --infra=gcp \
    -i 15 \
    --down \
    --async \
    -y \
    --secret USERNAME=${USERNAME} \
    --secret GIT_TOKEN=${GIT_TOKEN}

# Store the exit code
EXIT_CODE=$?

# Clean up temporary file
rm -f "$TEMP_YAML"

# Check if launch command was successful
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Sky launch command failed"
    exit $EXIT_CODE
fi

# Define cluster name
CLUSTER_NAME="sgl-jax-ci-$ACCELERATOR"

# Wait for cluster to be UP
echo ""
echo "Waiting for cluster $CLUSTER_NAME to be UP..."

TIMEOUT=600  # 600 seconds = 10 minutes
START_TIME=$(date +%s)

while true; do
    # Check if cluster is UP
    if sky status --refresh | grep "^$CLUSTER_NAME" | grep -q "UP"; then
        echo "Success: Cluster $CLUSTER_NAME is UP"
        exit 0
    fi
    
    # Check timeout
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "Error: Timeout waiting for cluster to be UP (waited ${TIMEOUT} seconds)"
        # Show current status for debugging
        echo "Current status:"
        sky status --refresh | grep "^$CLUSTER_NAME" || echo "Cluster not found in status"
        exit 1
    fi
    
    # Show progress
    echo "Checking status... (elapsed: ${ELAPSED}s / ${TIMEOUT}s)"
    
    # Wait before checking again
    sleep 10
done