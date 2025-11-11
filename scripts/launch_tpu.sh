#!/bin/bash

# Check if correct number of arguments provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <accelerator> <ref> [test_type]"
    echo "Example: $0 tpu-v6e-1 main"
    echo "Example: $0 tpu-v6e-1 main e2e"
    echo "Example: $0 tpu-v6e-1 main performance"
    exit 1
fi

# Get arguments
ACCELERATOR="$1"
REF="$2"
TEST_TYPE="${3:-}"

# Validate arguments
if [ -z "$ACCELERATOR" ]; then
    echo "Error: Accelerator type cannot be empty"
    exit 1
fi

if [ -z "$REF" ]; then
    echo "Error: ref name cannot be empty"
    exit 1
fi

# Function to sanitize ref name for cluster naming
sanitize_ref() {
    local ref="$1"
    # Remove refs/heads/ prefix if present
    ref="${ref#refs/heads/}"
    # Replace invalid characters with dash
    ref=$(echo "$ref" | sed 's/[^a-zA-Z0-9._-]/-/g')
    # Remove leading non-letter characters
    ref=$(echo "$ref" | sed 's/^[^a-zA-Z]*//')
    # Ensure it doesn't end with dash, dot or underscore
    ref=$(echo "$ref" | sed 's/[-._]*$//')
    # Truncate to reasonable length (max 20 chars to leave room for prefix/suffix)
    ref=$(echo "$ref" | cut -c1-20)
    # If empty after sanitization, use "default"
    if [ -z "$ref" ]; then
        ref="default"
    fi
    echo "$ref"
}

# Sanitize ref name
SANITIZED_REF=$(sanitize_ref "$REF")

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a temporary rendered yaml file
TEMP_YAML="${SCRIPT_DIR}/tpu_resource_rendered.sky.yaml"

# Read the template and replace variables
# Use | as delimiter to handle slashes in branch names
sed -e "s|\$ACCELERATOR|${ACCELERATOR}|g" \
    -e "s|\$REF|${REF}|g" \
    "${SCRIPT_DIR}/tpu_resource.sky.yaml" > "$TEMP_YAML"

# Create cluster name with ref
if [ -n "$TEST_TYPE" ]; then
    CLUSTER_NAME="sgl-jax-$ACCELERATOR-$SANITIZED_REF-$TEST_TYPE-$RANDOM"
else
    CLUSTER_NAME="sgl-jax-$ACCELERATOR-$SANITIZED_REF-$RANDOM"
fi

# Save cluster name to file for other scripts to use
echo "$CLUSTER_NAME" > "${SCRIPT_DIR}/../.cluster_name"

# Execute sky launch command
echo ""
echo "Executing command with:"
echo "  Accelerator: ${ACCELERATOR}"
echo "  Ref: ${REF}"
echo "  Sanitized Ref: ${SANITIZED_REF}"
echo "  Test Type: ${TEST_TYPE:-'(none)'}"
echo "  Cluster Name: ${CLUSTER_NAME}"
echo ""

sky launch "$TEMP_YAML" \
    --cluster="$CLUSTER_NAME" \
    --infra=gcp \
    -i 10 \
    --down \
    -y

# Store the exit code
EXIT_CODE=$?

# Clean up temporary file
rm -f "$TEMP_YAML"

# Check if launch command was successful
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Sky launch command failed"
    exit $EXIT_CODE
fi

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

echo "Waiting for 10 seconds, then submit job to the cluster"
sleep 10