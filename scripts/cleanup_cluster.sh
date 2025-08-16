#!/bin/bash

# Function to sanitize ref name for cluster naming (same as in launch_tpu.sh)
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
    # Truncate to reasonable length (max 40 chars to leave room for prefix/suffix)
    ref=$(echo "$ref" | cut -c1-40)
    # If empty after sanitization, use "default"
    if [ -z "$ref" ]; then
        ref="default"
    fi
    echo "$ref"
}

# Check if correct number of arguments provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <accelerator> <ref> [test_type]"
    echo "Example: $0 tpu-v6e-1 main"
    echo "Example: $0 tpu-v6e-1 main e2e"
    echo "Example: $0 tpu-v6e-1 main performance"
    exit 1
fi

ACCELERATOR="$1"
REF="$2"
TEST_TYPE="${3:-}"
SANITIZED_REF=$(sanitize_ref "$REF")

if [ -n "$TEST_TYPE" ]; then
    CLUSTER_NAME="sgl-jax-ci-$ACCELERATOR-$SANITIZED_REF-$TEST_TYPE"
else
    CLUSTER_NAME="sgl-jax-ci-$ACCELERATOR-$SANITIZED_REF"
fi

echo "Checking cluster status for: $CLUSTER_NAME"

# Check if cluster exists and get its status
CLUSTER_STATUS=$(sky status --refresh 2>/dev/null | grep "^$CLUSTER_NAME" -q && \
  (sky status --refresh 2>/dev/null | grep "^$CLUSTER_NAME" | awk '{print $5}') || \
  echo "NOT_FOUND")

echo "Cluster status: $CLUSTER_STATUS"

case "$CLUSTER_STATUS" in
    "UP")
        echo "Cluster is UP and running. Skipping cleanup to avoid disrupting active work."
        exit 0
        ;;
    "STOPPED"|"INIT"|"LAUNCHING")
        echo "Cluster is in $CLUSTER_STATUS state. Proceeding with cleanup..."
        sky down "$CLUSTER_NAME" -y
        if [ $? -eq 0 ]; then
            echo "Successfully cleaned up cluster: $CLUSTER_NAME"
        else
            echo "Failed to clean up cluster: $CLUSTER_NAME"
            exit 1
        fi
        ;;
    "NOT_FOUND"|"")
        echo "Cluster not found or already cleaned up: $CLUSTER_NAME"
        exit 0
        ;;
    *)
        echo "Unknown cluster status: $CLUSTER_STATUS. Attempting cleanup..."
        sky down "$CLUSTER_NAME" -y || true
        ;;
esac

echo "Cleanup completed for cluster: $CLUSTER_NAME"
