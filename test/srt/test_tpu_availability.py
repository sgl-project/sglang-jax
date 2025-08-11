#!/usr/bin/env python3

import sys
import jax

def test_tpu_availability():
    device_count = jax.device_count()
    devices = jax.devices()
    
    print(f"Device count: {device_count}")
    print(f"Devices: {devices}")
    
    if device_count == 0:
        print("ERROR: No devices found")
        return False
    
    # Check if all devices are TPUs
    device_types = set(device.platform for device in devices)
    
    if 'tpu' not in device_types:
        print(f"ERROR: No TPU devices found. Available device types: {device_types}")
        return False
    
    if len(device_types) > 1:
        print(f"WARNING: Mixed device types found: {device_types}")
    
    print(f"SUCCESS: Found {device_count} TPU device(s)")
    return True

if __name__ == "__main__":
    success = test_tpu_availability()
    sys.exit(0 if success else 1)