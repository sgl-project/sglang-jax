import jax
import numpy as np


class DeviceManager:
    def __init__(self):
        self.devices = jax.devices()
        self.allocatable = np.arange(len(jax.devices()))

    def allocate(self, num_tpus: int):
        if len(self.allocatable) < num_tpus:
            raise Exception("device is not enough")
        current_devices = self.allocatable[:num_tpus]
        self.allocatable = self.allocatable[num_tpus:]
        return current_devices


device_manager = DeviceManager()
