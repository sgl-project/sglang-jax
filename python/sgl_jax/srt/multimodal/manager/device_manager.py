import jax
import numpy as np


class DeviceManager:
    """Simple manager for available JAX devices and allocation tracking.

    This lightweight utility exposes the list of available JAX devices and
    keeps a simple integer-based pool (`allocatable`) for allocating a
    requested number of devices (by index). It is intentionally minimal and
    meant for scheduling tests and small pipelines; it does not perform
    advanced reservations, release bookkeeping, or concurrency control.
    """

    def __init__(self):
        """Capture the current JAX devices and initialize the allocatable pool.

        `self.devices` is the list returned by `jax.devices()` and
        `self.allocatable` is a NumPy array of integer indices representing
        devices that can be allocated.
        """

        self.devices = jax.devices()
        self.allocatable = np.arange(len(jax.devices()))

    def allocate(self, num_tpus: int):
        """Allocate `num_tpus` devices from the allocatable pool.

        Returns an array of integer indices representing allocated devices.
        Raises an Exception if there are not enough free devices.
        """

        if len(self.allocatable) < num_tpus:
            raise Exception("device is not enough")
        current_devices = self.allocatable[:num_tpus]
        self.allocatable = self.allocatable[num_tpus:]
        return current_devices
