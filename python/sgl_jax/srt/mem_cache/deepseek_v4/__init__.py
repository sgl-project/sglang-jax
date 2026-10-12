"""DeepSeek V4 cache resources: storage, compressor state and host allocation.

The runtime invokes the resource lifecycle interfaces and commits functional
pool updates. State uses global request slots; KV addresses are rank-local.
"""
