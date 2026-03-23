# FusedMoE kernel 走读

e\_sem\_id 在架构里扮演什么角色？

*   e\_sem\_id 是 expert 维度上的 ping-pong id（0/1），用来复用/区分两套通信缓冲和 DMA semaphore：

    *   a2a\_s\_x2\_vmem\[e\_sem\_id\]：scatter 后给当前 expert 的输入 token buffer

    *   a2a\_s\_acc\_x2\_vmem\[e\_sem\_id\]：当前 expert 计算后的输出 buffer

    *   send\_sems\[e\_sem\_id\] / recv\_sems\[e\_sem\_id\]：对应的异步通信完成信号


目前的问题:

1.  权重加载没有 overlap
