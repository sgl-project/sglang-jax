# ring-1T H200-TPUv7对比报告

# 对比设置说明：

1 根据硬件能力各自拉满batch\_size：各自硬件组成单实例，根据硬件能力在不爆kv前提尽可能拉满batch\_size，所测吞吐代表硬件构成的一个实例所能达到的最大吞吐，记为TH，根据公开资料得到单卡-单位时间的成本，结合卡数得到单实例单位时间成本，记为cost。最终比较指标为ROI=TH / cost，该值越大，代表硬件性价比越高。

2 使用相同典型batch\_size：作为补充测试，双方将batch-size拉到相同的一般典型值，对比完全相同设置下的ROI值

补充说明：TPU的主要优势在并发，所以实验二对TPU吃亏更大，仅作补充参考。

实验结果如下表，tpu空白部分待补测。

TPU v7成本 5.4$/h/卡 H200成本 5.3$/h/卡

**以下同样的颜色代表GPU与TPU控制变量，相互比较的2组测试。**

# bfloat16 32卡对比数据

对应32卡单实例，成本为 TPU v7 172.8$/h， H200 169.6$/h

| 测试组 | 显卡 | 对比设置 | batch size | 吞吐 (tokens/s) | ROI |
| --- | --- | --- | --- | --- | --- |
| random长尾<br>in: 1k; out: 31k; random\_ratio: 0.1 | GPU/32卡<br>tp32+attn\_dp4 | 1 根据硬件能力各自拉满batch\_size | 602 | 3047.3 | 17.97 |
|  |  | 2 使用相同典型batch\_size | 256 | 2234.46 | 13.17 |
|  | TPUv7/32卡 | 1 根据硬件能力各自拉满batch\_size | 1024 | 2880.14 | 16.67 |
|  |  | 2 使用相同典型batch\_size | 256 |  |  |
| random非长尾<br>in: 1k; out: 15k; random\_ratio: 0.9 | GPU/32卡<br>tp32+attn\_dp4 | 1 根据硬件能力各自拉满batch\_size | 384 | 4315.42 | 25.44 |
|  |  | 2 使用相同典型batch\_size | 256 | 3386.49 | 19.97 |
|  | TPUv7/32卡 | 1 根据硬件能力各自拉满batch\_size | 640 | 3610.21 | 20.89 |
|  |  | 2 使用相同典型batch\_size | 256 |  |  |
| orz\_math<br>真实数学推理数据集<br>disable\_ignore\_eos | GPU/32卡<br>tp32+attn\_dp4 | 1 根据硬件能力各自拉满batch\_size | 1392 | 4922.19 | 29.02 |
|  |  | 2 使用相同典型batch\_size | 1024 | 4021.96 | 23.71 |
|  | TPUv7/32卡 | 1 根据硬件能力各自拉满batch\_size | 2048 | 3273.44 | 18.94 |
|  |  | 2 使用相同典型batch\_size | 1024 |  |  |

# fp8 16卡对比数据

对应16卡单实例，成本为 TPU v7 86.4$/h， H200 84.8$/h

| 测试组 | 显卡 | 对比设置 | batch size | 吞吐 (tokens/s) | ROI |
| --- | --- | --- | --- | --- | --- |
| random长尾<br>in: 1k; out: 31k; random\_ratio: 0.1 | GPU/16卡 | 1 根据硬件能力各自拉满batch\_size | 352 | 2969.38 | 35.02 |
|  |  | 2 使用相同典型batch\_size | 256 | 2587.44 | 30.51 |
|  | TPUv7/16卡 | 1 根据硬件能力各自拉满batch\_size | 352 | 3216.69 | 37.23 |
|  |  | 2 使用相同典型batch\_size | 256 | 2800.33 | 32.41 |
| random非长尾<br>in: 1k; out: 15k; random\_ratio: 0.9 | GPU/16卡 | 1 根据硬件能力各自拉满batch\_size | 224 | 3422.45 | 40.36 |
|  |  | 2 使用相同典型batch\_size | 192 | 3013.31 | 35.53 |
|  | TPUv7/16卡 | 1 根据硬件能力各自拉满batch\_size | 294 | 3854.76 | 44.61 |
|  |  | 2 使用相同典型batch\_size | 256<br>192 | 3650.71<br>3451.63 | 42.25<br>39.95 |
|  |  | 3 另外batchsize | 512 | 4009.10 | 46.40 |
| orz\_math<br>真实数学推理数据集<br>disable\_ignore\_eos | GPU/16卡 | 1 根据硬件能力各自拉满batch\_size | 768 | 3574.02 | 42.15 |
|  |  | 2 使用相同典型batch\_size | 512 | 2926.38 | 34.51 |
|  | TPUv7/16卡 | 1 根据硬件能力各自拉满batch\_size | 1024 | 2805.63 | 32.47 |
|  |  | 2 使用相同典型batch\_size | 512 | 2906.76 | 33.64 |

# fp8 16卡对比数据 without eplb

对应16卡单实例，成本为 TPU v7 86.4$/h， H200 84.8$/h

| 测试组 | 显卡 | 对比设置 | batch size | 吞吐 (tokens/s) | ROI |
| --- | --- | --- | --- | --- | --- |
| random长尾<br>in: 1k; out: 31k; random\_ratio: 0.1 | GPU/16卡 | 1 根据硬件能力各自拉满batch\_size | 352 | 2969.38 | 35.02 |
|  |  | 2 使用相同典型batch\_size | 256 | 2587.44 | 30.51 |
|  | TPUv7/16卡 | 1 根据硬件能力各自拉满batch\_size | 400 | 3367.67 | 38.98 |
|  |  | 2 使用相同典型batch\_size | 256 | 2570.41 | 29.75 |
| random非长尾<br>in: 1k; out: 15k; random\_ratio: 0.9 | GPU/16卡 | 1 根据硬件能力各自拉满batch\_size | 224 | 3422.45 | 40.36 |
|  |  | 2 使用相同典型batch\_size | 192 | 3013.31 | 35.53 |
|  | TPUv7/16卡 | 1 根据硬件能力各自拉满batch\_size | 320 | 3942.65 | 45.63 |
|  |  | 2 使用相同典型batch\_size | 256<br>192 | 3603.83<br>3032.70 | 41.71<br>35.10 |
|  |  | 3 另外batchsize | 512 | 3893.01 | 45.06 |
| orz\_math<br>真实数学推理数据集<br>disable\_ignore\_eos | GPU/16卡 | 1 根据硬件能力各自拉满batch\_size | 768 | 3574.02 | 42.15 |
|  |  | 2 使用相同典型batch\_size | 512 | 2926.38 | 34.51 |
|  | TPUv7/16卡 | 1 根据硬件能力各自拉满batch\_size | 768 | 2852.48 | 33.01 |
|  |  | 2 使用相同典型batch\_size | 512 | 2837.76 | 32.84 |

附录：

gpu 32卡原始结果： [https://alidocs.dingtalk.com/i/nodes/l6Pm2Db8D4R2d54wFGlDyazQ8xLq0Ee4](https://alidocs.dingtalk.com/i/nodes/l6Pm2Db8D4R2d54wFGlDyazQ8xLq0Ee4)

gpu16卡原始结果： [https://alidocs.dingtalk.com/i/nodes/7QG4Yx2JpLl0bwLRFB3gML3jJ9dEq3XD](https://alidocs.dingtalk.com/i/nodes/7QG4Yx2JpLl0bwLRFB3gML3jJ9dEq3XD)

tpu 32卡原始结果：[https://alidocs.dingtalk.com/i/nodes/3NwLYZXWynN5ADnOtQ2MRoQRVkyEqBQm](https://alidocs.dingtalk.com/i/nodes/3NwLYZXWynN5ADnOtQ2MRoQRVkyEqBQm)

tpu 16卡原始结果： [https://alidocs.dingtalk.com/i/nodes/dxXB52LJqnKbYOnRtQ4ZOYwG8qjMp697](https://alidocs.dingtalk.com/i/nodes/dxXB52LJqnKbYOnRtQ4ZOYwG8qjMp697)
