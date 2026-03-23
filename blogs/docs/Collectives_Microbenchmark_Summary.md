# Collectives Microbenchmark Summary

# Mesh 设置

## 物理设备的排列

ref: [https://henryhmko.github.io/posts/tpu/tpu.html](https://henryhmko.github.io/posts/tpu/tpu.html)

coords对应申请的tpu topo, 相同的coord通过core\_on\_chip轴区分

```python
[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1), TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(0,1,0), core_on_chip=1), TpuDevice(id=6, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=7, process_index=0, coords=(1,1,0), core_on_chip=1)]
```

### k8s设置

gke-tpu-topology = 2x2x1 为一个tray

gke-tpu-topology=2x2x4 代表4个k8s node 进行分布式推理, 每个node上有2x2个chips

### Torus

下图是一个4x4x4 的3d torus 

可以看到, 在一个2x2平面上 chip间都是通过ici的连接的 (蓝色线) , 但是在正方体的面上, 通过ocs连接, 使得在每个轴上, 第一和第四个chip可以直接通信, 缩短了hop数. 因此只有在一个轴上chip数>2(至少是多机)才会用到ocs

逻辑上的4x4x4 3d torus对应物理上的rack, 

superpod 指可以通过ICI和OCS连接的最大芯片互联配置, 对于tpuv7 , 是 128 个 4x4x4,即9216 chip, rack之间也是通过ocs连接的

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Yvenve5vV9bQjloy/img/6e1e626a-c348-49db-9707-dfaa0feffb60.png)

### TPU Slices with OCS 

tpu拓扑选择会影响通信带宽:

例如，对于全对全通信（如数据并行或张量并行）来说，立方体（例如 8x8x8）会更受青睐，因为它拥有最高的二分带宽。不过，长条形（例如 4x4x32）更适合流水线并行，因为它能更快地与连续层通信（假设其中一层能容纳在 4x4 芯片的子片中）。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Yvenve5vV9bQjloy/img/59d3d091-4df8-4c58-9126-d29f1629b79f.png)

## mesh\_utils.create\_device\_mesh

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Yvenve5vV9bQjloy/img/14c94a2c-6b3f-4ce0-9ac3-16f6984df0d3.png)

`mesh_utils.create_device_mesh` 会自动根据逻辑shape和物理设备mesh 重新排列device mesh

### Path 1: 

以tpuv7x为例, 满足: 设备数是8的倍数且<=32 , 而且physical mesh的x轴y轴, 最多为2x2 (由于通过k8s node申请固定的slice是固定的拓扑, 所以这里都不会超出限制l), 执行path1

(这里会计算物理上的tpu mesh, 主要是通过jax.devices() 的coords计算物理上tpu devices的排列)

```python
# For the x and y axes, we only support at most 2x2 since we can make one ring
# along those axes and repeat with other separate rings along the z axis.
```

8个一组按照这个顺序排列: 

```python
_7X_TRAY_2x2x2_RING_ORDER = (0, 1, 2, 3, 6, 7, 4, 5)
```

因为根据device 坐标, 8个core的物理mesh:

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/mvXyjvK3u90JvL9w/5a5391d7ccb24a32a93f02a170110aa90266.png)

所以ring的顺序是: (0, 1, 2, 3, 6, 7, 4, 5), 有16/32个设备时, 按照继续按照这个结构循环, 这里相当于对device按照ring的顺序进行重排, 然后再按照输入的mesh\_shape进行reshape, 这样mesh\_shape最内侧的维度就可以优先按照ring的顺序进行通信

这个path的排列可以充分利用on-chip带宽

### Path 2:

如果不满足上述条件, 使用通用算法, 

反向遍历逻辑轴mesh\_shape (因为按照api定义, mesh\_shape按照性能敏感由低到高排序), 在避免分割物理轴的情况下优先尝试分配多个物理轴, 优先分配正方形拓扑 (4x4 而不是 1x16)

```python
  算法步骤（第298-352行）

  Step 1: 从最密集的逻辑轴开始分配
  # mesh_shape 按网络强度递增排序，所以反向遍历
  for logical_axis_index, logical_axis_size in reversed(list(enumerate(mesh_shape))):

  Step 2: 优先尝试分配多个物理轴
  # 从最多物理轴开始尝试（例如先试 3 个轴组合，再试 2 个，最后试 1 个）
  for num_axes in range(len(physical_mesh.shape), 0, -1):

  Step 3: 寻找乘积匹配的物理轴组合
  # 找到乘积等于逻辑轴大小的物理轴组合
  if np.prod(c_axes) == logical_axis_size:
      assignment[logical_axis_index] = c_indices

  Step 4: 转置和 reshape
  return physical_mesh.transpose(transpose).reshape(mesh_shape)
```

因此, 这里的通用算法不会考虑 cores\_on\_chip这个轴的特殊情况, mesh\_shape 2x2 时会fallback到通用的情况

# allgather

## 通信策略

查看HLO可知, 这里的通信策略: 是顺时针单向+逆时针单向

```python
ici_strategy_config":{"color_strategies":[{"phase_rings":[{"ring_type":"ICI_RING_TYPE_UNIDIR_CW","core_count":"8","ring_neighbor":"ICI_RING_NEIGHBOR_EXPLICIT","barrier_id":"2","has_reordering_map":false}]},{"phase_rings":[{"ring_type":"ICI_RING_TYPE_UNIDIR_CCW","core_count":"8","ring_neighbor":"ICI_RING_NEIGHBOR_EXPLICIT","barrier_id":"2","has_reordering_map":false}]}]},"constant_propagation_config":{"collective_config_parameter_index":"1"},"physical_core_indices":[0]}},"used_scoped_memory_configs":[]}
```

## 单轴带宽

data shape: \[matrix\_dim, 8, 128\] 固定对第一维进行allgather

Mesh devices:  \[TpuDevice(id=0, process\_index=0, coords=(0,0,0), core\_on\_chip=0)

TpuDevice(id=1, process\_index=0, coords=(0,0,0), core\_on\_chip=1)\]

阈值: 4910x8x128

max bw: 

~ 614GB/s (0-1)

~ 90 GB/s ( 0-2/ 0-4 / 0-6 )

由于twisted torus, 0-4 的耗时和0-2/0-6相同

fallback bw:

~ 107 GB/s

~53 GB/s

小于阈值时: 通过dma.general (hbm-to-vmem)进行通信

大于阈值时: 通过dma.general (hbm\_to\_hbm)

device 0-1:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Yvenve5vV9bQjloy/img/fb9febf2-75ac-4fdd-ab8e-dca1c6749f60.png)

device 0-2, 0-4, 0-6 数据基本相同: 

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Yvenve5vV9bQjloy/img/fc1eb3b4-ea80-4a25-94b3-1ec72c1a6645.png)

[Please go to the DingTalk Docs to view 「Spreadsheet」](https://alidocs.dingtalk.com/i/nodes/GZLxjv9VGqYaE5qnuxewlx2X86EDybno?iframeQuery=anchorId%3DX02mk4w5vgfudvdvygb5qq)

## 单轴

| 轴 | 实测带宽 | 理论带宽 | 实测/理论 |
| --- | --- | --- | --- |
| x | 86.0709385824996 | 100 | 0.860709385824996 |
| y | 91.7592674062763 | 100 | 0.917592674062763 |
| z | 90.3528843490842 | 100 | 0.903528843490842 |
| core | 311.355056720389 | 600 | 0.518925094533982 |
