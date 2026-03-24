# 从CUDA到 JAX/Pallas: 有哪些坑需要注意

在尝试开发kda pallas kernel中，踩中了非常多的坑，这里面有些坑是因为之前的自身经验导致的错误写法，有些是对pallas 不熟悉导致的， 还有些是 jax 官网也没有明说导致的，这里总结下，避免后来的人继续踩坑。

# block spec 需要和 grid 相互匹配

pallas 中 output 从vmem写回到hbm上是以 block 为单位的， 一次性固定写回blockspec中的大小，这也意味着你在当前的Grid iteration 中，只能写入并且计算当前block的部分，如果你想通过pl.ds 计算并写入其他block 的部分，会被下一次循环 覆盖。

禁止block spec 指向的hbm 被其他block 共享，会导致相互覆盖，除非只循环1次

也即 block\_shape \* grid = output shape

可以通过两个简单的kernel 来理解下。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/6d65c9db-6d41-47b7-9be3-8ecf905719a5.png)

note: 出现这种情况，是出现了block shape 定义的很大，但是只写入了一部分。

# pallas\_call 要求dims\[-1\], dims\[-2\] 满足特定shape

按照dim0, dim1, ...., dimN-1, dimN 的顺序来说。

pallas\_call 要求传进来的shape满足特定要求。

1.  dimN上 block dim size == tensor dim size or block dim size % 128 == 0

2.  dimN -1 block dim size == tensor dim size or block dim size % 8 == 0


jax 源码中是这样判定的

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/97c03075-27ca-4355-be0e-8e5189af3d2c.png)

# pallas\_call 要求 1d block 需要对齐512Byte 或者不切分

note: bool 按照4byte来算

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/ce7ff27f-f9d3-4d3d-918a-586e1fa565cd.png)

# pallas/mosaic 对于1d tensor, 并不接受任意的block size

当你对一个1d 的tensor，做一个binary/unary 计算的时候，pallas/mosaic 貌似并不接受任意的切分。

图为Size = 1024000 -> block size = 512

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/13144f47-f4c0-4696-8229-47a6a2dd64ad.png)

mosaic 貌似更期望

1.  size <= 1024时, block size == size

2.  size > 1024时，block size == 1024


note: 需要注意的是，这一条貌似并没有在官方文档中表现出来。

# 不支持strided slice

这里其实可以用pad + +reshape + slice来解决的

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/b280878c-3356-434e-8f0a-156a1d9b007f.png)

# 不支持1d的bitcast（位宽变化时）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/337a2ff6-169d-4ae3-a438-7687362e3f90.png)

# 不支持对1d array的reshape

无论是1d -> 2d 还是 2d -> 1d 都不支持

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/0ad6b28e-ab48-4af0-9788-fca221b70c56.png)

# 无法在vmem上 store scalar/ only support smem

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/50c7e217-a937-4fce-844d-f4db0c9e71bc.png)

# any memory space 只能使用async copy来访问

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/eb16f17b-585e-460d-8ee4-0f66cc35604d.png)

# store smem 不支持mask

# dot在lhs/rhs dtype 不同时，支持不全

# convert 类型支持不全

# gather只支持2d

# reduce 支持的类型不全/同时只支持一个维度做reduce

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/b1211331-4c1e-40b9-8e2c-8a1ea97f5848.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/c189c8d8-0751-41d8-8af3-1a1e7320ca32.png)

# cos/acos/tan 等函数不支持accuracy

# scan 语义支持不全

# custom\_jvp 语义支持不全

# custom\_vjp 语义支持不全

# SparseCore gather的时候，只支持int32/fp32

疑似本质上只支持load 32bit

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl05mKerbzBn34/img/18844f6f-9b42-4fc4-99a0-a68e3a777fe8.png)

# async copy的使用限制

总结：

1.  当一个新手写一个pallas kernel的时候，我们需要review 数据流设计，这个是必须的，因为这里面的坑太多了， 并且和cuda 编程不太一样。

2.  jax/pallas 和 triton中不太一样，我们需要尽量在上层，通过一些微小的代价，去除掉大部分的随机性，然后用各种映射，把tensor shape和计算映射到一个tpu的性能友好区间，这是因为tpu 本身支持的可编程性就很差，如果考虑到随机性的话，这个映射就会变的非常复杂，不利于性能。

3.  我们在上层做开发的时候，可能底层出现的报错和上层没有太大的关系，这种情况，是我们需要去解决的。

4.  协作开发的时候，一定要先沟通清楚语义，还有API 接口
