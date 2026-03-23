# Ring 1T 进度同步

# 目标

1.  测试 1K input, 31k output 的标准 benchmark 并评估 TPU / GPU 的性能对比

2.  完成 [https://huggingface.co/datasets/Open-Reasoner-Zero/orz\_math\_57k\_collection](https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection) 在 Ring 1T 上的模型性能与推理性能测试 


# 背景信息同步 

TPU 当前并行策略实现 : DP + TP Attn , DP + EP MoE

GPU 当前并行策略实现：机内Tp+机间PP

# 20260129

FP8 量化 (预期还需要一天工作量)

*   激活量化先实现了 ffn 计算量化，通信量化因为 vmem tiling 对齐有一些问题还没解，正在验证效果(如果效果符合预期，可以先暂时这样)

*   权重(动态)量化精度定位出在 SharedExpert 部分，已在修复中

*   权重静态量化完成代码编写，正在验证中


GPU 测试

*   回复短的问题已经解决，通过设置 sharegpt-output-len

*   探索 batch\_size 上限(调整服务启动参数，如 fraction 等)


# 20260128

FP8 量化

*   激活量化还在开发中，scale 不太好对齐(per token 只剩下一维， 需要考虑 128 对齐，还在解 bug...)

*   权重(动态)量化已经完成，目前还存在精度问题

*   权重静态量化还需要支持，主要是权重加载部分


GPU 测试 [orz\_math\_57k\_collection](https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection) 数据集(本次测试总共会测试两个数据集,  random 数据集 (random-ratio=0.1)测试长尾 case下的吞吐, [orz\_math\_57k\_collection](https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection) 数据集对应真实 RL 场景)

*   数据集格式需转换成 ShareGPT

*   探索 batch\_size 上限

*   目标数据集对应输出非常短，还需要定位一下(如果直接忽略 eos， 和实际场景也不相符，看看是否是 prompt 或者某一些设置没有打开?


# 20260127

FP8 量化 : Weights 量化完成, 待完成计算

ITL 不符合预期问题 Pending

Gpu使用32卡，ring-1T-fp8模型测出初步基线，结果见文档

确定测试1k input ，31k output， random\_range\_ratio 0.1。num\_prompts根据各自硬件性能极限尽量拉满，充分对比硬件能力

# 20260123

TPU

32卡 -> 64 卡 -> 128 卡 : batch\_size 256 -> 752, 吞吐翻倍. ITL 不符合预期(EP变大后预期延迟下降)

FP8 量化 -> MoE

GPU

继续调整并行方式

# 20260122    

TPU

32卡->64卡，理论bs大小扩展能大于2倍

*   测试数据：[《Ring 1T Benchmark(TPU)》](https://alidocs.dingtalk.com/i/nodes/X6GRezwJlAPmnlARFRMKx7oy8dqbropQ)


GPU

# 20260121

TPU

32卡/64卡推理性能测试数据

模型性能测试数据

TODO 项

*   FP8

*   v6e crash : 暂无进展

*   长序列 flash attention tuning


```shell
16 chips(32 cores)（ep=32, attn_tp=8）
单 core 权重：68.26 GB
单 core 可给 KV：88.32-68.26=20.06 GB
单 core 可存 KV tokens：525,860
集群可承载 KV tokens：525,860 * 4 = 2,103,440
单 replica local batch = 16
集群总 batch：16 * 4 = 64

32 chips(64 cores)（ep=64, attn_tp=8）
单 core 权重：39.76 GB
单 core 可给 KV：48.56 GB
单 core 可存 KV tokens：1,272,971
集群有效 KV tokens：1,272,971 * 8 = 10,183,768
单 replica local batch = 38
集群总 batch：38 * 8 = 304

64 chips(128 cores)（ep=128, attn_tp=8）
单 core 权重：25.51 GB
单 core 可给 KV：62.81 GB
单 core 可存 KV tokens：1,646,526
集群有效 KV tokens：1,646,526 * 16 = 26,344,416
单 replica local batch = 50
集群总 batch：50 * 16 = 800

=============================================

16 chips(32 cores)（ep=32, attn_tp=4）
单 core 权重：72.09 GB
KV 可用显存：16.23 GB
单 core 可存 KV tokens：212,731
单 replica local batch = 6
集群总 batch：6 * 8 = 48

32 chips(64 cores)（ep=64, attn_tp=4）
单 core 权重：43.59 GB
KV 可用显存：44.73 GB
单 core 可存 KV tokens：586,287
单 replica local batch = 17
集群总 batch：17 * 16 = 272

64 chips(128 cores)（ep=128, attn_tp=4）
单 core 权重：29.34 GB
KV 可用显存：58.98 GB
单 core 可存 KV tokens：773,064
单 replica local batch = 23
集群总 batch：23 * 32 = 736
```

GPU 

  需要给出当前 sglang 对 Ling/Ring 1T 的实现策略, 评估是否要集成类似的并行策略, 以及开发工作有哪些.
