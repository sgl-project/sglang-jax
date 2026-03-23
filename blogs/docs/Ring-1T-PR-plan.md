# Ring-1T-PR-plan

1.  核心成果

    1.  **峰值输出吞吐对比 OR 非长尾下平均吞吐/ROI对比**

    2.  ~~prefill 不行~~

    3.  ~~尾部延迟不行~~

2.  写文章分工

3.  跑一些profiling/benchmark结果


# 标题

Ring-1T-FP8 + **量化成果 + with SGLang-Jax**

# TL;DR 先行

简单描述一下，那几个团队合作，用了什么技术，达到了什么成果

# 每个优化独立成节

## Fused moe kernel优化

 $\color{#0089FF}{@Brian(xc)}$  $\color{#0089FF}{@炯轩}$

**核心章节**：

1.  原本fused moe kernel分析

2.  roofline分析

3.  消融实验

4.  优化工作

5.  最后性能结果

6.  编写优化pallas kernel等know how总结 @baihua 帮忙


## EPLB

 $\color{#0089FF}{@Brian(xc)}$

## DP Attention

 $\color{#0089FF}{@Brian(xc)}$

## FP8量化

 $\color{#0089FF}{@炯轩}$

# 消融实验/Profiling

 $\color{#0089FF}{@炯轩}$  $\color{#0089FF}{@Brian(xc)}$

看工作量，不推荐写。因为：

*   多个feature需要在一起才能发挥最大作用。如DP+EPMOE等

*   消融实验工作量比较大


倾向于加针对Fused moe kernel消融实验，配合profiling([https://github.com/openxla/xprof/blob/master/docs/custom\_call\_profiling.md](https://github.com/openxla/xprof/blob/master/docs/custom_call_profiling.md))展示瓶颈与优化下效果(见Fused moe kernel章节)

# Performance

## GPU Baseline  $\color{#0089FF}{@靖宛}$

## TPU 性能

# How to use

 $\color{#0089FF}{@Brian(xc)}$

# Future Work

# Acknowledgements

# Reference
