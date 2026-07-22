# Grammar Bug 总结

## 概述

Grammar 对象是有状态的。每次调用 `accept_token()` 都会推进内部状态，因此一个
grammar 实例只能属于一个请求。

在 normal overlap V2 测试中发现的问题主要有两类：多个请求共享可变 grammar
对象，以及 overlap sampling 使用了过期的 grammar mask。

## 1. 缓存中的 Grammar 状态被多个请求共享

Grammar cache 原来会直接返回缓存对象：

```python
return value, True
```

使用相同 JSON Schema、Regex 或 EBNF 的请求因此会复用同一个可变 grammar
对象：

```text
请求 A ----+
           +--> 缓存中的 GrammarObject
请求 B ----+
```

请求 A 调用 `accept_token()` 时，也会改变请求 B 随后看到的状态。这会导致请求
B 根据错误的 grammar 状态生成 vocabulary mask。

现在 cache 只保存 grammar 模板，每个请求得到一个独立副本：

```python
return value.copy(), True
```

`INVALID_GRAMMAR_OBJ` 不进行复制，以便继续通过对象身份判断它是否为无效
grammar。

## 2. 等待同一个 Future 的请求共享状态

首次 cache miss 时，grammar 会异步编译。多个使用相同 grammar 的请求可能同时
等待同一个 `Future`，而 `Future.result()` 会向所有等待者返回同一个编译结果
对象。

原来的代码会把这个对象直接赋给请求：

```python
req.grammar = req.grammar.result(timeout=0.03)
```

因此，即使修复了 cache hit 路径，这些同时等待首次编译的请求仍然会共享状态。
现在每个请求都会复制编译模板：

```python
compiled_grammar = req.grammar.result(timeout=0.03)
req.grammar = compiled_grammar.copy()
```

Cache 也会保存自己的独立副本，确保缓存模板不会被任何活跃请求修改。

## 3. 错误复制 Invalid Grammar Sentinel

`INVALID_GRAMMAR_OBJ` 是 grammar 编译失败时使用的哨兵对象，不支持普通 grammar
的 `copy()` 等操作。

原来的代码会先复制编译结果，再判断它是否为 sentinel。因此，非法 grammar
可能在 cache 处理阶段抛出 `NotImplementedError`，而不是进入正常的请求终止
流程。

现在会先区分 sentinel：

```python
req.grammar = (
    compiled_grammar
    if compiled_grammar is INVALID_GRAMMAR_OBJ
    else compiled_grammar.copy()
)
```

Sentinel 不会被复制，也不会作为普通 grammar 模板写入 cache。

## 4. Overlap Sampling 可能使用过期的 Vocabulary Mask

对于连续两个 batch A 和 B，B 的 grammar mask 依赖 A 刚刚采样出的 token：

```text
sample A
  -> process result A
  -> grammar.accept_token(A.token)
  -> 构造 B 的 vocabulary mask
  -> sample B
```

Normal overlap V2 会在处理 A 之前先提交 B 的 model forward。这是安全的，因为
model forward 不依赖 grammar mask；但是 B 的 sampling 依赖该 mask。

因此，单线程路径必须保持以下顺序：

```text
launch forward(B)
process result(A)
更新 grammar 状态和 B 的 vocabulary mask
launch sample(B)
```

`SamplingMetadata` 可以在提交 B 的 forward 时提前创建，但在提交 B 的 sampler
之前必须刷新其中的 `vocab_mask`，不能继续使用处理 A 之前捕获的旧 mask。

旧双线程 overlap 路径使用 `sampling_info_done` 保证这个顺序。V2 的操作都在同一个
scheduler 线程中执行，因此函数调用顺序本身已经完成同步，不再需要 Event wait。

## 可能的表现

这些问题可能表现为：

- 同一个 schema 的第一个请求成功，后续请求失败；
- 使用相同 grammar 的并发请求相互影响；
- 输出不符合指定的 JSON Schema、Regex 或 EBNF；
- `accept_token()` 拒绝采样出的 token，导致请求被终止；
- 非法 grammar 在 cache 处理阶段抛出 `NotImplementedError`；
- overlap 模式根据上一轮的 grammar 状态进行采样。

## 正确的所有权模型

```text
Grammar cache       -> 未被消费的只读模板
活跃请求             -> 模板的独立副本
accept_token()      -> 只修改当前请求的副本
下一批 sampling      -> 使用请求状态更新后生成的 mask
```

## 回归测试范围

Grammar 回归测试应覆盖：

1. 多个使用相同 grammar 的串行请求；
2. 首次编译期间共享同一个 grammar key 的并发请求；
3. 前面的请求结束后再次命中 grammar cache；
4. 非法 grammar 的首次编译和 invalid grammar cache hit；
5. overlap V2 下的多 token constrained decoding；
6. 同一个 batch 中混合 grammar 请求和普通请求。
