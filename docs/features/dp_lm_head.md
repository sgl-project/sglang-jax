# LM head parallelism

The LM head uses full tensor parallelism by default, independently of attention DP. Enable `--enable-dp-lm-head` to project tokens within each attention DP group. This flag restores the previous LM-head parallel layout when DP is greater than one. It does not change attention, expert parallelism, or request scheduling.

For example, on 16 JAX devices:

| Arguments | Attention TP | LM head TP | LM head weight copies |
| --- | --- | --- | --- |
| `--tp-size 16 --dp-size 16` | 1 | 16 | 1, sharded over 16 devices |
| `--tp-size 16 --dp-size 16 --enable-dp-lm-head` | 1 | 1 | 16 |
| `--tp-size 16 --dp-size 4` | 4 | 16 | 1, sharded over 16 devices |
| `--tp-size 16 --dp-size 4 --enable-dp-lm-head` | 4 | 4 | 4 |

The mesh remains `(data=dp_size, tensor=tp_size/dp_size)`. Full-TP LM head weights use `P(("data", "tensor"), None)` for `[vocabulary, hidden]`. Token rows are replicated for the projection, and the vocabulary-sharded result is redistributed to the original data groups before general sampling. JAX lowers these layout changes to collectives. There is no separate host gather and no assumption that every DP group has the same number of live requests (the existing padded batches apply).

Fused speculative greedy verification and topk=1 draft calls instead keep the global vocabulary layout through argmax, then redistribute only token IDs to their DP groups. This avoids moving full logits between DP groups and allows the compiler to fuse projection with the local vocabulary reduction. Fused greedy prefill uses the same path. Calls returning target logits/logprobs, non-greedy target sampling, and vocabularies not divisible by global TP retain the existing full-logits path. The flag still controls weight partitioning for both target and draft; this optimization does not change its meaning.

With the flag enabled, weights use `P("tensor", None)` and the projection keeps token rows sharded over `data`. With DP=1 both choices have the same physical partitioning.

Standalone `ParallelLMHead` weights are loaded directly in their selected layout when vocabulary is divisible by the partition count. Other vocabulary sizes are padded once after loading and logits are trimmed to the real vocabulary before sampling. Non-divisible vocabularies temporarily load replicated, so their peak loading memory can exceed final weight memory. Tied input/output embeddings retain their input embedding layout and are reshared for the projection; they do not get the persistent-weight memory saving of an untied head.

Draft runners use the same flag. Target-to-draft head sharing retains the target array layout; this is not an independent draft TP control.

Full TP reduces per-device standalone head weight storage but adds communication. Measure prefill, decode and speculative verification on the intended hardware before drawing performance conclusions.
