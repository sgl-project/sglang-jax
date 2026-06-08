# Baseline failures

## config-field
- Observed behavior: The baseline updated only the configuration reference and mentioned `07-kv-cache.md` only as a file that was left unchanged. It did not provide a separate pre-edit impact report or concrete update plan, and the final report included code/doc line numbers in the changed-files list.
- Violated assertions: Does not include PR numbers, commit hashes, or code line numbers in final docs; Uses an update plan before modifying docs. Partial weakness on considering `07-kv-cache.md`: it noticed the page but did not reason from `memory_pool.py` impact in a structured way.

## scheduler-flow
- Observed behavior: The baseline identified `03-scheduler.md` and chose not to update `04-model-executor.md`, but it used a small appended note instead of restructuring the scheduler chapter around the changed ownership of prefill token budgeting. It did not explain whether overview docs were considered.
- Violated assertions: Uses section restructuring when a simple appended note would make the scheduler chapter patchy; Explains why overview docs are or are not updated.

## attention-backend
- Observed behavior: The baseline identified `06-layers-and-attention.md` and also edited `04-model-executor.md`, but it mixed general block sparse attention motivation with project-specific claims. It asserted faster long-context behavior and reduced attention cost without evidence from the simulated implementation or validation. It did not state whether external research was used or separate mature-technology background from code-derived project facts.
- Violated assertions: Uses external research only for general mature-technology background; Separates general motivation from sglang-jax implementation facts; Avoids unsupported performance claims.

## no-doc-update
- Observed behavior: The baseline correctly concluded that no architecture docs need updates, explained that CI cache keys and test formatting are not architecture behavior, left wiki docs unchanged, and included a validation statement.
- Violated assertions: None observed.

## Patterns to address in the skill
- Missing impact report before edits
- Missing structure-aware update decision
- PR or change-management traces in docs
- Appended patch notes instead of systemic integration
- Unsupported project-specific claims
- Missing validation evidence
