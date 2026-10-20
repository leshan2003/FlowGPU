# MoE experiment bundles

> **证据状态警告：** `moe_2026_final_v4` 生成后 ranked/budget 源码发生过修改，
> 不能继续作为最终 canonical 证据。最后相对完整的参考包是
> `moe_2026_final_v3`，但当前源码哈希也已变化。正式重跑合同见
> [`docs/MOE_高置信度实验设计_v2_冻结版.md`](../docs/MOE_高置信度实验设计_v2_冻结版.md)。

## Stage 0 canonical bundles

当前源码对应的语义/机制门控结果：

- `moe_2026_stage0_final_mixtral`：Mixtral-8x7B；
- `moe_2026_stage0_final_deepseek`：DeepSeek-V3；
- `moe_2026_stage0_final_qwen`：Qwen3-235B-A22B；
- `moe_2026_stage0_final_gptoss`：gpt-oss-120B。

这些 bundle 验证模型驱动几何、显式 C、C=1、H/B 模式、版本迁移和小规模 serving，
不是最终性能结论。更早的 Stage 0 开发检查点已被忽略。

## Canonical result

Use [`moe_2026_final_v4`](moe_2026_final_v4/) for the current result set.
It combines:

- full DeepSeek-V3-shaped static capacity experiments;
- fixed-size mechanism experiments for co-activation, C, node/batch, skew,
  refresh, seeds, and hardware sensitivity;
- combined `details.json`, `manifest.json`, CSV tables, and plots.

The human-readable interpretation is:

[`docs/MOE_EXPERIMENT_RESULTS_v1.md`](../docs/MOE_EXPERIMENT_RESULTS_v1.md)

## Source bundles

| Directory | Status |
|---|---|
| `moe_2026_static_final_v1` | canonical full-scale static source bundle |
| `moe_2026_mechanisms_v4` | canonical fixed-size mechanism source bundle, including ranked policies |
| `moe_2026_final_v1`, `moe_2026_final_v2`, `moe_2026_final_v3` | superseded combined bundles |
| `moe_2026_quick_v1` … `moe_2026_quick_v5` | development smoke/screening runs |
| `moe_2026_full_v1` … `moe_2026_full_v4` | interrupted development runs |
| `moe_2026_full_v5` | static-only intermediate run superseded by `static_final_v1` |
| `moe_2026_mechanisms_v2` | interrupted ranked-replication exploration; not a final result |

Only the canonical bundle should be cited. Development and interrupted
directories are retained for audit but must not be used as final evidence.

## Claim boundary

The bundle uses a paper-constrained synthetic routing generator and projected
brain hardware curves. It does not contain a real request-ordered router trace
or KA200 measured kernel/network/power artifacts.

## Reproduction

```powershell
python scripts/run_moe_research.py --suite full --components static `
  --output NEW_STATIC_DIR
python scripts/run_moe_research.py --suite full `
  --components community,node_batch,skew,refresh,seeds,sensitivity `
  --output NEW_MECHANISM_DIR
python scripts/assemble_moe_bundle.py `
  --static NEW_STATIC_DIR --mechanisms NEW_MECHANISM_DIR `
  --output NEW_COMBINED_DIR
python scripts/analyze_moe_research.py NEW_COMBINED_DIR
```

The full static sweep is capped at 64 brain nodes because the current Python
flow scheduler entered a multi-GB, hours-long regime at larger topologies.
