# MoE experiment bundles

文档入口：
[`docs/MOE_文件与结果索引.md`](../docs/MOE_文件与结果索引.md)。

> **最终 canonical 版本（2026-10-20）：**
> Stage 0 为 `v13`，下游 screen/matrix/C/20-seed 为 `v9`。
> 证据边界：synthetic workload + projected/estimated hardware。
> `moe_2026_final_*`、`static_final_*`、`mechanisms_*` 是早期开发结果，
> 不再作为最终证据。

完整执行和审计见
[`docs/MOE_最终实验执行与审计_v1.md`](../docs/MOE_最终实验执行与审计_v1.md)。

## Stage 0 canonical bundles

- `moe_2026_stage0_final_v13_mixtral`
- `moe_2026_stage0_final_v13_deepseek`
- `moe_2026_stage0_final_v13_qwen`
- `moe_2026_stage0_final_v13_gptoss`

这些 bundle 验证模型驱动几何、显式 C、C=1、G/H/B 模式、版本迁移、
continuous serving 和 shared migration，不是最终性能结论。

## Stage 1 fast screen

- `moe_2026_screen_final_v9_mixtral_zipf05`
- `moe_2026_screen_final_v9_mixtral`
- `moe_2026_screen_final_v9_mixtral_zipf20`
- `moe_2026_screen_final_v9_gptoss4`（前 4 个 MoE 层）
- `moe_2026_screen_final_v9_qwen4`（前 4 个 MoE 层）
- `moe_2026_screen_final_v9_deepseek4`（前 4 个 MoE 层）

## Node × batch

- `moe_2026_matrix_final_v9_n{1,2,4}_b{1,2,4}`

batch 表示同时 admitted 的 request 数。九个点均已生成；旧版本缺失的
`n2/b2` 已补齐。

## C isolation

- `moe_2026_cscan_final_v9_c{1,2,4,8,14}`：64 copies / 64 chips；
- `moe_2026_cscan32_final_v9_c{2,4,8,14}`：32 copies / 16 chips。

## C=1 clustering stress

- `moe_2026_c1_cluster_stress_v5`：固定 3 nodes × 4 chips、9 experts、C=1，
  分开测试对齐、home-node 排列和 expert-to-community 映射。

该实验用于修正原 Stage 0 聚类微基准的退化对照。重验证计划见
[`docs/MOE_聚类与层级调度重验证计划_v1.md`](../docs/MOE_聚类与层级调度重验证计划_v1.md)。

## D0 / D1 dispatch

- `moe_2026_d0_matrix_v7`：Mixtral 32 层、5 seeds、1/2/4 nodes、
  batch 1/4/16，比较 D0-shared、D1-shared、M2/M3 和多种 scheduler；
- `moe_2026_d0_highbatch_v7`：Mixtral 前 4 个 MoE 层、10 seeds、
  2 nodes、batch 64 的高 batch 专项 screen。

## D0 C and clustering

- `moe_2026_d0_matrix_c{1,2,4}_v3`：32-copy budget、Mixtral 32 层、
  5 seeds、1/2/4 nodes、batch 1/4/16；
- `moe_2026_d0_cluster_v3`：相同 D0/C/node/batch 边界下比较
  unique、pair 和 full-hyperedge。

D0-per-GPU 没有进入 canonical 结果。当前每个专家只有一份 GPU 副本时，
无法公平表示“16 个独立 GPU 入口各自持有专家副本”，需要另建
request-to-GPU placement 和 full/partitioned GPU replicas。

Valid owner-GPU follow-up:

- `moe_2026_d0_pergpu_v1`：4 个 owner GPU，每个 GPU 持有全部 Mixtral experts；
  比较 shared-minimal、shared-replicated 和 per-GPU ingress。

## M1 full-model B-stage

- `moe_2026_m1_full_v1`：hot=8，batch 1/4；
- `moe_2026_m1_full_hot16_v1`：hot=16，batch 1/4；
- `moe_2026_m1_full_hot24_v1`：hot=24，batch 1/4；
- `moe_2026_m1_full_hot24_highbatch_v1`：hot=24，batch 16/64。

结果解释：
[`docs/MOE_M1与D0_PerGPU结果_v1.md`](../docs/MOE_M1与D0_PerGPU结果_v1.md)。

Heavy D0 sweeps can be sharded across seed workers:

```powershell
python scripts/run_moe_d0_parallel.py --jobs 5 ... --output NEW_DIR
```

The parallel launcher has been checked to produce byte-identical `raw.csv` and
`aggregate.csv` to serial execution for the same work.

## 20-seed boundary

- `moe_2026_boundary20_v9_mixtral`
- `moe_2026_boundary20_v9_mixtral_zipf05`
- `moe_2026_boundary20_v9_mixtral_zipf20`
- `moe_2026_boundary20_v9_qwen4`
- `moe_2026_boundary20_v9_deepseek4`
- `moe_2026_boundary20_v9_gptoss4`

## Reproduction

Stage 0：

```powershell
python scripts/run_moe_stage0.py `
  --model deepseek_v3 --dtype fp8 `
  --c-max 4 --replicas 32 --hot-experts 16 `
  --seed 211 --brain-nodes 2 `
  --output NEW_STAGE0_DIR
```

Screen / matrix：

```powershell
python scripts/run_moe_screen.py `
  --model mixtral_8x7b --dtype fp8 `
  --zipf-exponent 1.3 `
  --seeds 101,102,103,104,105 `
  --c-max 2 --replicas 16 --hot-experts 8 `
  --brain-nodes 2 --requests 4 `
  --output NEW_SCREEN_DIR
```

每个 `summary.json` 记录决定实验结果的参数；`manifest.json` 记录 runner、
`flowgpu/units.py` 和全部 `flowgpu/sim_v2/*.py` 的 SHA-256。

## Claim boundary

当前结果使用 paper-constrained synthetic routing generator 和 projected/
estimated hardware。它不包含真实 request-ordered router trace、KA200
kernel/link/power 实测或可信 TCO 输入，因此不能把结果表述为真机结论。
