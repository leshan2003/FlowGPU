# MoE 仿真完整矩阵结果 v1

> 日期：2026-10-20  
> 类型：synthetic routing + projected/estimated hardware  
> 状态：Stage 0 gate + Stage 1 fast screen + node/batch/C matrix 已完成

## 1. 实验边界

本结果验证仿真平台和机制方向，不代表 KA200 实测性能。

输入：

```text
GPU profile:   h100_sxm
Brain profile: lynxi_hp300_proj
Bridge:        pcie5_x16
Routing:       paper-constrained synthetic task/community trace
Kernel:        projected hardware-relative curve
```

所有 bundle 的源码哈希均为零差异。当前测试：

```text
137 passed
COMPILEALL_OK
```

## 2. Node × batch

Mixtral-8x7B 完整 32 层，C=2，16 copies，5 seeds。
表中是 coverage-first 的 paired speedup，方括号为 95% CI 下/上界。

| Brain nodes | Batch | Static | Serving |
|---:|---:|---:|---:|
| 1 | 1 | 1.061 [1.013, 1.108] | 1.019 [1.013, 1.026] |
| 1 | 2 | 1.087 [1.027, 1.147] | 1.026 [1.013, 1.040] |
| 1 | 4 | 1.073 [1.019, 1.126] | 1.032 [1.011, 1.054] |
| 2 | 1 | 1.069 [0.987, 1.152] | 1.044 [1.020, 1.068] |
| 2 | 2 | 1.103 [1.080, 1.127] | 1.032 [1.003, 1.061] |
| 2 | 4 | 1.058 [1.029, 1.087] | 0.981 [0.895, 1.068] |
| 4 | 1 | 1.078 [1.050, 1.106] | 1.046 [1.012, 1.080] |
| 4 | 2 | 1.059 [1.047, 1.071] | 1.025 [0.999, 1.051] |
| 4 | 4 | 1.043 [1.020, 1.065] | 1.030 [1.009, 1.051] |

结论：

- 增加 node 或 batch 不是单调收益；
- grouped-M、chip queue、网络争用和额外硬件共同影响结果；
- 当前最稳定的部署点不是最少 node，也不是最多 node；
- 需要以 same-SLO/node-efficiency frontier 选点，不能只看总吞吐。

## 3. C-isolation

### C1 panel：64 copies / 64 chips

此 panel 保证 C=1 与更高 C 都可行；C>=2 不产生 packing 约束。

| C | Static speedup | Serving speedup |
|---:|---:|---:|
| 1 | 1.275 | 1.125 |
| 2 | 1.262 | 1.123 |
| 4 | 1.262 | 1.123 |
| 8 | 1.262 | 1.123 |
| 14 | 1.262 | 1.123 |

C=1 没有退化，反而在 64/64 轻载铺开后略优。

### C-scan panel：32 copies / 16 chips

此时 C=2 是首个可行点，C 更高允许装箱但当前 planner 仍倾向分散。

| C | Static speedup | Serving speedup |
|---:|---:|---:|
| 2 | 1.175 | 1.051 |
| 4 | 1.175 | 1.052 |
| 8 | 1.175 | 1.052 |
| 14 | 1.175 | 1.052 |

结论：

- 在当前 workload 和 placement 规则下，`C>=4` 不是瓶颈；
- 只有“每 chip 必须装超过当前分配数”时 C 才有影响；
- 不能继续用“C 越大越好/越小越快”的单变量叙事，必须与 packing/fanout 联合解释。

## 4. 多模型 20-seed 结果

Mixtral 使用完整 32 层；Qwen、DeepSeek、gpt-oss 使用前 4 个 MoE 层的 fast screen。
每个单元使用 20 个 paired seeds。

| Model | Coverage static | Overlay serving | H-exclusive serving |
|---|---:|---:|---:|
| Mixtral, Zipf 0.5 | 1.053 | 1.026 | 1.025 |
| Mixtral, Zipf 1.3 | 1.085 | 1.033 | 1.015 |
| Mixtral, Zipf 2.0 | 1.081 | 1.036 | 1.022 |
| Qwen3-235B-A22B, 4 layers | 1.112 | 1.085 | 1.026 |
| DeepSeek-V3, 4 layers | 1.081 | 1.074 | 1.007 |
| gpt-oss-120B, 4 layers | 1.176 | 1.051 | 1.041 |

20-seed CI 均为 paired seed interval；它不覆盖 trace sampling 或硬件误差。

## 5. Initial mechanism conclusions

### Coverage vs replication

- Coverage-first 在普通偏斜下最稳定；
- Replica-first 仅在极端偏斜时开始转正，但当前多数场景收益很小；
- Joint-marginal 与 Coverage 在部分模型上接近，DeepSeek 四层略偏 joint；
- HBM fallback 下 `R_copy<1` 并不要求先覆盖全部专家，当前 screen 也允许极热专家多副本。

### Qwen/Deepseek/gpt-oss

- 四个模型均出现正向 overlay 信号；
- 小型 Mixtral 和 gpt-oss 的机制差异不能直接外推到 DeepSeek；
- Qwen 四层的 serving 增益高于 DeepSeek/gpt-oss，说明模型形状和 expert union 很重要。

### Clustering

- C=1 full-hyperedge event-model gate 通过，没有增加 completion；
- 当前合成 community 没有产生明显 completion 增益；
- 不能据此否定真实 trace 中的聚类收益。

### H/B

- H-exclusive 在 Mixtral/gpt-oss 20-seed 上 CI 为正，但 DeepSeek 四层仍跨 1；
- B-stage 已完成 grouped miss、DRAM 传输和 SRAM staging 语义验证；
- B-stage 不是 B-stream，不能据此评价流式执行。

### Refresh

- 版本切换、旧请求保留旧版本、新请求使用新版本均已通过；
- drift 触发和 migration diff 已执行；
- migration 与 serving 共用流量的完整模型尚未集成，因此本轮不报告刷新净收益。

## 6. Result artifacts

- `experiments/moe_2026_stage0_final_v5_*`
- `experiments/moe_2026_screen_final_v5_*`
- `experiments/moe_2026_matrix_final_v5_n*_b*`
- `experiments/moe_2026_cscan_final_v5_c*`
- `experiments/moe_2026_cscan32_final_v5_c*`
- `experiments/moe_2026_boundary20_v5_*`

## 7. Remaining limitations

- 所有 workload 仍为 synthetic；
- 类脑 kernel/NoC/power 仍是 estimate/projected；
- 20-seed 只覆盖 seed 随机性，不覆盖真实 trace sampling；
- 4-layer fast screen 不是完整模型；
- refresh migration-serving 共流和 20-block 真实 block bootstrap 仍待后续 calibration
  phase。
