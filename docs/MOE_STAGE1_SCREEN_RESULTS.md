# MoE Stage 1 Fast-Screen Results

> 日期：2026-10-20  
> 状态：small paired-seed screening  
> 证据边界：synthetic routing + projected hardware；不是 final performance。

## 1. 运行范围

已完成：

- Mixtral-8x7B：5 seeds，Zipf exponent `0.5/1.3/2.0`；
- gpt-oss-120B：前 4 个 MoE 层，5 seeds，mxfp4，Zipf exponent `1.3`；
- DeepSeek-V3/Qwen3：四种模式 Stage 0 geometry/semantic gate；
- 每个 screen 含 GPU-only、coverage、replica、fixed-share ranked、
  marginal proxy 和 continuous-batch serving。

未完成：

- 20-block 功效实验；
- node/chips-per-node 全 factorial；
- 真实 request-ordered trace；
- migration 与 serving 共享流量；
- KA200 校准和可信 TCO。

Qwen/DeepSeek 的全层 5-seed screen 运行时间远超 fast-screen 预算，已终止。
随后用前 4 个 MoE 层、相同 expert 几何和 5 seeds 完成 fast screen。完整层数只在
Stage 0 gate 和后续边界点 detailed replay 中使用。

## 2. Mixtral 结果

`G-overlay` paired speedup，5 seeds：

| Zipf | Coverage | Replica | Ranked-0.25 | Marginal proxy |
|---:|---:|---:|---:|---:|
| 0.5 | **1.047** [1.025, 1.068] | 1.003 [0.997, 1.010] | 1.029 [1.001, 1.057] | 1.022 [1.000, 1.045] |
| 1.3 | **1.103** [1.080, 1.127] | 1.008 [0.995, 1.020] | 1.083 [1.051, 1.114] | 1.026 [1.010, 1.042] |
| 2.0 | **1.083** [1.040, 1.126] | 1.011 [0.996, 1.025] | 1.070 [1.037, 1.104] | 1.019 [1.007, 1.032] |

serving goodput speedup：

| Zipf | Overlay | H-exclusive |
|---:|---:|---:|
| 0.5 | **1.021** [1.013, 1.030] | 1.013 [0.971, 1.056] |
| 1.3 | **1.032** [1.003, 1.061] | 1.019 [0.971, 1.068] |
| 2.0 | 1.029 [0.989, 1.069] | 1.002 [0.888, 1.115] |

观察：

- 当前小规模下 coverage-first 是最稳定的候选项；
- fixed-share replica-first 没有形成收益；
- marginal proxy 的收益较小，但仍为正；
- H-exclusive 的 CI 在多个点跨 1，不能判定优于 GPU-only；
- 当前 workload 太小，不能据此断言“覆盖优于复制”是最终结论。

## 3. gpt-oss 四层结果

Zipf 1.3、5 seeds：

| Policy | Speedup | 95% CI |
|---|---:|---:|
| Coverage | **1.166** | 1.130–1.201 |
| Ranked-0.25 | **1.166** | 1.130–1.201 |
| Ranked-adaptive | 1.157 | 1.113–1.201 |
| Marginal proxy | 1.106 | 1.020–1.192 |
| Replica | 1.015 | 0.973–1.057 |
| Overlay serving | **1.057** | 1.019–1.095 |
| H-exclusive serving | 1.047 | 0.991–1.102 |

这与 Mixtral 的方向一致，但仍是小 trace、projected kernel 结果。

## 3.1 Qwen / DeepSeek 四层 fast screen

Qwen3-235B-A22B，前 4 个 MoE 层，5 seeds：

| Policy | Speedup | 95% CI |
|---|---:|---:|
| Coverage | **1.165** | 1.101–1.229 |
| Joint-marginal | 1.164 | 1.121–1.207 |
| Ranked-0.25 | 1.082 | 1.012–1.153 |
| Marginal proxy | 1.050 | 0.977–1.123 |
| Replica | 1.005 | 0.997–1.013 |
| Overlay serving | 1.094 | 1.050–1.138 |
| H-exclusive serving | 1.053 | 0.985–1.120 |

DeepSeek-V3，前 4 个 MoE 层，5 seeds：

| Policy | Speedup | 95% CI |
|---|---:|---:|
| Coverage | **1.100** | 1.035–1.166 |
| Joint-marginal | 1.105 | 0.998–1.212 |
| Ranked-0.25 | 1.075 | 0.998–1.153 |
| Ranked-adaptive | 1.056 | 1.004–1.108 |
| Marginal proxy | 1.028 | 0.980–1.076 |
| Replica | 1.022 | 0.987–1.057 |
| Overlay serving | **1.077** | 1.028–1.126 |
| H-exclusive serving | 1.013 | 0.902–1.123 |

这两个 screen 只覆盖前 4 个 MoE 层，不能替代完整模型或真实 trace。

## 4. C=1 聚类

在相同 8-expert copy multiset、8 个 C=1 chips、2 nodes 和 event model 下：

- unique-balanced、pair-clustered、full-hyperedge 的 completion 均为
  `13.20087890625 µs`；
- full-hyperedge 没有增加 fanout，但当前 micro workload 也没有暴露可利用收益；
- 该结果支持“no-worse plumbing”，不支持“full-hyperedge 已经优于 baseline”。

## 5. 结果决策

进入后续详细实验的优先级：

1. coverage-first 作为主 baseline；
2. marginal-event 必须使用真实 event-model evaluator，而不是当前 proxy；
3. replica-first 仅在更极端 skew 和更高 copy budget 下继续；
4. H-exclusive 必须扩大 model/batch/trace，才能判断 CI 是否稳定跨 1；
5. C=1 clustering 需要更真实的 hyperedge communities 才能验证收益；
6. 不能用当前结果决定最终 node 数或迁移周期。

## 6. Artifacts

- `experiments/moe_2026_screen_final_v4_mixtral_zipf05/`
- `experiments/moe_2026_screen_final_v4_mixtral/`
- `experiments/moe_2026_screen_final_v4_mixtral_zipf20/`
- `experiments/moe_2026_screen_final_v4_gptoss4/`（4-layer）
- `experiments/moe_2026_screen_final_v4_qwen4/`（4-layer fast screen）
- `experiments/moe_2026_screen_final_v4_deepseek4/`（4-layer fast screen）
- `experiments/moe_2026_stage0_final_deepseek/`
- `experiments/moe_2026_stage0_final_qwen/`
