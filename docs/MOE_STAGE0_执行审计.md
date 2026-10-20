# MoE Stage 0 执行审计

> 日期：2026-10-20  
> 基线提交：`fb6c1b3`  
> Stage 0 代码检查点：`2f0ed4a`  
> 当前结论：**Stage 0 semantic gate PASS；Stage 1 fast screen PASS；
> synthetic/projected simulation sweep GO；measured-hardware claim NO-GO。**

仿真平台当前输入清单和后续实测替换点见
[`MOE_仿真数据与校准接口.md`](MOE_仿真数据与校准接口.md)。缺少实测数据不阻塞
synthetic/projected 仿真实验，只限制证据等级。

## 1. 执行结果

代码与测试：

```text
python -m pytest tests -q
137 passed

python -m compileall -q flowgpu scripts
COMPILEALL_OK
```

新建 Stage 0 runner：

```text
python scripts/run_moe_stage0.py --output NEW_DIR ...
```

已覆盖：

- 显式 device role；
- 显式 `C_max` 和 per-chip expert count；
- heterogeneous expert byte budget；
- decimal capacity units；
- 模型配置驱动 expert shape/top-k/first MoE layer；
- full-hyperedge placement objective；
- C=1 固定 copy multiset；
- marginal-event planner proxy 和 event callback；
- H-exclusive no-fallback；
- B-stage grouped DRAM→SRAM loading；
- source-aware migration diff；
- double-buffer/version activation/in-flight request semantics；
- drift detector 和 paired block bootstrap；
- continuous batching serving 的 layer offset；
- 原子结果目录和 source SHA-256。

## 2. 四模型门控

| 模型 | expert bytes | G | C=1 | H | B | refresh version | serving |
|---|---:|---|---|---|---|---|---|
| Mixtral-8x7B | 176,160,768 | pass | pass | pass | pass | pass | pass |
| DeepSeek-V3 | 44,040,192 | pass | pass | pass | pass | pass | pass |
| Qwen3-235B-A22B | 18,874,368 | pass | pass | pass | pass | pass | pass |
| gpt-oss-120B | 13,219,200 (`mxfp4`) | pass | pass | pass | pass | pass | pass |

DeepSeek geometry：

```text
58 MoE layers × 256 routed experts
routed expert bytes = 653,908,770,816 (609.00 GiB)
first MoE layer = 3
top-k = 8
shared experts = 1
```

这验证了 trace 的 layer offset、GPU fallback、H-exclusive 和 B-stage 在非零
first-MoE-layer 模型上不再出现 key mismatch。

## 3. 当前 bundle

当前源码对应的门控 bundles：

- `experiments/moe_2026_stage0_final_v5_mixtral/`；
- `experiments/moe_2026_stage0_final_v5_deepseek/`；
- `experiments/moe_2026_stage0_final_v5_qwen/`；
- `experiments/moe_2026_stage0_final_v5_gptoss/`。

每个 bundle 包含：

```text
config.json
gate_results.csv
placements.json
summary.json
warnings.json
stage_report.md
manifest.json
```

`manifest.json` 记录本次生成时的源码 SHA-256。早期 `_v1/_v2/_v3/_v4` 目录是开发
过程中的检查点，不可作为最终引用；以当前审计列出的目录为准。

## 4. 门控观察

这些是极小规模语义 smoke，不是最终性能结论：

- 四种模型都能完成 G/H/B/refresh/serving 路径；
- C=1 的 unique、pair 和 full-hyperedge 在固定 copy multiset 下通过 event model，
  当前微基准没有观察到 fanout 或 completion 改善；
- overlay 和 H-exclusive 在极小 batch smoke 中能运行，但收益必须由正式多
  seed/block 实验决定；
- B-stage 只验证 grouped miss 和 DRAM 传输计费，不代表 B-stream；
- migration bundle 验证旧请求和新请求的版本隔离，但迁移流量尚未与 serving
  flow 共享带宽。

因此，Stage 0 的“机制可执行”通过，不等于“机制有收益”。

## 5. 仿真实现在下一步仍需补齐

1. migration 与 serving 共享流量的 flow-level 调度；
2. event-model marginal planner 在正式模型网格上的运行；
3. full model 的 20-block/功效分析；
4. node/chip/batch 的完整 factorial 和 saturation sweep；
5. Stage 1–5 的统一自动 runner；
6. 大型结果目录和 canonical source set 的自动索引。

真实 request-ordered trace、KA200 measured kernel/link/power、可信 acquisition cost
是 **calibration-level upgrades**，不是 synthetic/projected 仿真实验的阻塞条件。

## 6. 决策

**当前允许：**

- 进入多模型 Stage 1 fast screen；
- 运行 C=1、普通/极端 skew、coverage/replica/marginal 的筛选；
- 运行小规模 H/B 对照；
- 基于现有 profile 执行完整 synthetic/projected simulation sweep。

**仍不允许：**

- 将当前 synthetic 数值写成 KA200 实测；
- 在没有价格/功率数据时声称真实 TCO；
- 把 projected profile 的方向一致性当成已完成硬件校准。

下一里程碑应是：完成 migration-serving 共流和正式自动 runner，然后执行完整仿真
sweep；实测数据到来后通过 calibration 接口重跑。
