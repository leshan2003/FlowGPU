# MoE Stage 0 执行审计

> 日期：2026-10-20  
> 基线提交：`fb6c1b3`  
> Stage 0 代码检查点：`2f0ed4a`  
> 当前结论：**Stage 0 semantic gate PASS；Stage 1 screening GO；final sweep NO-GO。**

## 1. 执行结果

代码与测试：

```text
python -m pytest tests -q
133 passed

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

- `experiments/moe_2026_stage0_final_v2_mixtral/`；
- `experiments/moe_2026_stage0_final_v2_deepseek/`；
- `experiments/moe_2026_stage0_final_v2_qwen/`；
- `experiments/moe_2026_stage0_final_v2_gptoss/`。

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

## 5. 仍未关闭的 final-sweep 门

1. 真实 request-ordered router trace 仍不存在；
2. KA200 measured kernel/link/power artifact 仍不存在；
3. migration 与 serving 共享流量的 flow-level 调度尚未接入；
4. event-model marginal planner 还没有在正式模型网格上运行；
5. full model 的 20-block/功效分析还没有执行；
6. acquisition cost、系统功率和 TCO 输入仍缺失；
7. 最终 `stage1–stage5` 自动 runner 尚需基于本 Stage 0 API 完成；
8. 大型结果目录仍需清理和选择唯一 canonical source set。

## 6. 决策

**允许：**

- 进入多模型 Stage 1 fast screen；
- 运行 C=1、普通/极端 skew、coverage/replica/marginal 的筛选；
- 运行小规模 H/B 对照。

**不允许：**

- 将当前 smoke 数值写成最终结论；
- 启动数十小时 final sweep；
- 声称 KA200 实测或真实 TCO。

下一里程碑应是：完成 migration-serving 共流和正式自动 runner，然后重新做门控审计。
