# 06 DeepInfer 借鉴与 EPLB 实验

## 1. 审计对象

本轮只把 DeepInfer 中能够映射到 FlowGPU MoE v2、且不依赖其专用模型和
内核的部分转成可执行实验。

| 项目 | 审计版本 | 许可证 |
|---|---|---|
| DeepInfer | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` | Apache-2.0 |
| deepseek-recipe | `935817c66f5d6e64838f419692545f06787e485f` | MIT |

DeepInfer 当前公开的是离线推理框架，不是完整在线 serving 系统的全部组件。
因此它提供的是算法和实验合同参考，不是 FlowGPU 的真机替换实现。

## 2. 借鉴结论

### 2.1 已进入本轮实现

1. **专家并行负载均衡（EPLB）**

   DeepInfer 先给热点 logical expert 增加冗余 physical copies，再把
   physical experts 分配到 rank。核心目标是降低每个 step 中最慢 rank 的
   相对负载，而不是只降低平均负载。

   本项目映射为两阶段：

   ```text
   stage 1: 在固定 Brain copy budget 内分配 unique copies / redundant copies
   stage 2: 保持 copy multiset，按 step/layer 负载重排 Brain node/chip
   ```

2. **负载指标**

   对每个 step/layer 区间计算：

   ```text
   moe_lb = max(rank_load) / mean(rank_load)
   ```

   rank 分别映射到 Brain node 和 Brain chip，并同时报告完整 event-model
   completion。只改善 `moe_lb`、却让更多 assignment 回退 GPU，不能算系统
   加速。

3. **固定预算和因果切分**

   所有策略共享：

   - 同一 GPU primary plan；
   - 同一 planning/evaluation trace 切分；
   - 同一有效 Brain copy budget；
   - 同一 C、node 数、kernel curve 和 evaluation trace。

4. **冷热不是 0/1**

   第 1 份副本给出覆盖，后续第 2、第 3 份副本按边际高峰负载递减。当前
   实验进一步区分“覆盖不足时提前复制”和“覆盖饱和后复制”两种语义。

### 2.2 只作为后续设计参考

- Prefill / decode 使用不同统计窗口和重排周期；
- Wide-EP、prefill-decode 解耦和 phase-aware deployment；
- KV cache TTL、cache hit fraction 与 cache-aware routing；
- prefill/decode 分开 benchmark、warmup 与 steady-state 分开；
- CPU scheduling 与 GPU execution overlap；
- speculative decoding 和 structured output。

DeepInfer 中的窗口和冗余初值产生于其 GPU 生产环境，不能直接写成 FlowGPU
的最优值，只能作为扫描范围和实验设计参考。

### 2.3 不可直接迁移

- DeepSeek V4.1 专用 SWA / YOCO bounded replay；
- DSpark confidence head；
- SM100 / Ascend 950 专用 kernel；
- Mega MoE、FlashMLA、DeepGEMM、DeepEP；
- 3FS、内部 KV-cache 管理和未开源的生产负载均衡组件。

## 3. 可执行设计任务

| 优先级 | 任务 | 状态 |
|---|---|---|
| P0 | coverage 与冗余副本的固定预算对照 | 已完成 |
| P0 | step/layer-aware Brain replica 重排 | 已完成 |
| P0 | 满容量 chip 的副本交换 | 已完成 |
| P0 | node × batch × redundancy 网格 | 已完成 |
| P1 | 覆盖饱和后启用 DeepInfer 式冗余副本 | 已形成设计，待合适硬件预算 |
| P1 | prefill/decode 两套窗口与周期扫描 | 未完成 |
| P1 | KV-cache-aware routing 与命中率 | 未完成 |
| P2 | 与 DeepInfer 公开 EPLB 同 trace 对照 | 缺少可执行 trace/环境 |

## 4. 算法与实现

实现文件：

- `flowgpu/sim_v2/eplb.py`
- `scripts/run_moe_eplb.py`
- `tests/test_sim_v2_eplb.py`

### 4.1 第一阶段：副本预算分配

给定 Brain copy budget `B`：

```text
coverage_budget = floor(B * (1 - redundancy_fraction))
```

先按 observed load 选择不同 experts；剩余预算按下一份副本能降低的峰值负载
依次分配：

```text
marginal_value(e, c) = peak_load(e) / (c * (c + 1))
```

其中 `c` 是当前副本数。这个规则允许第 2、第 3 份副本继续增加，而不是把
专家固定切成 hot/cold 两类。

需要注意：第一份 Brain copy 同时提供“覆盖收益”，冗余 copy 只提供“负载
分摊收益”。当 `B` 远小于 expert 总数时，二者不是等价资源，本实验专门
量化了这项代价。

### 4.2 第二阶段：位置重排

重排只移动 Brain replicas：

- GPU primary replicas 不动；
- physical-copy multiset 不变；
- primary 标志不变；
- 检查 bytes、C 和“同一 expert 不重复放同一 chip”；
- 有容量空位时移动副本；
- chip 装满时交换两个副本；
- planning trace 不能换成 evaluation trace。

## 5. 控制实验

### 5.1 固定变量

```text
模型             Mixtral-8x7B，32 MoE layers
experts          256
top-k            2
dtype            fp8
seeds            101,102,103,104,105
nodes            1,2,4
batch            1,16
C                2
copy ceiling     32（受 chips × C 限制后取有效值）
dispatch mode    D0-shared
dispatch policy  brain-first
routing          Zipf 1.3 + paper-constrained synthetic communities
hardware         H100/HBM + projected HP300 Brain SRAM + PCIe5 bridge
```

### 5.2 比较策略

| 策略 | 含义 |
|---|---|
| `coverage` | 固定预算优先覆盖不同热点专家 |
| `coverage-rebalanced` | 不改变副本数，只按 step/layer 负载重排 |
| `eplb-0.000` | EPLB 分配逻辑，但不预留冗余预算 |
| `eplb-0.125` | 12.5% 预算用于热点专家额外副本 |
| `eplb-0.250` | 25% 预算用于冗余副本 |
| `eplb-0.500` | 50% 预算用于冗余副本 |

所有点均为 paired seed 比较；95% CI 使用 Student-t。

## 6. 结果

表中数字为“coverage completion / 当前策略 completion”，大于 1 表示当前
策略比 coverage 快。

| Nodes / Batch | EPLB-0 | EPLB-12.5% | EPLB-25% | EPLB-50% |
|---|---:|---:|---:|---:|
| 1 / 1 | 1.0000× | 0.9900× | 0.9750× | 0.9691× |
| 1 / 16 | 1.0000× | 0.9932× | 0.9796× | 0.9474× |
| 2 / 1 | 1.0004× | 0.9790× | 0.9725× | 0.9447× |
| 2 / 16 | 1.0000× | 0.9624× | 0.9299× | 0.8808× |
| 4 / 1 | 1.0000× | 0.9771× | 0.9609× | 0.9560× |
| 4 / 16 | 1.0004× | 0.9711× | 0.9521× | 0.8897× |

`eplb-0` 与 coverage 基本重合；有时只改善约 0.04%，不构成稳定加速。

### 6.1 负载确实更均衡

以 2 nodes / batch 16 为例：

| 策略 | Brain assignment 比例 | Node `moe_lb` | Chip `moe_lb` | Speedup vs GPU |
|---|---:|---:|---:|---:|
| coverage | 34.88% | 1.976 | 15.589 | 1.2738× |
| EPLB-12.5% | 30.67% | 1.740 | 13.000 | 1.2260× |
| EPLB-25% | 26.27% | 1.491 | 10.135 | 1.1847× |
| EPLB-50% | 17.46% | 1.006 | 4.545 | 1.1216× |

这说明 EPLB 的局部目标确实生效，但更多专家失去 Brain 首份副本后回退到
GPU，端到端性能下降。

### 6.2 Zipf 偏斜边界

2 nodes、batch 16、单 seed 的补充扫描：

| Zipf | Coverage | EPLB-12.5% | EPLB-25% | EPLB-50% |
|---|---:|---:|---:|---:|
| 0.5 | 1.2239× | 1.2023× | 1.1953× | 1.0811× |
| 1.3 | 1.3030× | 1.2421× | 1.2040× | 1.1166× |
| 2.0 | 1.2992× | 1.2349× | 1.1717× | 1.1176× |
| 3.0 | 1.2981× | 1.2708× | 1.2175× | 1.1380× |

提高偏斜会减轻 12.5% 冗余的损失，但到 Zipf 3.0 仍未超过 coverage。该扫描
是单 seed screen，只用于定位边界，不作为置信区间结论。

## 7. 解释

### 7.1 为什么 `moe_lb` 变小却更慢

DeepInfer 的 EPLB 假设 logical experts 已经分布到 rank 上，增加冗余副本主要
解决热点 rank 排队。本项目的 Brain copy budget 远小于 256 个 experts：

```text
Brain copy budget = 16 或 32
experts per model   = 256
```

此时预留一份冗余副本，往往意味着某个不同专家完全失去 Brain 覆盖。这个
专家只能回退 GPU/HBM，产生额外的 bridge 和 GPU 工作。局部 rank 负载下降
不能抵消这部分全局损失。

### 7.2 正确的启用条件

DeepInfer 式 EPLB 应作为第二阶段：

```text
先确保目标 expert 集有 unique coverage
再使用剩余 Brain copy budget 增加热点副本
再做 step/layer-aware placement
```

如果未来硬件扩大 Brain SRAM、node/chip 数增加，或 workload 只激活少量
experts，使 coverage 已饱和，则需要重新测试冗余比例；不能从本轮负结果
推出“EPLB 永远无效”。

### 7.3 指标使用规则

- `moe_lb` 是诊断指标，不是加速结论；
- 必须同时报告 completion、Brain assignment fraction 和 bridge bytes；
- 比较冗余策略必须固定总 copy budget；
- 不能把 coverage completion 与 overlay SLO throughput 直接混比。

## 8. 复现

```powershell
python scripts/run_moe_eplb.py `
  --output experiments/NEW_EPLB_DIR `
  --model mixtral_8x7b --dtype fp8 `
  --seeds 101,102,103,104,105 `
  --nodes 1,2,4 --batches 1,16 `
  --c-max 2 --copies 32 `
  --redundancy-fractions 0.000,0.125,0.250,0.500 `
  --jobs 5
```

canonical bundle：

```text
experiments/moe_2026_eplb_v1/
```

## 9. 证据边界

本实验属于：

```text
software-model, synthetic-workload, projected-hardware evidence
```

它支持以下结论：

1. EPLB 两阶段机制已实现并可复现；
2. 当前 D0/Brain capacity 下 coverage 优先于冗余复制；
3. 降低 Brain 内部负载不均衡不等于端到端更快。

它不支持以下结论：

1. DeepInfer 在其生产环境中无效；
2. 所有真实 router trace 都不需要热专家冗余；
3. 当前结果代表 KA200 实测；
4. 当前结果包含真实 KV cache 命中或生产 TCO。
