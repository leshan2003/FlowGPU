# MoE 聚类与层级调度重验证计划 v1

> 日期：2026-10-20
>
> 状态：FROZEN FOR REVALIDATION
>
> 证据边界：后续主实验仍使用 synthetic workload + projected hardware；
> 真实 trace 和 KA200 实测作为升级实验。

## 1. 本次要解决的三个问题

### Q1：三种专家执行层级到底如何比较

本文冻结以下三种模式，后续不允许再用含混的 “类脑模式” 代替：

| 模式 | 权重位置 | 热专家执行 | 冷专家执行 | GPU 作用 |
|---|---|---|---|---|
| M1 Brain-only | 全部专家在类脑 DRAM；热集在 SRAM | 类脑 SRAM | 类脑 DRAM → SRAM staging | 不执行专家 |
| M2 Tier-exclusive | 热专家只在类脑 SRAM；其余只在 HBM | 类脑 SRAM | GPU/HBM | 只执行冷专家；热专家没有 GPU fallback |
| M3 HBM overlay + parallel | 全部专家在 HBM；热专家另有 SRAM 副本 | GPU 或类脑 SRAM | GPU/HBM | 与类脑共同服务不同 assignment |

对应当前实现：

- M1 ↔ `B-stage`，但当前只有单层 grouped staging，尚不是完整模型执行；
- M2 ↔ `H-exclusive`；
- M3 ↔ `G-overlay` 与 `earliest_finish` dispatcher。

### Q2：聚类到底优化什么

聚类不是单纯最小化 pair cut，也不是只要把共激活专家放在同一位置。

真实目标必须同时考虑：

1. 跨 node dispatch/combine 流量；
2. node 内跨 chip 流量；
3. 请求 home node 到专家副本的距离；
4. 最忙 node/chip 的负载；
5. SRAM 容量、每 chip 专家数 `C`；
6. 副本对社区结构的破坏或补偿；
7. batch 内与 batch 间共激活差异；
8. 时间漂移和迁移成本。

### Q3：C=1 与聚类是否矛盾

C=1 禁止多个专家共享同一 chip，但不禁止多个专家位于同一 node 的不同 chip。

因此：

- C=1 不妨碍 node-level clustering；
- C=1 妨碍 chip-level co-residency；
- C=1 不会自动增加专家串行计算；
- 聚类仍可能因为放错 home node 或造成 node 热点而变差。

## 2. 理论预期

### 2.1 M3 对 M2 的条件支配

若满足以下条件：

- HBM 容量充足；
- M2 与 M3 使用相同 GPU 和类脑 chip 数；
- 两者使用相同数量的 SRAM 热副本；
- M3 scheduler 能按实际完成时间分配 assignment；
- 不存在额外的副本一致性或调度开销；

则 M3 是 M2 的可行动作超集：

```text
M2 能做的 assignment → GPU(冷) 或 Brain-SRAM(热)
M3 能做的 assignment → GPU(全部) 或 Brain-SRAM(热)
```

M3 额外拥有“热 expert 也可回 GPU”和“同一热 expert 的不同
assignment 可同时分到 GPU/Brain”两项动作。

理论上，M3 应不差于 M2，且在 GPU 未饱和、Brain 热副本排队时更好。

但这不是无条件结论。M3 可能输给 M2 的情况包括：

- GPU 已饱和，再给 GPU 派热 assignment 只会排队；
- 双副本导致调度、缓存一致性或所有权管理开销；
- 副本放置占用了本可用于覆盖更多专家的 SRAM；
- `earliest_finish` 只做逐 assignment 贪心，不是全局 makespan 最优；
- bridge/HBM 成为瓶颈，而 M2 的热路径完全避开它。

因此必须实现并比较：

- strict-tier scheduler；
- Brain-first + GPU-overflow；
- earliest-finish；
- 每 batch/layer 的全局 load-aware scheduler。

### 2.2 M1 对 M2/M3 的条件优势

M1 的潜在优势：

- 不经过 GPU↔Brain bridge 获取专家权重；
- 对低 batch、高 SRAM hit rate 的专家更友好；
- 可能减少 GPU 算力被小 expert invocation 低效占用。

M1 的主要风险：

- 类脑 DRAM 带宽通常远低于 GPU HBM；
- cold expert 需要 DRAM→SRAM transfer；
- B-stage 当前采用 grouped full-expert staging，不是 block streaming；
- 所有计算都依赖类脑 kernel 曲线；
- DeepSeek 级别的全部专家需要大量 DRAM node。

理论判断：

> M1 不必假设一定最差；当 GPU/bridge 饱和且 SRAM hit rate 足够高时，
> M1 可能胜出。但在当前 projected profile 和低 DRAM 带宽下，不应假定
> M1 能普遍击败 M2/M3。

### 2.3 当前可以直接下的结论

当前已完成实验支持：

- M3 在固定 2 nodes、batch 2、16 copies、C=2 下相对 GPU-only 稳定为正；
- M2 只有条件性正收益；
- M1 只有单层语义结果，没有完整模型性能对比。

当前不能支持：

- M3 已证明优于 M2；
- M1 必然最差；
- M2 必然优于 M1；
- “GPU 充分利用”已经由当前 dispatcher 实现。

## 3. 当前实验审计

### 3.1 有效性判断

| 实验 | 有效内容 | 不能支持的结论 |
|---|---|---|
| 20-seed 固定 2 nodes/batch 2 | paired seed 下策略均值与 CI | node 趋势、全模型、相同硬件效率 |
| Node × batch 9 点 | Mixtral 5-seed 趋势 | 最终显著性、成本有效性 |
| C 64-chip 扫描 | 当前 planner 的实际 packing | C 在所有 workload 中无效 |
| C 16-chip 扫描 | 小资源 panel 的可行性 | 与 64-chip panel 合并成统一曲线 |
| 原 Stage 0 C=1 聚类 | 代码路径可执行 | 聚类收益或损失 |
| C=1 聚类压力实验 | 有区分度布局下的机制证据 | 当前聚类器已可稳定获益 |
| H-exclusive | 条件性 serving 结果 | 全部专家类脑化 |
| B-stage | grouped DRAM staging 单层语义 | 完整 M1 性能 |
| shared migration | 网络共享和版本语义 | 完整 refresh 净收益 |

### 3.2 C 实验审计

5 个 seed 的实际布局：

| C | 使用类脑 chip 数 | 每 chip 最大专家数 | 布局 |
|---:|---:|---:|---|
| 1 | 64 | 1 | 与 C≥2 不同 |
| 2 | 42–48 | 2 | 基准 |
| 4 | 42–48 | 2 | 与 C=2 完全相同 |
| 8 | 42–48 | 2 | 与 C=2 完全相同 |
| 14 | 42–48 | 2 | 与 C=2 完全相同 |

审计结论：

- 当前 workload 下 C=4/8/14 不是活跃约束；
- C=1 使用更多 chips，略快，不代表 C=1 对所有工作负载都更好；
- 真正需要补的是让每 chip 候选数超过 C 的压力场景；
- C 必须和 node affinity、chip fanout、batch 一起扫描。

### 3.3 聚类实验审计

原 Stage 0 微基准中，三种方法生成完全相同的 chip 布局：

```text
unique、pair、full-hyperedge 的 completion 都是 19.2007 µs
```

因此该结果只能作为 plumbing gate，不能作为收益判断。

新增 C=1 压力实验：

- 3 nodes × 4 chips；
- 9 experts；
- 3 个互斥三专家 hyperedge；
- 固定 C=1、副本数和 token 数；
- 只改变 expert 到 community 的映射。

community 与 home node 对齐时：

| 策略 | Node cut | 本地比例 | Completion |
|---|---:|---:|---:|
| unique | 40 | 55.6% | 20.951 µs |
| pair | 40 | 55.6% | 20.951 µs |
| full-hyperedge | 0 | 100% | 16.151 µs |

full-hyperedge 在该案例为 `1.297×`。

20 个随机 community 映射中：

- pair 在 20/20 案例中与 unique 布局相同；
- full-hyperedge 将平均 node cut 从 32.38 降到 9.52；
- completion 平均只改善 `1.015×`；
- 只有 1/20 明显改善；11 个案例有小于 `0.002%` 的细微下降，
  其余基本不变。

审计结论：

- 聚类机制本身有明确潜力；
- 当前 full-hyperedge 目标缺少 home-node affinity；
- 当前 pair 聚类在这个压力结构中无法改变布局；
- 不能把 node cut 降低直接当作系统收益。

## 4. 重验证总体设计

### 4.1 共同控制

每个 mode/cluster 对比必须满足：

- 同一 request-ordered trace；
- 同一 placement/evaluation 时间切分；
- 同一 seed；
- 同一 GPU 数；
- 同一 Brain node/chip 数；
- 同一 SRAM copy budget；
- 同一 SRAM 预留比例；
- 同一 bridge/NoC/DRAM profile；
- 同一 SLO；
- 同一 batch 到达序列；
- 同一 kernel curve；
- 结果包记录全部决定参数和源码 SHA-256。

### 4.2 两类硬件比较

同时保留：

1. **Mode-only frontier**
   - 每种模式使用其所需的最少硬件；
   - 回答“给定功能约束，哪种系统可用”的问题。

2. **Iso-resource comparison**
   - 所有模式使用相同 GPU 数和 Brain chip 数；
   - 回答“在相同设备集合内，哪种调度更好”的问题。

有可信价格和功耗后，再加入 iso-cost/iso-energy。

### 4.3 主指标

Primary：

```text
paired SLO goodput
```

Secondary：

- total completion；
- mean TTFT/TPOT；
- P95/P99；
- GPU utilization；
- Brain assignment fraction；
- local assignment fraction；
- bridge bytes；
- Brain DRAM bytes；
- scale-out bytes；
- max/mean node 和 chip load；
- network/system energy；
- migration bytes/time/overhead。

## 5. 模式重验证实验

### Stage M0：语义门控

实现并验证：

- M1 full-model Brain-only execution；
- M2 strict-tier；
- M3 overlay with GPU/Brain replicas；
- M3 的 strict-tier、Brain-first、earliest-finish、load-aware scheduler。

门控：

- assignment 守恒；
- byte 守恒；
- C/capacity 守恒；
- no silent fallback for M2 wrong tier；
- M1 禁止 GPU expert execution；
- M3 允许 GPU fallback；
- 每层完成时间单调；
- 小拓扑 oracle 可复现。

### Stage M1：batch × node 网格

第一阶段：

```text
models: Mixtral full 32, DeepSeek first 4, Qwen first 4, gpt-oss first 4
nodes: 1, 2, 4
batch: 1, 2, 4, 8, 16, 32, 64
modes: M1, M2, M3
C: 1, 2, 4
hot copy budget: fixed absolute and fixed fraction
seeds: pilot 10, boundary 20+
```

如果 Mixtral 全模型 run 超过预算，必须按预注册的分层 screen 执行，
不能把四层结果直接当全模型。

### Stage M2：scheduler isolation

固定同一 M3 plan、trace、GPU/Brain 资源，仅改变：

- strict-tier；
- Brain-first + GPU overflow；
- earliest-finish；
- per-layer global load-aware。

需要回答：

- GPU 在哪些 batch/layer 未饱和；
- 同一 expert 的不同 assignment 是否真的分流到两边；
- earliest-finish 与全局调度差多少；
- 双副本是否降低 P99，还是只提高平均 goodput。

### Stage M3：M1 B-stage 深挖

变量：

- SRAM hit rate；
- DRAM bandwidth；
- batch；
- grouped miss 大小；
- prefetch/on-demand；
- DRAM→SRAM 与计算 overlap；
- per-node DRAM vs shared DRAM。

目标：

- 找到 M1 开始优于 M2/M3 的条件；
- 如果没有条件成立，必须报告 null result。

### Stage M4：同硬件模式对照

固定同一 GPU+Brain 设备集合：

- M1 禁用 GPU expert execution；
- M2 热 expert 只走 Brain；
- M3 允许 GPU/Brain 并行。

这组回答：

> 在设备已经同时存在时，是否应该把 GPU 用于热 expert 的并行服务。

不把这组结果与“只购买类脑、不购买 GPU”的 M1 成本结论混合。

## 6. 聚类重验证实验

### Stage C0：小拓扑 oracle

范围：

```text
6–12 experts
2–4 nodes
2–4 chips/node
C=1/2/4
离散 top-k trace
```

比较：

- unique-balanced；
- popularity-balanced；
- home-affinity-only；
- pair-clustered；
- full-hyperedge；
- home-aware full-hyperedge；
- load-aware hypergraph；
- exhaustive oracle。

验收：

- oracle completion 与目标函数一致；
- 启发式目标下降必须对应 held-out completion/goodput 改善；
- 若 cut 下降但 completion 不变，标记为 objective mismatch。

### Stage C1：只改变共激活

固定：

- 每个 expert 的 marginal popularity；
- node affinity；
- batch 分布；
- C、副本、芯片数。

变化：

- `coactivation_strength = 0, low, medium, high`；
- hyperedge 大小 = top-k 的完整集合；
- pair-only 与 full-hyperedge 两种 workload。

目的：

> 证明收益来自共激活，而不是热点专家自然更常一起出现。

### Stage C2：只改变 home-node affinity

固定共激活：

- 单 ingress；
- community 与 node 一一对应；
- 多 ingress 混合；
- 随机 home affinity。

目的：

> 验证 home-aware 目标是否必要。

### Stage C3：C × cluster 隔离

```text
C = 1, 2, 4
nodes = 1, 2, 4
chips/node = 4, 8
copy multiset = fixed
```

比较：

- node-level cluster；
- chip-level cluster；
- node+chip 联合；
- unique。

明确拆分：

- cross-node fanout；
- intra-node chip fanout；
- max node load；
- max chip load。

### Stage C4：cluster × replica

比较：

- 单副本聚类；
- 纯热副本；
- cluster-first + replica；
- replica-first + cluster；
- home-aware cluster + replica。

固定 SRAM byte budget。

目的：

> 区分 locality 收益与排队/负载收益。

### Stage C5：batch 与时间漂移

变化：

- batch `1/2/4/8/16/32/64`；
- stable、gradual、abrupt、alternating task family；
- 历史窗口长度；
- refresh trigger 和 min-dwell。

对每个策略使用相同的未来 held-out trace。

## 7. 统计与判定

### 7.1 置信

- policy screen：至少 10 个 paired seeds；
- 关键边界：至少 20 个 paired seeds；
- 真实 trace：按 request/task block bootstrap，不得按 token 独立采样；
- 同一 seed/block 对所有策略重用；
- 多模型、多 node 的确认性比较使用 Holm 校正。

### 7.2 聚类收益判定

只有在以下同时成立时才标记 `supported`：

1. held-out 层/请求完成时间或 SLO goodput 的 paired 95% CI 下界 > 1；
2. cross-node bytes 或 remote fraction 下降；
3. P99 不显著恶化；
4. max node/chip load 未超过预注册阈值；
5. 在随机 home-node 映射下方向不反转；
6. 相同 C、副本数和设备集合；
7. 结果可在新 bundle 中重建。

仅 cut 下降、但 completion 不改善时，只能标为：

```text
objective improvement without end-to-end benefit
```

### 7.3 模式收益判定

M3 vs M2：

- 主判定为 paired SLO goodput；
- 同时报告 GPU 利用率、Brain fraction、bridge bytes、P99；
- 如果 M3 只在固定 batch 胜出，标记为 conditional；
- 若 GPU 饱和导致 M3 输给 M2，必须保留反例。

M3/M2 vs M1：

- 先做 iso-resource；
- 再做 mode-only frontier；
- 不把新增 GPU 硬件的收益写成调度收益。

## 8. 需要实现的对象

### 8.1 层级

- `ModeM1BrainOnly`
- `ModeM2TierExclusive`
- `ModeM3OverlayParallel`
- strict-tier scheduler
- Brain-first overflow scheduler
- earliest-finish scheduler
- global load-aware scheduler

### 8.2 聚类

- home-aware hyperedge objective；
- node/chip load terms；
- C feasibility；
- pair/full comparison；
- balanced hypergraph optimization；
- cluster + replica joint budget；
- small-topology oracle。

### 8.3 统计

- paired seed CI；
- paired request/task block bootstrap；
- Holm correction；
- per-layer and per-batch decomposition；
- GPU utilization timeline；
- node/chip load timeline。

## 9. 执行顺序和停止条件

### 执行顺序

1. 修正 M1 全模型执行语义；
2. 增加 scheduler policies；
3. 建立小拓扑 oracle；
4. 跑共激活和 home-affinity 受控实验；
5. 跑 C×node×batch 主矩阵；
6. 跑 M1/M2/M3 的 iso-resource 比较；
7. 跑 cluster×replica；
8. 真实 trace 接入后做 block bootstrap；
9. 汇总 canonical bundle 和负结果。

### 停止条件

- 小拓扑 oracle 都无法产生聚类收益：停止聚类参数 sweep，转入 null-result；
- home-aware 改进使 held-out completion 稳定改善：扩大到主矩阵；
- M3 在 iso-resource 下等价于 M2：检查 GPU 是否饱和或分派是否真正分流；
- M1 在所有合理参数下均无收益：停止 M1 主矩阵，保留条件性反例；
- 只有增加硬件才有效：结论改为 mode-only frontier，不写 iso-cost 优势；
- 无法获得真实 trace/硬件：最终结果保持 synthetic/projected 等级。

## 10. 预注册核心假设

| 假设 | 当前状态 | 判决实验 |
|---|---|---|
| M3 在 HBM 充足、GPU 未饱和时不劣于 M2 | 理论预期，未验证 | iso-resource M2/M3 |
| M3 的并行分派对高 batch 更有价值 | 理论预期 | batch sweep |
| M1 在高 SRAM hit、GPU 饱和时可能胜出 | 理论预期 | M1 stress |
| 聚类收益来自共激活而非 marginal popularity | 尚未验证 | fixed-marginal experiment |
| 聚类必须显式考虑 home affinity | 当前压力实验支持 | home-affinity sweep |
| C=1 只限制 chip 共驻，不禁止 node 聚类 | 代码语义支持，需端到端验证 | C×cluster matrix |
| 副本和聚类收益可以相加 | 不应预设 | cluster×replica factorial |
| 一个固定聚类布局可在 task 漂移后一直有效 | 不应预设 | temporal drift |

## 11. 最终证据等级

全部 synthetic/projected 门控通过后，最高可写：

```text
software-model, synthetic-workload, projected-hardware evidence
```

接入真实 request-ordered trace 与 KA200 原始测量后，才可升级为：

```text
trace-driven, KA200-measured evidence
```

在此之前不得给出真实 TCO、最终 node 数或量产性能结论。
