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

### Q1.1 GPU 到底能否直接发到专家 node

这是本轮审计新增的独立架构轴，不能和聚类策略混在一起。

| Dispatch 模式 | 数据路径 | 适用假设 |
|---|---|---|
| D0 GPU-direct | GPU/Bridge → 专家所在 node → GPU | GPU 持有 activation，并可直接选择专家 node |
| D1 GPU-to-home | GPU/Bridge → token home node → 专家 node → home node → GPU | 分布式 token/router 状态固定在 home node |
| D2 home-resident | token home node → 专家 node → home node | router 输出已经位于 home node，不含 GPU→home bridge |

当前 MoE 主实验的 `run_gate` / `run_serving_gate` 使用：

```text
gpu-origin + forced home-node route
```

即对 Brain 专家副本：

```text
GPU → home node → expert node → home node → GPU
```

如果专家位于 home node，则只保留必要的 bridge 和 node-local 路径；如果专家在
另一个 node，则额外经过 scale-out。

这一强制路径用于表达以下系统：

- router/activation 的 request state 与 home node 绑定；
- 远端 expert 结果需要回到原 home node 做 combine；
- node-conditional routing trace 有意义。

如果真实系统采用 D0：

```text
GPU → expert node → GPU
```

则“聚类必须靠近 home node”的结论需要重新计算。D0 下更重要的是：

- 共激活专家尽量在同一专家 node；
- GPU bridge 到该 node 的流量；
- 返回 GPU 的 combine 流量；
- 专家 node 的负载与出口带宽。

因此后续所有聚类结论必须标注 dispatch mode。当前 `1.297×` 受控收益和
“home-node 不对齐时无收益”的结论，严格属于 D1，不可直接外推到 D0。

当前执行状态：

- D1：已用于主 MoE 实验和聚类压力实验；
- D2：foundation 支持，但未进入本轮 MoE canonical matrix；
- D0-shared：已实现实际 event flow，并完成 Mixtral 全模型
  `1/2/4 nodes × batch 1/4/16 × 5 seeds` 对照；
- D0-per-GPU：当前一份 GPU expert 副本无法服务多个独立 GPU ingress，
  该尝试无效，需重建 request-to-GPU placement 后再测。

当前 D0 初步结果：

- 低/中 batch 下，D0 比 D1 减少约 45% bridge bytes，但 completion 基本不变；
- 2-node、batch 64 的四层 screen 中，M2 的 D0 completion 比 D1 快约 3.9%，
  paired 95% CI `[1.0244, 1.0528]`；
- 同条件下 M3 `earliest_finish` 比 M2 快约 8.2%（D0）和 11.6%（D1）；
- 低/中 batch 下 eager parallel 仍会变差，`brain-first` 与 M2 等价；
- 高 batch 结果目前只覆盖前 4 个 MoE 层。

### Q1.2 home node 有什么潜在优势

home node 只有在真实系统确实存在 request-affine 状态时才有意义。典型优势：

1. attention/KV cache、router 状态或残差累加器位于该 node；
2. expert 输出可在 home node 本地 weighted reduce，不必全部回传中心 GPU；
3. token 的后续 layer 仍可在同一 node 继续执行，减少跨 node 往返；
4. 多个 node 可各自作为请求入口，形成分布式 serving；
5. `lambda[layer, home_node, expert]` 可以表达不同 ingress 的专家偏好；
6. 多副本可以用来为不同 home node 提供就近服务。

home node 的代价：

- 一个全局 placement 难同时最优匹配所有 home node；
- expert 不在 home node 时仍需 scale-out；
- 多 home node 会削弱“把共激活专家放在单一 node”的收益；
- 如果真实系统没有 home-local 状态，强制 home 路由只会增加一次额外跳转。

因此，“设计 home node”本身不是优势；只有它对应真实 request state 时才是优势。

### Q1.3 硬件修改边界

home-node、D0/D1/D2 和专家聚类首先是**系统级拓扑 + runtime 调度问题**，
原则上不需要修改类脑芯片内部的 MAC、SRAM、core 或神经形态计算单元。

可能只改软件/配置的部分：

- placement compiler；
- dispatcher 和 routing table；
- home-node 映射；
- node/chip 选择；
- GPU bridge 使用策略；
- combine 位置；
- trace 字段和 runtime API。

可能需要板卡/系统硬件或固件配合的部分：

- GPU bridge 是否真正能访问每个目标 node；
- ingress NIC/switch 数量与拓扑；
- 是否支持 GPU→目标 node 直达；
- DMA、地址映射、队列和 completion notification；
- 多播、fan-in/reduction、同步和背压；
- 跨 node 链路、交换机端口和带宽预算。

只有以下能力缺失时才可能需要芯片内部或芯片配套 RTL/runtime 改动：

- chip 级 multicast/fanout；
- chip 内或 node 内 reduction；
- per-expert 动态权重切换；
- 多专家并发 resident/eviction；
- expert 流式加载与计算 overlap。

如果目标芯片和 SoC 已经支持“由板级网络把 activation 送到指定 chip、执行 expert、
返回 partial output”，那么 D0/D1 的差别主要是运行时路径选择和 placement 目标，
不需要重新设计芯片计算架构。

### Q1.4 D0 必须单独冻结的实验语义

D0 不是“D1 改一个延迟参数”。它改变请求的数据流：

```text
D0：
GPU activation
→ bridge / fabric
→ 专家 node
→ 专家 chip
→ partial output
→ bridge / fabric
→ GPU combine
```

D0 没有 token home node 参与 expert dispatch/combine。node 仍是专家放置的
scale-up locality domain，但不是请求状态位置。

#### D0 的四个实现变体

| 变体 | GPU 入口 | 到专家 node 的路径 | 是否支持 multicast |
|---|---|---|---|
| D0-shared | 所有 GPU 共用一个 bridge endpoint | 共享全局 bridge | 否 |
| D0-per-gpu | 每个 GPU/NIC 有独立 endpoint | 独立 bridge/port | 否 |
| D0-multicast | 共享或多入口 | 一份 activation fanout 到多 node | 是 |
| D0-local-reduce | 专家 partial output 在 node 内先 reduce | 每 node 只回一份 | 取决于实现 |

当前 topology 中多个 `gpu0..gpu15` 逻辑 chip 都映射到单一
`cluster.gpu_endpoint`，因此现有 D1 结果不能被解释为“16 个独立 GPU bridge”。
后续 D0-per-gpu 必须建立独立 GPU endpoint，或在统计中明确共享 bridge。

#### D0 聚类目标

对每个 layer/hyperedge，至少计入：

```text
bridge activation bytes to distinct expert nodes
+ bridge partial-output bytes
+ node-local chip fanout
+ per-node/chip queueing
+ combine at GPU
```

双副本 D0 的目标是：

```text
选择 GPU 或 Brain 副本
并使 GPU 入口到专家 node 的总 dispatch/combine 时间最小
```

D0 不需要 home-node affinity，但需要：

- GPU ingress 数；
- 每个 GPU 的 bridge 数；
- GPU 到各 node 的连通矩阵；
- multicast 能力；
- node-local reduce 能力；
- 返回 GPU 的关联/路由关系。

#### D0 核心对照

| 实验 | 固定变量 | 主动变化 | 目的 |
|---|---|---|---|
| D0 vs D1 | 同一 plan、trace、GPU/Brain/node 数 | dispatch 路径 | 量化 home hop 的纯开销 |
| D0 shared vs per-GPU | 同一 workload、总 GPU 数 | GPU endpoint 数 | 检查共享 bridge 瓶颈 |
| D0 no-multicast vs multicast | 同一 hyperedge 和 plan | fanout 能力 | 判断聚类收益是否依赖 multicast |
| D0 cluster vs unique | 同 C、副本、node/chip | 专家位置 | 判断专家共置收益 |
| D0 cluster × M2/M3 | 同一设备集合 | tier policy | 判断并行 GPU/Brain 与聚类能否叠加 |
| D0 × batch | 同一 model/placement | batch 1..64 | 判断 GPU/Brain 是否真饱和 |

#### D0 预注册假设

- 若 GPU bridge 充足且无 multicast，共激活专家在同一专家 node 可减少目标
  node 数、fanout 和 combine 路径；
- 若所有 GPU 共享一个已饱和 bridge，专家聚类对 completion 的帮助可能被
  bridge 总量限制；
- 若 GPU 能对多 node multicast，则不同 node 共置的收益可能下降；
- M3 在 D0 下应比 M2 更容易利用 GPU 空闲算力，但前提是不存在双副本调度开销；
- D0 下 home-node 不对齐不应造成 D1 中的额外绕行。

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

固定相同 hyperedge、只排列 6 种 community→home-node 对应关系时：

- full-hyperedge 的 node cut 在全部 6 种情况下都是 0；
- 本地比例出现 `100% / 33.3% / 0%`；
- 只有完全对齐时 completion 达到 `1.297×`；
- 其余 5 种基本不变。

因此当前压力实验直接表明：降低 community 内部 node fanout 不等于降低
request-home 到专家副本的通信。

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

同时实现 dispatch-mode 门控：

- D0 只在 GPU 与目标专家 node 之间计费；
- D1 强制经过 home node；
- D2 从 home node 开始计费；
- 禁止“拓扑里有 bridge 但从 home node 免费开始”的漏算；
- 同一 trace 在三种模式下分别生成 case id，不得混用结果。

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

这一实验只适用于 D1/D2。必须同时运行 D0：

```text
D0：GPU 直接到专家 node
D0 + cluster：只优化共激活专家的目标 node 和 GPU bridge/combine
```

否则会把“home-node 架构的通信成本”误写成“聚类算法固有成本”。

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

1. 实现 D0 shared/per-GPU dispatch 和 D0/D1 门控；
2. 修正 M1 全模型执行语义；
3. 增加 scheduler policies；
4. 建立小拓扑 oracle；
5. 跑 D0/D1 路由隔离；
6. 跑共激活和 dispatch-aware locality 受控实验；
7. 跑 C×node×batch 主矩阵；
8. 跑 M1/M2/M3 的 iso-resource 比较；
9. 跑 cluster×replica；
10. 真实 trace 接入后做 block bootstrap；
11. 汇总 canonical bundle 和负结果。

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
| D0 比 D1 少一次 home round-trip，但共享 GPU bridge 可能成为瓶颈 | 待验证 | same-plan D0/D1 |
| D0-per-GPU 与 D0-shared 的结果不能互相外推 | 待验证 | bridge topology sweep |
| M3 在 HBM 充足、GPU 未饱和时不劣于 M2 | 理论预期，未验证 | iso-resource M2/M3 |
| M3 的并行分派对高 batch 更有价值 | 理论预期 | batch sweep |
| M1 在高 SRAM hit、GPU 饱和时可能胜出 | 理论预期 | M1 stress |
| 聚类收益来自共激活而非 marginal popularity | 尚未验证 | fixed-marginal experiment |
| D1 聚类必须显式考虑 home affinity | 当前压力实验支持 | D1 home-affinity sweep |
| D0 聚类不需要强制靠近 home node | 待验证 | D0/D1 same-trace comparison |
| C=1 只限制 chip 共驻，不禁止 node 聚类 | 代码语义支持，需端到端验证 | C×cluster matrix |
| 副本和聚类收益可以相加 | 不应预设 | cluster×replica factorial |
| 一个固定聚类布局可在 task 漂移后一直有效 | 不应预设 | temporal drift |

## 11. 额外必须补齐的风险清单

### P0：不补齐就不能裁决 M1/M2/M3 或聚类

| 风险 | 当前缺口 | 必须补齐 |
|---|---|---|
| D0 未实测 | 主结果全部为 D1 | D0/D1 同 trace 对照 |
| GPU ingress 被合并 | 16 个 GPU 逻辑 chip 共用一个 `gpu_endpoint` | D0-per-gpu 与 shared bridge 两种模型 |
| scheduler 非全局最优 | `earliest_finish` 是逐 assignment 贪心 | per-layer load-aware 与 small-batch oracle |
| 硬件不等价 | GPU-only 少 16 个 Brain chips | iso-resource 与 mode-only 两套比较 |
| M1 不完整 | B-stage 只做单层 grouped staging | 全模型 B-stage、prefetch/overlap |
| 聚类目标不完整 | 全局 cut 未含 home affinity/GPU bridge/node load | dispatch-aware objective |
| 聚类缺少 oracle | 无法判断启发式与真实最优差多远 | 6–12 expert exhaustive oracle |
| 多模型层数不等 | 三个大模型只测前 4 个 MoE 层 | 全模型或明确分层 screen |

### P1：会显著改变结论

| 风险 | 当前缺口 | 必须补齐 |
|---|---|---|
| multicast/fanout | topology 有能力位，但主实验未系统扫描 | no-multicast vs node/chip multicast |
| node-local reduce | 未与全部回 GPU combine 对照 | local reduce vs central reduce |
| GPU 真正利用率 | 未输出时间加权 GPU/Brain utilization | timeline 和 per-node utilization |
| bridge 竞争 | GPU 侧共享 endpoint 可能隐藏/集中瓶颈 | shared/per-GPU bridge sensitivity |
| C 活跃边界 | 当前 C≥4 从未成为约束 | 构造每 chip 候选数超过 C 的 stress |
| pair vs full hyperedge | pair 在压力实验中未改变布局 | fixed-community pair/full decomposition |
| 副本与聚类交互 | 尚无相同 SRAM budget 的 factorial | cluster × replica matrix |
| batch 构成 | 大部分主实验 batch 很小 | batch 1..64 与 mixed request length |
| 时间漂移 | refresh 只做前两层代表重放 | task phase 长窗口完整 replay |

### P2：提高外部有效性，但不阻塞第一轮 revalidation

- 真实 request-ordered router trace；
- request/task block bootstrap；
- KA200 kernel、链路、并发、功耗实测；
- 多租户、故障、抢占和重试；
- prefill/decode 交互；
- 真实 acquisition price、功率和 TCO；
- 芯片内 multicast/reduction/RTL 能力核验。

### 当前最不妥的三件事

1. 把 D1 的 home-node 结论写成通用聚类结论；
2. 把增加类脑硬件后的 Overlay 加速写成类脑芯片效率；
3. 把 C 未成为约束写成 C 永远无用。

这三条在最终论文和汇报中必须明确禁止。

## 12. 最终证据等级

全部 synthetic/projected 门控通过后，最高可写：

```text
software-model, synthetic-workload, projected-hardware evidence
```

接入真实 request-ordered trace 与 KA200 原始测量后，才可升级为：

```text
trace-driven, KA200-measured evidence
```

在此之前不得给出真实 TCO、最终 node 数或量产性能结论。
