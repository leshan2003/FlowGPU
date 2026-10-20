# MoE 高置信度实验设计 v2（冻结版）

> 冻结日期：2026-10-20  
> 收紧审计：2026-10-20（v2.1）  
> 状态：**研究语义、评价合同和实施顺序冻结；性能结论未冻结。**  
> 适用范围：`FlowGPU-main` 中的 MoE 专家分层、复制、聚类和刷新研究。  
> 本文取代 `MOE_研究计划_v1_冻结版.md` 作为后续实现和自动实验的唯一设计入口。v1 保留为审计历史。
> 当前启动门控和代码证据见
> [`MOE_STAGE0_执行审计.md`](MOE_STAGE0_执行审计.md)。Stage 0 执行前快照见
> [`MOE_实验启动门控审计_v2.md`](MOE_实验启动门控审计_v2.md)。

## 0. 结论先行

本项目的核心问题应冻结为：

> 在 GPU HBM 是完整模型锚点、类脑 DRAM/SRAM 是可选专家层的前提下，如何联合决定每个专家在哪个层级、保留几份副本、放在哪个 node/chip，以及何时以多大代价更新布局，使系统在固定 SLO 下用更少的 GPU/类脑设备和更低总成本获得更高 goodput？

当前设计已经具备可执行的变量、对照组和评价框架，但**尚不具备直接启动数十小时最终实验的条件**。必须先解决以下实现阻塞项：

1. 统一 HBM fallback 与“所有专家至少一份 SRAM 主副本”两套互斥语义；
2. 以连续边际价值实现真正的渐进式复制，而不是固定冷热比例或固定 top-k 分层；
3. 在固定 trace、node/chip 数、容量、batch 和副本预算下独立扫描 `C`，加入完整超边目标和 no-worse guard；
4. 把统计更新、布局重算、物理迁移和版本切换拆开，并接入真实 serving 时间线；
5. 清理陈旧结果包和源码哈希不一致，形成可复现的唯一结果入口。

v2.1 审计进一步补充三个启动前必须关闭的风险：

6. 统一 capacity/bandwidth 单位，禁止把 decimal GB 与 binary GiB 混写；
7. 为统计主张完成预先功效分析，并规定多重比较和 primary endpoint；
8. 在没有可信价格/系统功率时，禁止输出总成本优势，只输出 SLO-feasible frontier。

完成这些事项后，后续自动编码和执行可以按本文的 Stage 0–5 顺序运行，不需要再临时决定核心语义。

## 1. 主张、反例与贡献边界

### 1.1 主系统模式

第一阶段主模式是 **GPU HBM-anchored Brain Overlay**：

```text
GPU/HBM：每个专家至少一份可执行副本
Brain SRAM：为部分专家增加 0、1、2... 份额外副本
GPU 与 Brain：可以并行执行同一专家的不同 token assignment
同一个 token-expert assignment：只选择一个副本，不拆分单次 expert GEMM
```

因此，“热专家放 SRAM”准确含义是“为热专家增加脑侧副本”，而不是“从 HBM 删除唯一副本”。该模式不释放 HBM 容量，研究重点是带宽、排队、通信、局部性和跨池协同。

### 1.2 可证伪主张

| 编号 | 主张 | 成立条件 | 主要反例 |
|---|---|---|---|
| C1 | Brain SRAM overlay 在低 batch decode 可改善 SLO goodput，并可能降低 TPOT | 相同 GPU、trace、arrival、SLO、容量和拓扑下配对 goodput 改善，TPOT 作为关键 secondary metric | Brain kernel 或 bridge 延迟使 `earliest-finish` 始终选 GPU |
| C2 | 连续 marginal-event 复制优于二元冷热、纯覆盖和 fixed-share ranked | 在相同 SRAM 字节和副本上限下，多个 skew/模型上 paired goodput 更高 | 普通 skew 下 full coverage 已接近最优，额外副本无收益 |
| C3 | 超边聚类能降低 node/chip fanout（机制主张） | 固定 `C`、node 数、chip 数、副本数和 batch 后，完整超边 cut 下降且 load 不恶化 | fanout 下降但 P99 因单 node/chip 过载而恶化；此时不能升级为系统加速结论 |
| C4 | 增量刷新只在足够长且持续的需求漂移后回本 | `预测收益 > 迁移成本 + 争用 + SLO 损失 + 版本切换成本` | 任务快速抖动，迁移未完成即失效；never-refresh 的累计 goodput 更高 |
| C5 | 独占分层能减少 GPU/HBM 需求，但不自动提高成本效益 | 在相同 SLO、总功率或可信成本下，更少 GPU 的组合仍不差 | 为满足类脑容量需要过多芯片，或 DRAM/SRAM 流量抵消收益 |
| C6 | Brain DRAM 更适合冷存储、预取和 staging，而非每 token 热权重读取 | 低访问频度数据经 DRAM staging 后摊销成本；高频读仍由 SRAM 服务 | 42 MiB expert 的 DRAM→SRAM 搬运频繁进入关键路径 |

每个主张必须同时报告 workload、trace、batch、node/chip、容量、baseline、ablation、置信区间和 provenance。不得用机制指标替代端到端结论。

所有端到端系统主张的预注册主指标是**同一 request trace、arrival 和 SLO 下的
paired SLO goodput**。P99 TPOT、`tokens/s/*` 和 J/token 是必须并列报告的关键
指标。纯机制主张可以使用完整 hyperedge cut 或 bridge bytes 作为机制 primary，
但不得把它们单独写成系统加速结论。

### 1.3 不作出的主张

- 不声称“首次实现动态专家放置”或“首次实现热专家复制”；
- 不把 ReXpert、HCRMap、Cobalt、Director 的公开百分比直接映射到类脑系统；
- 不把 projected HP300 参数重命名成 KA200；
- 不把合成 trace 的结果称为真实 router 分布；
- 不把 MoE 单层固定窗口 completion 称为 serving goodput；
- 不在 attention/KV 未进入模型时声称完整端到端 TTFT 或系统 J/token；
- 不把同一 expert 的单次 GEMM 说成已经在 GPU 和类脑之间切分。

## 2. 冻结的系统语义

### 2.1 模式 G：HBM-anchored overlay

这是第一阶段主模式。

| 项目 | 冻结语义 |
|---|---|
| Attention/KV | GPU |
| routed/shared expert 权重 | HBM 至少一份 |
| Brain DRAM | G 模式不作为热路径；只能作可选 staging/overflow 敏感性 |
| Brain SRAM | 额外专家副本 |
| expert 执行 | GPU 或 Brain 副本二选一 |
| 单 GPU 与类脑协作 | 按 assignment 并行，不拆分同一 assignment 的 GEMM |
| HBM 容量目标 | 检查专家、attention、KV、activation/buffer 后仍可容纳 |

主模型 DeepSeek-V3 的 58 个 MoE 层 × 256 个 routed experts × 42 MiB
共 653,908,770,816 bytes，即 653.91 decimal GB 或 609.00 GiB。MoE 正式实验
冻结单张 H100 为 `80,000,000,000 bytes`，16 张在保留 20% 后共有
1,024,000,000,000 bytes（953.67 GiB）。routed experts 之后还剩 370.09 GB
（344.67 GiB）可用于 attention、shared expert、KV、activation、buffer 和碎片；
必须逐 batch/context 检查，不能只看专家总量。

当前项目中 `flowgpu/units.py` 将“80 GB”容量解析为 binary GiB，而
`scripts/run_moe_research.py` 使用 `80_000_000_000` decimal bytes。两条口径
不能混用。Stage 0 必须统一为显式 bytes 或明确单位；在此之前所有容量数字都要
附带口径。

shared expert 在第一阶段固定由 GPU 执行，不参与 routed expert 的 copy budget。若研究 shared expert 是否能放到类脑，应作为独立敏感性，不得暗中混入主结果。

### 2.2 模式 H：独占容量卸载

模式 H 用于回答“类脑能否替代部分 HBM/GPU 专家容量”，结果不能从 G 外推。

| 项目 | 冻结语义 |
|---|---|
| Attention/KV | GPU |
| shared expert | 第一阶段固定在 GPU，不占 routed expert copy budget |
| 冷专家 | GPU HBM 一份，由 GPU 执行 |
| 热专家 | Brain SRAM 一份，由 Brain 执行；不保证 HBM fallback |
| warm/overflow | 只允许 Brain DRAM staging；若保留 GPU HBM 副本，则该实验属于 G，不属于 H |
| 迁移 | HBM↔Brain、DRAM↔SRAM 均显式计费和争用 |
| 失败路径 | 没有 GPU fallback；SRAM miss 必须等待 DRAM→SRAM，Brain 队列和网络直接进入 SLO |

该模式的主比较不是“G 加更多硬件是否更快”，而是“在相同 SLO、可信成本或总功率下，能否减少 GPU 数量/HBM 容量”。

冻结为 **H-exclusive**：hot expert 有 Brain SRAM 副本时不得同时保留可执行
GPU HBM fallback。否则它会退化成 G，不能解释为容量替代。若容量不足导致某些
专家既不在 HBM 也不在可服务位置，实验应判为 infeasible，不能静默回退。

### 2.3 模式 B：Brain-only expert execution

模式 B 用于检验“MoE 不占用 GPU expert compute，仅由类脑 DRAM/SRAM 服务”的可行性。
第一阶段冻结为 **B-stage**：assignment 需要某 expert 时，先从 Brain DRAM
找到完整 expert，再 staging 到 Brain SRAM 执行。`B-stream`（按矩阵块直接流式
执行、不形成完整 SRAM 副本）只有存在可信 block/submatrix kernel 后才作为独立扩展，
不得与 B-stage 合并。B-stage 必须先在当前 layer/batch 内合并同一 expert 的多个
assignment，只执行一次 DRAM→SRAM 装载，再用 SRAM 副本执行这些 assignment；
不能对每个 token 重复搬运同一个 42 MiB expert。

| 项目 | 冻结语义 |
|---|---|
| Attention/KV | 可保留在 GPU；不计作 GPU expert compute |
| shared expert | 属于 expert execution，按 Brain-only 语义执行 |
| 所有专家存储 | Brain DRAM 至少一份 |
| hot experts | Brain SRAM 副本 |
| expert 执行 | 类脑芯片 |
| DRAM→SRAM | 只允许低频 promotion/prefetch；每次进入关键路径必须计费 |

由于 42 MiB expert 已接近 estimated KA200 的 48 MB SRAM，模式 B 若使用 KA200 profile，几乎必然依赖 DRAM、跨 chip 分片或大量芯片。第一阶段可用 projected HP300 做机制探索，但不能称为 KA200 结论。

### 2.4 副本、assignment 与热冷

- 一份副本是一个完整 expert；第一阶段不允许同一 expert 在同一 chip 重复；
- 一个 token-expert assignment 必须且只能在 GPU、Brain SRAM 或 Brain DRAM-supported path 中选择一个执行位置；
- 不同 token 的同一 expert assignment 可以并行落到不同副本；
- “冷/温/热”是解释标签，不是离散输入；
- 最终物理动作仍是 `r(e)=0,1,2,...`，但分配顺序由连续边际价值决定；
- `R_brain<1` 并不要求先存满所有专家。极热专家获得第二份副本可以优于覆盖更多冷专家。
- 每个 replica/chip/memory tier 必须显式携带 role：`gpu`、`brain_sram` 或
  `brain_dram`；不能仅通过 chip id 或 capacity 推断设备类型。

### 2.5 容量的精确定义

分别记录：

```text
R_unique = Brain 可用于专家权重的 SRAM 字节 / 所有唯一 routed expert 权重字节
R_copy   = Brain 中实际专家副本总字节 / 所有唯一 routed expert 权重字节
```

`R_unique` 是硬件容量参数，`R_copy` 是当前布局使用量。报告不得共用一个模糊的 `R`。同样，HBM fallback 模式下即使 `R_copy=0`，系统仍能运行。

所有预算以 bytes 为第一约束、copy count 为第二诊断量。
`max_brain_replicas × min_expert_size` 只能作为快速上界，不能作为异构 expert
大小下的真实容量检查。

## 3. 连续热度与渐进式复制

### 3.1 从统计到行动

对每个时间窗口 `w` 统计：

```text
p_w(l,e)          = 层 l 中专家 e 的 assignment 比例
p_w(l,n,e)        = 从 home node n 发往专家 e 的 assignment 比例
W_w(H)            = 一个 top-k hyperedge H 的出现权重
```

专家价值不是单一热度，而是当前 placement 上增加一份副本的边际净收益：

```text
Delta(e,c) =
    reduced_weight_read_or_queue_time
  + reduced_node_or_chip_fanout
  + reduced_remote_replica_load
  - batch_fragmentation_penalty
  - migration_amortized_cost
  - incremental_power_or_static_cost
```

约束包括：

```text
sum_c x[l,e,c] * size[l,e] + runtime_reserve <= chip_capacity
sum_(l,e) x[l,e,c] <= C_max
每 chip 同一专家最多一份
总副本数不超过预算
```

### 3.2 “渐进式”的正式定义

渐进式策略不是预先规定“最热 10% 拿 3 份、次热 20% 拿 2 份”这类固定阈值，而是对**第一份、第二份、第三份副本逐步竞争同一容量预算**。

对正式的 `marginal-event` 策略：

1. 对所有专家建立候选集合；
2. 每次选择 `Delta(e,c)/bytes(e)` 最大的可行副本；
3. 如第一份副本的局部性价值高，则覆盖更多专家；
4. 如新增同 node 副本降低队列的收益更高，则头部专家获得更多副本；
5. 放置后重新计算候选价值，直到预算或收益阈值耗尽。

为避免每步对全部 candidate 运行完整事件模型，允许两阶段实现：

1. 用解析代理筛选 top-L 个候选；
2. 只对 top-L 运行 placement-window detailed event model，选择实际净收益最高者；
3. 候选只有在 `Delta > migration amortization + SLO risk margin` 时接受。

当前代码中的 `ranked-<fraction>` 会先保留固定比例的副本预算，再对已覆盖专家
增加副本；它本质上是 **ranked-fixed-share**，只能作为 baseline，不能称为文档
定义的渐进式联合边际策略。`ranked-adaptive` 使用 top-decile 阈值，也只是
adaptive baseline。

可用阶梯预算测试复制的连续程度：

```text
副本预算 = 0%, 10%, 25%, 50%, 75%, 100%
压缩后复制分布 = r(e) 的直方图，而非只报告总副本数
```

### 3.3 baseline 的准确含义

| 名称 | 定义 | 回答的问题 |
|---|---|---|
| Coverage | 优先让更多唯一专家获得脑侧副本 | Brain 容量是否应先买“命中率” |
| Replica | 只持续给最热专家增加副本 | 极端热点下队列/负载收益是否压过覆盖 |
| Joint-marginal | 每一步重算 first/extra copy 的价值 | 在第一份和多份副本之间联合取舍 |
| Ranked-fixed-share | 固定 coverage/replica 比例，再按热度补充副本 | 低复杂度历史 baseline |
| Marginal-event | 以实际 event-model 边际收益逐步消费字节预算 | 正式渐进式联合策略 |
| Oracle | 小拓扑穷举/ILP | 启发式距离最优多远 |

Coverage 不代表“冷专家优先”；Replica 也不代表“完全不命中冷专家”。两者的区别是当前预算下的动作顺序。所有策略必须使用相同 trace、总字节、packing 规则和 dispatcher。

### 3.4 必须改变的历史解释

历史 `hot_frac=0.2, hot_traffic=0.9` 只是人为场景，不是真实专家热度。即使 `R_brain<1`，replica-only 或 mixed ranked 仍是合法策略，不能先强制覆盖所有唯一专家。

## 4. 共激活、聚类与 `C`

### 4.1 使用的数据

原始数据是每个 token 在每层选择的完整 top-k 集合：

```text
H = {e1, e2, ..., ek}
```

必须同时保存和报告：

- marginal popularity：`p(e)`；
- hyperedge frequency：`W(H)`；
- pair support：`p(e,f)`；
- lift：`p(e,f)/(p(e)p(f))`；
- top-set Jaccard 和时间稳定性；
- node/chip fanout 与完整 hyperedge cut。

Pair affinity 只能作为诊断或低阶 baseline。把 `H` 展开成所有 pair 会放大大队列的影响，而且放置用 pair、评估用 hyperedge 会造成目标不一致。

### 4.2 聚类目标

建议的主目标不是“让所有 pair 同 chip”，而是减少一个 token 的跨 node/chip 扇出：

```text
Cut(H) = W(H) * [
    lambda_node  * (|nodes(H)| - 1)
  + lambda_chip  * (|chips(H)| - 1)
]
```

同时加入：

- busiest-chip predicted load；
- busiest-node assignment load；
- SRAM capacity 和 `C`；
- 不得为了 fanout 把多个极热专家塞入同一 node；
- 在 placement window 上以实际 event-model 目标做 acceptance guard。

### 4.3 `C` 的定义与隔离

`C` 冻结为“一颗 chip 最多同时驻留多少个 expert”，不等于 chip 数、副本数或 batch。

为消除历史实验中的混杂，冻结三组实验：

1. **C1-clustering panel**：固定 node 数、chips/node、trace、batch、字节预算
   `B_bytes` 和副本项预算 `M_copies <= node_count × chips_per_node`。
   baseline 与 clustering candidate 必须使用**完全相同的 expert-copy multiset**
   （每个 expert 的副本数相同），只允许改变副本所在的 chip/node。所选
   `B_bytes` 必须在所有测试 C 下都能被实际装下。这样 `C=1` 及所有更高 C 都可行。
   该 panel 专门回答“相同副本、相同资源下，聚类是否改善 fanout/completion”，
   不回答容量扩张收益。
2. **C-scan panel**：固定 node/chip、字节预算 `B_bytes` 和副本项上限
   `M_copies`，只扫描满足 `C >= ceil(M_copies / total_chips)` 且
   `B_bytes` 可装入的点；否则不生成该预算。
   与 C1 panel 分开报告。
3. **C-expansion panel**：固定 node/chip，每个 `C` 允许填满所有 chip 的可用
   bytes。对异构 expert，容量只按 `sum(chip.usable_weight_bytes)` 和实际装箱计算；
   `node_count × chips_per_node × C` 仅是同尺寸 expert 的简化上界。它回答容量
   和串行度共同变化的部署 frontier，不能与 C1/C-scan 混成一条 ablation 曲线。

共同预算 `B` 必须在查看 evaluation trace 前确定；不能只挑“成功可行”的 C 点后
忽略失败点。所有容量不足、装箱失败和无法运行点都必须进入结果表。

### 4.4 `C=1` 的强制验收

`C=1` 必须单独测试：

```text
baseline: unique-balanced / layer-packed
candidate: popularity clustering
           pair clustering
           full-hyperedge clustering
dispatcher、node 数、chip 数、trace、batch、总副本数全部相同
```

聚类算法必须带 no-worse guard：

1. 先满足容量和负载约束；
2. 在 placement window 上比较端到端目标；
3. 只有 SLO violation、P99 和 busiest-node/chip load 不恶化，
   且完整 hyperedge cut 降低时才接受；
4. 记录被拒绝 move/swap 的数量，防止只报告接受后的布局。

这只保证模型内 placement-window 不倒退，不能保证 held-out 永不倒退；held-out 结果仍需配对置信区间。若 `C=1` 聚类仍变慢，应检查 chip 队列、node ingress 和算法未优化完整超边，而不是直接断言“聚类对 `C=1` 无效”。

## 5. 刷新：统计、重算、迁移与切换

### 5.1 “刷新”的准确含义

刷新不是重新训练模型，也不是每个 token 改 routing。它包含四个可独立计时的动作：

| 动作 | 成本 | 频率 |
|---|---|---|
| 统计更新 | 采样、EMA、分布距离 | 可每 token/每 step |
| 布局重算 | CPU/GPU 优化求解 | 可每次重算，不移动权重 |
| 物理迁移 | HBM/DRAM/链路/SRAM 写入 | 只在净收益为正时发生 |
| 版本切换 | placement version 发布、双缓冲、旧副本回收 | 新副本完成和 request 安全点之后 |

因此，“布局重算发现合算”后可以实时启动局部刷新，但不应同步阻塞每个 decode step。合理做法是 receding-horizon incremental refresh：

```text
持续估计需求
→ 判断当前布局的预测累计损失
→ 选少量正收益迁移
→ 后台复制新版本
→ 在 safe point 激活
→ 观察实际收益并撤销错误迁移
```

主要 trade-off 是预测误差、迁移抖动、双缓冲容量、迁移与推理争用、版本复杂度和迁移回滚成本。

冻结状态机：

1. `P_old` 继续服务；
2. 在 compute-bound 或 iteration safe point 后台创建 `P_new`；
3. 迁移流量占用显式预留比例的 bridge、DRAM、node link 和 SRAM write bandwidth；
4. `P_new` 所有目标副本完整并校验后才发布 version；
5. 新 assignment 使用 `P_new`，在途 request 继续绑定 `P_old`；
6. 旧 version 引用归零后回收；
7. 任一复制失败或 SLO 越界超过预算时，停止新增迁移并回滚到 `P_old`。

若双缓冲容量不足，只能采用 stop-the-world 或先 eviction 再 load；该退化必须
作为独立策略计费，不能假定旧、新副本总能同时存在。

布局求解器、drift detector、目录更新和 version publish 的 CPU/GPU 时间也必须
进入预算；若它们不能在 safe phase 或决策间隔内完成，策略视为 overrun，不得忽略。

### 5.2 时间粒度

统计粒度、决策粒度和迁移粒度不能混为一个“刷新周期”。

| 层级 | 可作为统计/触发点 | 是否直接全局迁移 |
|---|---|---|
| token / decode step | 是，更新在线统计 | 否 |
| batch / iteration boundary | 是，低冲突 safe point | 只做小范围局部迁移 |
| layer boundary | 是，可利用计算重叠 | 可做局部迁移 |
| request boundary | 是，适合版本切换 | 可迁移中等集合 |
| conversation turn | 可用 task-family 信号 | 可重新评估 |
| agent task / dataset phase | 强漂移边界候选 | 可做全局兜底 |
| 固定时间 epoch | 作为对照策略 | 由实验决定 |

第一阶段实验中，所有策略在相同 epoch/request boundary 上可观察相同历史窗口。未来信息只能进入明确标记的 clairvoyant oracle。

### 5.3 策略族

| 策略 | 触发/范围 | 作用 |
|---|---|---|
| Never | 无 | 静态上界/基线 |
| Periodic-global | 固定周期，全布局 | 简单但迁移大 |
| Task-boundary | 已知 task family 变化 | 检验可观测边界 |
| Drift-triggered | EMA/分布距离超阈值 | 未知边界 |
| Local-incremental | 每 safe point，少量 copy | 主动态策略 |
| Local + periodic global | 快慢双时钟 | Cobalt 式思路的简化对照 |
| Predictive | 使用可观测 prefill/下一请求特征 | Director 式思路 |
| Clairvoyant oracle | 使用未来窗口 | 不可实现上界 |

### 5.4 workload 场景

刷新实验不能只用递减任务混合，应至少包括：

1. **Stable**：分布稳定，验证 never-refresh 应胜出；
2. **Gradual drift**：code/math/chat/general 比例连续变化；
3. **Abrupt known shift**：task boundary 可观测；
4. **Abrupt unknown shift**：边界不可观测，只能 drift detector；
5. **Burst + recovery**：短时热点爆发后回归；
6. **Alternating tasks**：快速来回切换，测试 thrash；
7. **Long-tail event**：低频但高价值专家突然成为热点。

### 5.5 迁移服务模型

必须分别实现和报告：

- **stop-the-world**：保守下界；
- **background shared bandwidth**：迁移与 serving 共享链路和 SRAM 写带宽；
- **compute-overlap**：在 compute-bound 区间迁移；
- **double buffering/versioning**：新副本完整后发布，在途请求继续旧版本；
- **rollback**：预测失败或 SLO 恶化时恢复旧 placement。

迁移字节不能只用“新增副本数 × expert size ÷ 单条带宽”。至少按源位置、目标位置、共享资源、并行链路、DMA/写入开销和 serving 争用建模。

### 5.6 刷新指标

- 累计 SLO goodput 和总完成时间；
- 迁移前后的 P50/P95/P99；
- migration bytes、完成时间和完成比例；
- 迁移期间 rejected/admitted requests；
- 每次迁移实际节省时间、回本步数和回本墙钟时间；
- 被后续漂移淘汰的无用迁移比例；
- 版本切换期间的双缓冲峰值容量；
- 与 never-refresh 和 clairvoyant oracle 的差距。

## 6. Workload、硬件与公平比较

### 6.1 模型分层

| 层级 | 模型 | 目的 |
|---|---|---|
| Screen-small | Mixtral-8x7B | 8 experts、top-2、快筛 |
| Screen-medium | Qwen3-235B-A22B / gpt-oss-120B | 多 expert、不同 expert 大小 |
| Main | DeepSeek-V3 | 58 MoE layers、256 experts、top-8、42 MiB |

每个模型必须从本地 `flowgpu/workload/models.py` 的真实公开配置生成 expert shape；`_proj` 模型只能作为敏感性，不进主结论。

### 6.2 独立变量

至少覆盖：

```text
batch: 1, 4, 16, 64, 128, 512, 1024
gpu_count: mode-dependent; G 主实验固定 16，H/B 扫描满足容量/SLO 的最小值
brain_nodes: 2, 4, 8, 16, 32, 64
chips/node: 4, 8, 16
C: 1, 2, 4, 8, 14
R_copy budget: 0, 0.1, 0.25, 0.5, 1.0, >1
prefill:decode ratios: 1:8, 1:2, 1:1, 8:1
context: short, medium, long
ingress: all nodes, 1/4 nodes, single ingress
```

本文后续单独写 `nodes` 时，默认指 Brain node；GPU 数必须写 `gpu_count`。
不得把“增加 Brain node”和“增加 GPU”混称为同一个扩展实验。

不需要做全笛卡尔积。先用小模型和单因素扫描找相变，再在 DeepSeek-V3 上验证边界点。

### 6.3 四类公平比较

1. **同 GPU**：固定 GPU 数和 trace，比较 G0（纯 GPU）与 G+overlay，隔离类脑 overlay 的边际价值；
2. **同 Brain 容量**：固定 SRAM 字节、副本预算和拓扑，比较 coverage/replica/ranked；
3. **同 SLO 与总资源/可信成本**：比较 G、H、B 的最小可行资源配置；
4. **同负载扩展语义**：分开 strong scaling 和 weak scaling，不能画在同一条“加 node 收益”曲线。

若没有可信类脑设备价格和系统功率，第三类比较只能报告 SLO-feasible frontier，不能给出绝对成本结论。

“更低总成本”要成为正式 claim，至少需要预先冻结：

- GPU、类脑芯片、板卡、switch、host、NIC、内存和 bridge 的 acquisition cost；
- 折旧/租用、利用率、故障率和冷却成本口径；
- idle/compute/SRAM/DRAM/link 分项 power；
- 相同 SLO 下每种模式的最小可行系统；
- 尾延迟、拒绝请求和迁移开销均计入 goodput。

缺失任一项时，只能报告 `(node count, brain capacity, P99, goodput)` frontier，
禁止输出 `$/token` 或“成本降低 x%”。

`tokens/s/node` 不是足够明确的指标。最终结果必须分别使用：

```text
tokens/s/system
tokens/s/GPU
tokens/s/brain-node
tokens/s/normalized-hardware-unit
```

其中 `brain-node` 只统计类脑节点，`system` 统计实际参与服务的 GPU 与类脑节点；
不得省略分母后把额外的类脑硬件收益描述成单节点效率提升。

### 6.4 Brain-node 扩展停止规则

固定 GPU 数、chips/node、trace、batch 和 SLO 后，先求
`brain_nodes_min`。从该点继续增加 Brain node，至少检查连续两个增量点：

- 若连续两个 paired goodput 增量的 95% CI 上界都小于 2%；
- 且 `tokens/s/brain-node` 连续下降；
- 且 P99 没有显著改善；

则声明达到 empirical saturation，停止该 GPU 配置下的 Brain-node 扩展。

若要同时改变 GPU 数，只能报告二维 `(gpu_count, brain_nodes)` Pareto frontier，
不能用一个方向的饱和点代替二维成本判断。

## 7. 实验阶段与自动执行合同

### Stage 0：语义和数据完整性

目的：在任何新结果前排除已知不一致。

- 统一 HBM fallback 文档和 schema；
- 加入显式 `C_max`、`R_unique`、`R_copy`；
- `task_family`、`stage`、timestamp/window 进入可选 trace schema；
- 修订 KA200 registry 与官方公开值不一致的字段；
- 所有 projected 链路/kernel/power 参数保留统一标签；
- 修复 pair placement 与 hyperedge evaluation 的目标不一致；
- 建立唯一结果目录和源码哈希。

失败则停止，不进入性能实验。

### Stage 1：CPU/fast screen

- 小拓扑 exact oracle；
- `C=1` clustering no-worse；
- coverage/replica/ranked 单因素；
- 正常 skew、极端 skew、uniform；
- 只回答“哪些机制值得进入 detailed”。

验收：所有容量/字节守恒通过；fast 与 detailed 在小点上方向一致；oracle gap 可解释。

### Stage 2：G 模式主实验

- Mixtral/Qwen/gpt-oss 快筛；
- DeepSeek-V3 完整静态 placement；
- fixed-window mechanism 与 continuous serving 分开；
- 同一 trace block 对所有策略 paired replay；
- 报告 copy distribution、GPU/Brain assignment share、P99 和 `tokens/s/node`。

验收：至少两个模型方向一致；关键结论在 bridge/kernel/SRAM reserve 敏感性区间不翻转；否则降级为条件结论。

### Stage 3：H/B 模式

- 实现 Brain DRAM active tier；
- 实现 exclusive placement、无 fallback 的失败/排队路径；
- 对比 equal-SLO 的 G、H、B；
- 报告 attention/KV 成本、DRAM→SRAM 流量、系统功率或明确缺失。

验收：模式语义通过小规模 oracle；无 fallback 时不得静默回退 GPU。

### Stage 4：刷新实验

- migration DAG、source-aware link cost；
- background 争用、double buffering、version switch；
- stable/gradual/abrupt/burst/alternating workload；
- never/periodic/task/drift/local/global/hybrid/predictive/oracle；
- 所有策略使用同一个 `P0`、同一 demand trace、同一 arrival trace、同一 candidate
  optimizer 和同一 migration bandwidth budget；只允许触发/scope 策略不同；
- 累计 goodput、迁移回本和 thrash。

验收：迁移期间 SLO 可见；新 placement 只在完整版本激活；旧副本在无在途引用后回收。

### Stage 5：校准与最终验证

- 若获得 KA200 原始 artifact，重新拟合 kernel/link/power；
- 未获得时保留 `projected/estimate` 边界；
- 运行 request/task-block bootstrap；
- 进行关键边界点的高重复详细实验；
- 输出唯一 canonical bundle 和复核报告。

## 8. 统计置信与判定规则

### 8.1 随机性和重采样

- 合成 workload：screen 至少 5 个 paired seeds，最终边界点至少 20 个；每个 seed 同时用于所有策略；
- 真实 trace：按 request 或 task block 做 paired bootstrap，建议 10,000 次；不得按 token 独立采样；
- 同一 trace block 必须被所有策略完全相同地 replay；
- placement、evaluation 和刷新决策使用严格时间因果切分；
- 多事件/请求可保留 request correlation，不将 token 当作独立样本。

20 个 seed 本身不是统计功效证明。最终实验前必须用至少 10 个 pilot blocks
估计 paired effect 的标准差，并按下式预先计算样本量：

```text
target effect = 5% relative goodput improvement
two-sided alpha = 0.05
power >= 0.80
minimum paired units = 20
```

若计算所需样本超过运行预算，结论只能标为 `underpowered`，不能减少样本后仍声称
supported。

多个模型、batch、node 和策略不能共同使用一个未校正显著性门槛。每个 claim 只能有
一个 primary contrast family；族内采用 Holm 校正。其余组合明确标为
secondary/exploratory，CI 不得被包装成确认性结论。

### 8.2 报告

- paired speedup 的均值/中位数、95% CI 和原始配对差；
- 对百分比延迟同时报告绝对值；
- 报告 seed/request 数量和 drop-out；
- 报告每个 `tokens/s` 指标的分母和系统总硬件范围；
- 对每个关键结论报告 worked/failed model、batch、node 和参数区间；
- 若使用 hardware Monte Carlo，只有 measured repeated samples 才能称为校准误差；projected 参数只能做确定性敏感性。

### 8.3 预注册判定

| 结论 | 规则 |
|---|---|
| supported | paired 95% CI 下界 > 1.0 |
| practically supported | 下界 > 1.05，且主指标不牺牲 P99/SLO |
| inconclusive | CI 跨 1.0 |
| unsupported | CI 上界 ≤ 1.0 |
| conditional | 只在明示 workload/hardware 子集成立 |

表中的 `>1.0` 统一表示“越大越好”的比值。对 TPOT、P99 等越低越好的指标，
必须使用 `baseline/候选` 定义 ratio，或反向解释区间，不得混用原始 latency 差值。

收益必须分解为 spill removal、locality、fanout/reduction、compute balance、batch efficiency、queueing、额外硬件和 migration overhead。不能只给一个总加速比。

## 9. 数据与证据审计

### 9.1 可以较高置信使用

| 数据 | 来源 | 当前等级 |
|---|---|---|
| H100 80 GB、3350 GB/s、NVLink/IB 公开规格 | GPU registry、厂商 datasheet | vendor |
| DeepSeek-V3 61 层、58 MoE 层、256 experts、top-8、d_model=7168、expert FF=2048 | 本地公开 config | public config/derived |
| 42 MiB expert | `3×7168×2048` bytes | derived |
| SHIP 的 SRAM serving、low-batch、容量/网络机制 | 本地 `data/stage1/pdfs/paper.pdf` | paper，非 KA200 |
| KA200 30 cores、48 TOPS INT8、24 TFLOPS FP16、约 12 W | 厂商公开页面 | vendor |

### 9.2 只能作为估计或 projection

| 数据 | 问题 |
|---|---|
| KA200 SRAM 48 MB、2.4 TB/s、NoC、chip/board link、68 GB/s DRAM | estimate，非实测 |
| KA200 expert `t_expert(M)`、multicast/reduce、bridge 曲线、功耗分项 | 缺失 |
| HP300 768 MB SRAM、51.2 TB/s、200 GB/s DRAM、150 W | simulator projection |
| GPU↔Brain 51.2 GB/s、node 内 90 GB/s、node 间 39 GB/s | projected/derived scenario，不是设备测量 |
| GPU/Brain 小 M expert kernel 曲线 | synthetic curve |
| task-conditioned synthetic routing | 由论文趋势约束的生成器，不是本项目真实 trace |
| 设备 acquisition price、host/switch/NIC/cooling 成本 | 当前没有统一可信输入 |
| 完整系统 idle/compute/network power account | 当前只有部分网络能耗和 projected power 参数 |

### 9.3 需要修正的明确不一致

1. `flowgpu/hardware/registry.py` 中 KA200 记录为 FP16 16 TFLOPS、INT8 32 TOPS、15 W，而厂商页面公开为 FP16 24 TFLOPS、INT8 48 TOPS、约 12 W；应保留 URL 和访问时间后修正。
2. `docs/sim_v2/00_scope_and_claims.md` 仍假设所有专家至少一份 SRAM 主副本，与冻结的 HBM fallback 冲突。
3. 当前结果运行器把所有链路 provenance 标成 DeepSeek-derived，未区分 bridge/node link 的 projection。
4. 当前 42 MiB expert 与 estimated KA200 48 MB SRAM 很接近，单 expert 加 buffer 的适配性不能默认通过。
5. `flowgpu/units.py` 把容量单位的 `GB` 转成 binary，而 MoE research runner 直接使用
   `80_000_000_000` bytes；在修复前不得把两套代码输出的容量数字混在同表。
6. 当前 creative synthetic trace 的 `request_id` 全部为 0，且 loader 拒绝额外
   task/stage/window 字段；真实 trace block bootstrap 尚不可执行。

### 9.4 当前结果包状态

| 结果目录 | 状态 |
|---|---|
| `experiments/moe_2026_final_v3` | 最后一个相对完整的组合结果，但源码哈希已与当前源码不一致 |
| `experiments/moe_2026_final_v4` | 含 ranked 结果，但预算修正后已陈旧 |
| `experiments/moe_2026_mechanisms_v5` | 中断且缺少 manifest，不可作为最终证据 |
| 当前源码 | 有 ranked/budget 修正，但尚无重新生成的 canonical full bundle |

因此，现有对比数和加速比只能用于形成假设和选择实验点，不能作为 v2 冻结设计的最终实验证据。

## 10. 前人工作与借鉴原则

表中 2025–2026 arXiv 工作已完成本地 PDF acquisition、SHA-256、文本抽取和
章节级机制核对，详见
[`MOE_前人工作全文审计_v2.md`](MOE_前人工作全文审计_v2.md)。尚未完成逐页/chunk
人工 curation，因此仍不能作为已确认的 executable baseline 或绿色 novelty 证据。

| 工作 | 借鉴内容 | 本项目如何采用 | 不能直接借鉴 |
|---|---|---|---|
| Expert Activation Patterns, arXiv:2604.23150 | 100k+ 真实请求、task-family 漂移、prefill/decode 相关性 | 设置 task drift、已知/未知边界和真实 trace 采集要求 | 其 GPU EP 百分比与类脑拓扑 |
| Cobalt, arXiv:2609.36959 | 周期全局 co-location + step 级 node 内平衡 | 快慢双时钟、local+periodic-global 策略 | 训练任务和 B200 数值 |
| Director, arXiv:2607.08782 | 预测 incoming request、compute-bound 阶段迁移、容量约束优化 | predictive policy、safe-point migration、版本切换 | 需要其预测器和 GPU 网络 |
| HCRMap, arXiv:2607.11586 | hot/warm promotion/demotion/eviction 与资源压力 | 边际价值、pressure-aware cache、dynamic residency | 3.5D 封装内层级 |
| ReXpert, arXiv:2608.13962 | resident MoE FFN pool、coactivation placement、load-aware fetch | 最接近的系统基线设计，必须比较 GPU-fallback overlay 差异 | ReRAM 和其 workload |
| Sieve, arXiv:2605.11277 | GPU/PIM 动态 partition 与通信/计算 overlap | H/B 模式的动态 expert 执行分区 | cycle-accurate PIM 假设 |
| Expert Streaming, arXiv:2603.27624 | 低 batch、多 chiplet、动态 expert trajectory | 局部刷新、stream overlap 和低 batch 基线 | package 内 D2D |
| Not All Models Suit Expert Offloading, arXiv:2505.16056 | 20 个 MoE 模型、缓存约 2×active experts、模型相关局部一致性 | 多模型筛选、cache/copy sweep、SRP/SCH 指标 | 单一 cache 结论 |
| HybriMoE, arXiv:2504.05897 | 动态层内调度、跨层 prefetch、score cache | BG/H/B 调度器与 staging 对照 | CPU-GPU 具体实现 |
| SHIP, MLSys 2026 Industry，本地方 `paper.pdf` | 已实测 SRAM serving、低 batch、容量与网络机制 | 机制参照和系统评价模板 | Groq LPU 不是 KA200 |

相关工作先作为**发现线索/机制候选**。后续必须完成全文核验，并在经过接口适配、
相同 workload/metric 重放后才成为 executable baseline；如果没有可获得的实现，
只能作为相关工作和设计动机。不得把论文名直接当作实验对照组。

## 11. 代码实现缺口

### 11.1 必须补齐

- 显式 `C_max` 与 `R_unique/R_copy`；
- 从本地 model config 自动生成每个模型的 expert shape/top-k/shared expert；
- full-hyperedge clustering objective 和 no-worse guard；
- 对 HBM-fallback 的 extra-brain-copy placement 做聚类，不只聚 primary；
- first-copy/extra-copy 连续 marginal planner；
- H/B 模式的 active Brain DRAM tier；
- task-family/time-aware trace schema；
- migration DAG、source-aware bandwidth、background contention；
- placement version、double buffering、in-flight request 语义；
- drift detector、hysteresis、min-dwell、rollback；
- request/task-block bootstrap；
- complete result contract 和唯一 canonical bundle；
- node-count saturation stop rule。

### 11.2 不应阻塞第一阶段

- packet/cycle-level NoC；
- 单 expert 跨 chip TP；
- speculative decoding/MTP；
- 故障恢复、多租户和完整 prefill/decode 抢占；
- 无校准数据时伪造的硬件参数 Monte Carlo。

### 11.3 新机制的最低测试集

以下测试缺失时不得开始正式 sweep：

- 不同 expert 大小下的 byte budget、`C_max`、packing；
- `C=1` clustering 的资源守恒与 no-worse guard；
- pair placement 与 full-hyperedge cut 的区分；
- marginal-event 策略每次动作的净收益、预算和确定性；
- H-exclusive 无 fallback、B-stage DRAM→SRAM 路径；
- migration 字节守恒、source/target、共享带宽和完成时间；
- 新旧 placement version、在途 request、double buffer、回收和 rollback；
- 任务顺序、request correlation、drift trigger 和 block bootstrap；
- 所有 decisive 参数的 provenance 与缺失成本/功率时的 warning。

### 11.4 自动修改代码前的工程保护

当前目录已建立 Git 基线；继续自动改写时仍必须：

1. 每个阶段开始前确认 clean worktree 或显式保存用户改动；
2. 为 canonical bundle 写入源码、配置、trace 和模型 config 的 SHA-256；
3. 结果目录只允许新建，不覆盖；
4. 自动实验遇到测试失败、守恒失败、registry provenance 缺失或结果目录冲突时立即停止；
5. 每个阶段生成独立 `stage_report.md`，下阶段只能读取已冻结的上一阶段产物。

## 12. 最终验收门槛

启动最终实验前，以下全部必须通过：

- [x] 文档和代码统一使用 HBM fallback 语义；
- [x] G/H/B 三种模式的副本、显式 fallback、DRAM 和失败路径无歧义；
- [x] `C=1` clustering 有 no-worse guard 和固定资源实验；
- [x] full-hyperedge placement/evaluation 目标一致；
- [x] marginal-event 连续规划不再依赖固定 hot fraction；
- [ ] 迁移与 serving 共享资源且 SLO 可见；
- [ ] 所有 decisive 参数有 provenance；
- [ ] current source 能重建其声称的 canonical bundle；
- [ ] 真实 trace 不存在时，所有结论标为 synthetic；
- [ ] KA200 测量不存在时，所有结论标为 public-estimate/projected；
- [ ] 最终结果给出多模型、多 seed/block CI 和负结果；
- [ ] 最小可行 node 及饱和点的停止规则已预注册。

全部通过后，才能把结果升级为：

```text
software-model, synthetic-workload, projected-hardware evidence
```

若获得真实 request-ordered trace 和 KA200 原始测量，才可进一步升级为：

```text
trace-driven KA200-measured evidence
```

### 当前门控快照（2026-10-20）

Stage 0 最低语义已通过，当前状态为
**synthetic/projected simulation GO / measured-hardware claim NO-GO**：

| 检查项 | 当前状态 |
|---|---|
| 现有单元/回归测试 | 137 passed，compileall 通过 |
| HBM fallback | G 模式已实现并有事件模型门控 |
| 显式 C、异构 expert 容量 | 已实现并通过测试 |
| marginal-event 连续规划 | surrogate 与 event callback 已实现；尚需完整网格 |
| full-hyperedge clustering + guard | 已实现并通过 C=1 门控 |
| C=1 固定资源实验 | 已实现并通过微基准 |
| Mixtral/Qwen/gpt-oss 统一模型驱动 | 已实现 |
| H-exclusive / B-stage | 最小语义已实现并通过门控 |
| drift、版本、后台迁移争用 | drift/version semantics 已实现；serving 争用尚未整合 |
| task/request block bootstrap | 已实现工具，真实 trace 输入仍缺失 |
| canonical bundle/当前源码一致性 | Stage 0 bundles 已建立；最终实验 bundle 尚未生成 |
| Git/回滚保护 | 已初始化并提交基线/Stage 0 |

测试通过只覆盖 Stage 0 语义；不覆盖真实 trace、KA200 校准、migration-serving
共流、正式功效分析和最终成本结论。

## 13. 冻结决定

本轮冻结以下研究决策，后续自动编码不得自行改义：

1. 主模式是 G：HBM fallback + selective Brain SRAM overlay；
2. H 和 B 是独立模式，不从 G 自动外推；
3. 冷热连续、动作离散，复制使用逐步边际价值；
4. `R_unique` 与 `R_copy` 分开；
5. `C` 是每 chip expert 上限，必须在固定资源下独立扫描；
6. 共激活以完整 hyperedge 为主目标，pair 为 baseline/诊断；
7. 刷新由统计、重算、迁移、切换四部分组成；
8. 动态刷新只能在 safe point 激活新版本，迁移成本进入 serving 时间线；
9. 真实 trace 和 KA200 测量缺失时，结论等级不得越级；
10. 现有 `final_v3/final_v4/v5` 数值只用于设计筛选，不用于最终论文结论。
11. H 模式是 H-exclusive，不允许静默 GPU fallback；B 模式第一阶段是 B-stage；
12. 所有预算优先使用 bytes，异构 expert 不得只数 copy；
13. primary endpoint 为 paired SLO goodput，多个确认性比较使用 Holm 校正；
14. 无价格/功率数据时只能给出 frontier，不能给出成本优势。
