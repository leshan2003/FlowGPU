# MoE 前人工作全文审计 v2

> 审计日期：2026-10-20  
> 证据目录：`data/stage1/moe_review/`  
> 状态：已获取 PDF 和抽取文本，尚未建立逐页/chunk 的人工 curated KG。  
> 用途：判断机制重叠、baseline 可实现性和实验指标，不直接迁移论文数值。

## 1. 审计结论

九个最近工作已完成全文级机制核对：

- Expert Activation Patterns；
- Cobalt；
- Director；
- HCRMap；
- ReXpert；
- Sieve；
- Expert Streaming/FSE-DP；
- Not All Models Suit Expert Offloading；
- HybriMoE。

没有一篇工作直接实现“外部 GPU HBM fallback + 类脑 SRAM overlay + GPU/Brain
assignment-level co-serving + cross-pool migration”这一完整组合。

但通用组成机制均已被覆盖：

- 热专家驻留、复制和迁移：HCRMap、EPLB、Sieve；
- coactivation-aware placement：Cobalt、ReXpert、Expert Activation Patterns；
- 两时间尺度布局：Cobalt、HCRMap；
- 在线预测与低停机迁移：Director；
- GPU/PIM 动态执行分区：Sieve；
- CPU/GPU offload、prefetch 和 score cache：HybriMoE；
- 多 chiplet 动态 expert trajectory：Expert Streaming；
- resident SRAM/ReRAM expert pool：SHIP、ReXpert。

因此，本项目不能把“动态放置”“热复制”“共激活聚类”“在线迁移”分别当作主要
novelty。可辩护的系统差异只能是这些机制在 **HBM-anchored external Brain
overlay** 约束下产生的跨池 co-dispatch、容量前沿和迁移干扰问题。

## 2. 机制与实验对照

| 工作 | 全文确认的机制 | 实验形态/baseline | 对本项目的约束 |
|---|---|---|---|
| Expert Activation Patterns | 100k+ 真实请求；task-family popularity shift；prefill/decode 激活相关；request clustering + expert placement | 三模型、真实 trace、linear/EPLB/placement 对照；all-to-all 数据最多降 20%，MoE layer runtime 改善至多约 6% | synthetic 必须保留 request/task 顺序；应加入 prefill 信号预测 decode 的策略；不能只报机制指标 |
| Cobalt | pair coactivation/Ochiai；周期 MILP 全局 co-location；每 step node 内负载重平衡；communication-aware task assignment | 32×B200 训练；EP+FSDP、SmartMoE、LAER-MoE；step time、cross-node volume、imbalance | 支持快慢双时钟和“先选少 remote node 再选 replica”；训练结论不能作为 serving baseline |
| Director | 对 pending requests 预测 routing；compute-bound 阶段迁移；容量约束近优布局 | 多个 MoE 模型和 H20/H200/4090 集群；offline、reactive、uniform、SA、predictor/oracle ablation | 在线 proactive 不是新颖点；必须报告 predictor error、迁移重叠和 oracle gap |
| HCRMap | hotness/load/migration/pressure 驱动 promote/retain/demote/evict；slow residency + fast token mapping；hysteresis/min residency；versioned directory；migration throttle | LEGOSim，同 3.5D substrate 重放 Hydra/MoEntwine/PIMoE | 本项目必须实现版本目录/双缓冲/迁移节流，否则落后于现有机制；必须比较 pressure-aware 策略 |
| ReXpert | attention/FFN disaggregation；resident ReRAM FFN；bounded core-local multicast pooling；coactivation placement；load-aware fetch | measured+modeled，真实 decode trace；H20/H100/H800；同计算/同带宽归一；机制 ablation | 这是最接近的系统。仅“SRAM resident + coactivation”不足；必须突出 GPU HBM fallback、跨池选路和外部类脑拓扑 |
| Sieve | runtime token-to-expert distribution 决定 GPU/PIM expert partition；考虑算术强度、通信、GPU/PIM 吞吐；overlap compute/communication | Ramulator 2.0 cycle simulator；PIMoE/AllExp/NoExp；throughput-interactivity Pareto | H/B 分区不能只按冷热，必须按 batch、expert 内 token 数和通信成本动态选择；要报告 Pareto frontier |
| Expert Streaming/FSE-DP | 低 batch、多 chiplet；expert micro-slice trajectory；D2D + DDR；paired-load/token buffering；compute-communication overlap | EP、Hydra；端到端 attention+MoE；内存和带宽 DSE | 不能假设“source/sink bandwidth 简单相加”；需要局部 stream/overlap 和 on-chip buffer 敏感性 |
| Not All Models Suit Expert Offloading | SRP/SCH 量化 local routing consistency；20 个 MoE 模型；cache≈2×active experts 的折中；shared experts 会改变局部一致性 | 真实模型 trace、toy model 控制变量 | 热集、cache size 和 refresh 结论必须模型相关；加入 SRP/SCH 和多模型筛选 |
| HybriMoE | intra-layer CPU/GPU scheduling；impact-driven inter-layer prefetch；score cache | 三模型、kTransformers/llama.cpp/AdapMoE；TTFT、decode throughput、cache ratio | B 模式/H 模式必须有 prefetch、cache miss 和 layer pipeline 才能公平比较 |

## 3. Baseline 接受状态

| Candidate baseline | 状态 | 原因 |
|---|---|---|
| GPU-only / HBM fallback | accept | 本项目必须 |
| Coverage / Replica / Fixed-share ranked | accept | 同硬件、同 trace 可直接实现 |
| Marginal-event ranked | project method | 必须与 above baselines 比较 |
| Full-hyperedge clustering | project method | pair affinity 为 ablation |
| Cobalt-style periodic-global + per-step-local | needs implementation | 只迁移机制，不复现 B200 训练系统 |
| Director-style predictive + safe-phase migration | needs implementation | 需要 online request queue 和 predictor |
| HCRMap-style pressure-aware residency | needs implementation | 需要 versioned directory、pressure counters |
| ReXpert-style resident-FFN coactivation/fetch | needs implementation | 用于 H/B 或 resident-pool 场景；不能冒充类脑 KA200 |
| Sieve-style runtime GPU/Brain partition | needs implementation | 与 H/B 模式最接近的执行分区 baseline |
| Expert Streaming-style local trajectory | needs implementation | 只在有 D2D/stream model 时公平 |
| HybriMoE exact system | reject as executable by default | CPU-GPU runtime 与类脑 kernel/网络不同；采用其调度思想 |

## 4. 新增实验要求

1. `copy budget` 扫描必须同时报告固定 unique coverage、copy distribution 和命中率；
2. 刷新实验必须加入 min residency、hysteresis、最大迁移数和 versioned directory；
3. H/B 模式必须按 runtime token-to-expert distribution 分区，而不是固定冷热阈值；
4. 必须比较 prefetch/no-prefetch 和 cache miss；
5. 必须报告 SRP/SCH 或等价局部一致性指标；
6. 必须报告 Pareto frontier：goodput/interactivity 对 node 数、容量、能耗/成本；
7. 必须将 coactivation placement、load-aware dispatch、expert replication 分开消融；
8. 所有相关工作的具体百分比只作为外部依据，不能移植为本项目预测。

## 5. 证据等级

已获得：

- PDF 文件；
- PDF SHA-256；
- 抽取文本；
- 章节级机制核对。

仍缺：

- 每篇的页码/chunk ID 索引；
- 人工 curated claim relation；
- 可获得的 executable baseline 适配评估；
- 同行评审状态确认（除 Not All Models/ICLR 2026 与 SHIP/MLSys 2026 外，
  其余当前按 arXiv emerging evidence 处理）。

因此“最近工作差异”可从黄升级为**黄绿**，但还不能给绿色 novelty 结论。
