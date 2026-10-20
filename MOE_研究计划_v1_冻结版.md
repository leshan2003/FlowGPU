# MoE 专家分层放置与刷新研究计划 v1（冻结版）

> **已被取代：** 本文保留为历史审计记录。后续实现和实验以
> [`docs/MOE_高置信度实验设计_v2_冻结版.md`](docs/MOE_高置信度实验设计_v2_冻结版.md)
> 为唯一设计入口。本文中关于 SRAM 主副本、未实现迁移和旧结果包的描述不得直接沿用。

> 冻结日期：2026-10-19  
> 项目路径：`FlowGPU-main`  
> 文档状态：第一阶段模型、变量和实验矩阵冻结；软件模型实验已完成一版。  
> 最新结果见 [`docs/MOE_EXPERIMENT_RESULTS_v1.md`](docs/MOE_EXPERIMENT_RESULTS_v1.md)。

## 0. 冻结结论

MoE 部分经过讨论，已经足够冻结第一阶段的系统语义和实验计划，但还不足以冻结最终性能结论。原因是：

- 尚未接入真实 request-ordered router trace；
- GPU/HBM 硬件参数和 v1 实验配置已存在，但尚未选定唯一的主参考场景，并转成
  完整的 v2 大模型配置；
- KA200 的 device-level 估计参数已存在，但尚未获得原始 kernel、链路、并发曲线
  和功耗测量 artifact；
- 尚未实现动态迁移事件；
- 尚未实现完整超图聚类和显式 `max_experts_per_chip`。

这里不是“文件里完全没有参数”，而是“参数的证据等级、适用场景和曲线完整度
不同”。GPU/HBM 参数大多有 vendor/derived 来源；KA200 的决定性 MoE 参数目前
仍是 estimate；真实路由数据则确实不在仓库中。

第一阶段冻结六项语义：

1. **主场景采用 HBM fallback。** HBM 默认保存所有专家的完整副本；类脑 SRAM 为部分专家增加副本。
2. **热/冷不是 0/1 输入。** 热度是连续的、按层和来源 node 条件化的需求；最终是否增加 SRAM 副本由边际价值决定。
3. **GPU 与类脑共同服务热专家。** 冷专家只由 HBM/GPU 路径服务；热专家同时有 GPU 和类脑副本，由 dispatcher 按当前队列和预计完成时间分流。
4. **共激活按层统计和优化。** 不使用一个全局共激活图，也不把 pair 统计直接等同于完整 top-k 超边。
5. **刷新采用两个时钟。** 统计更新与物理专家迁移分开；物理迁移只在预计净收益覆盖成本和 SLO 干扰时发生。
6. **主刷新形态采用实时动态局部刷新。** 持续更新统计，在安全边界重算局部布局，只迁移净收益高的副本，并通过新旧版本共存完成切换。触发阈值、迁移粒度、批处理方式和全局兜底频率作为策略变量，由实验决定。

第一阶段的主问题是：

> 在 GPU 数量和 HBM 足以保存完整专家的前提下，增加类脑 SRAM 副本、把同层共激活专家聚近，并在需求持续漂移时做最小迁移，能否以更少的类脑 node 达到目标 SLO，并改善 goodput、P99 和单位 node 效率？

动态刷新不是只作为最后阶段的外部扩展，而是主研究问题的一部分。第一阶段需要比较：

- 静态布局；
- 实时局部增量刷新；
- 慢速全局重算；
- 局部增量刷新 + 周期性全局兜底。

不确定的触发方式、局部范围、迁移粒度和服务重叠方式不在计划阶段硬编码，而作为策略参数进入仿真比较。

第一阶段不回答：

> 类脑是否可以通过替代 HBM/GPU 来降低 GPU 数量、capEx 或系统能效。

这个问题需要第二阶段的“独占分层/容量卸载”模式，不能在 HBM fallback 的结论上直接外推。

### 0.1 参考配置与论文证据匹配

真实匹配采用“分层证据”而不是把不同论文的数字混成一个真机配置：

| 对象 | 可作为参考的内容 | 当前证据源 | 使用方式 |
|---|---|---|---|
| GPU/HBM | H100 80 GB、3.35 TB/s；BI-V100 32 GB、1.2 TB/s | `flowgpu/hardware/registry.py`；厂商/datasheet provenance | GPU fallback 和全 GPU 基线 |
| 模型 | DeepSeek-V3：61 层、256 routed experts、top-8、1 shared expert、58 个 MoE 层 | 公开模型配置；`flowgpu/workload/models.py` | 固定主模型，不再混用 projected V4 Flash |
| 全 GPU 系统 | 16/32×H100、NVLink4、IB NDR、5 s TTFT / 50 ms TPOT 参考 SLO | `configs/experiments/grounded_published_hw.yaml` | HBM fallback 的 published-hardware 对照 |
| SRAM 服务机制 | 生产级 SRAM LLM 服务、低 batch、数千 chip、同步 C2C、Qwen3-235B 和 gpt-oss-120B 实测 | 本地论文 [`paper.pdf`](../data/stage1/pdfs/paper.pdf)，SHIP, MLSys 2026 Industry | 验证低 batch/SRAM 机制与网络/容量约束，不冒充 KA200 |
| KA200 公开参数 | 30 个类脑核、25 万神经元、2500 万突触、48 TOPS INT8、24 TFLOPS FP16、约 12 W；支持 ANN/SNN 混合、众核和存储计算融合 | Lynxi 官方 KA200 产品页（2026-10-19 核查）：<https://lynxi.com/lq2001/18.html> | 修正设备身份和公开计算参数 |
| KA200 MoE 决定性参数 | SRAM 容量/带宽、NoC、片间/板间、bridge、`t_expert(M)`、并发、功耗分项 | 仓库只有 estimate 和空测量模板 | 只能作为 KA200 estimate scenario |
| 真实路由行为 | 100k+ 请求、三类 MoE、task-family 变化、prefill/decode 相关性 | Expert Activation Patterns, arXiv:2604.23150 | 用于设定 synthetic priors、实验假设和预期边界 |
| 共激活演化 | pair co-activation 随训练变化；周期全局调整 + 步级局部调整 | Cobalt, arXiv:2609.36959 | 指导刷新策略空间，不直接迁移数值 |
| 在线迁移 | 预测 incoming requests、迁移与计算重叠、近零停机 | Director, arXiv:2607.08782 | 指导版本切换和迁移 overlap 设计 |

注意：官方 KA200 页面没有公开 MoE 所需的 SRAM 容量/带宽、NoC 并发曲线或
`t_expert(M)`。另外，当前 registry 中部分 KA200 计算参数与官方页面不一致，
应在正式实验前修正并保留原始 URL/访问日期。SHIP 的 Groq LPU 不能被重命名为
KA200；它只提供一个“已经实测的 SRAM 推理系统参照”。

推荐的主配置是：

```text
主模型：DeepSeek-V3 fp8
GPU 主参考：16×H100，HBM 保存完整 fallback
GPU 敏感性：32×H100
Brain 目标 profile：KA200 public estimate
SRAM 机制参照：SHIP/Groq LPUv1 measured profile
路由数据：真实 request-ordered trace；未取得前只运行受论文先验约束的 synthetic
```

真实 trace 尚未在本地获得。论文中的统计可以约束 synthetic generator 的偏斜、
task-family 变化和漂移模型，但不能被表述成本项目已经采集的 trace。

---

## 1. 研究范围

### 1.1 模型阶段

- 主模型：DeepSeek-V3 fp8。其他模型只作为泛化实验。
- 主阶段：decode。
- 专家粒度：一个 expert 完整驻留一颗 chip，不在 chip 间拆分单个 expert。
- 第一阶段不模拟 expert TP。
- GPU 保留所有 expert 的 HBM fallback。
- 类脑芯片可以在 SRAM 中保存额外副本。

### 1.2 主要研究区间

定义类脑容量比：

```text
R_brain =
    可用于专家权重的类脑 SRAM 总字节
    /
    所有唯一专家权重总字节
```

第一阶段扫描：

```text
R_brain = 0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5
```

`R_brain=0` 是纯 GPU/HBM 基线。  
`R_brain<1` 表示类脑 SRAM 中所有副本的总字节不足全部唯一专家的一份，
**不表示必须先覆盖所有专家以后才能复制**。

每个专家的脑侧副本数可以是：

```text
r_e = 0, 1, 2, ...
```

当 `R_brain<1` 时，仍需在以下动作之间直接比较：

- 为尚未覆盖的专家增加第一份副本；
- 为已经覆盖的极热专家增加第二份或更多副本。

例如，在 `R_brain=0.1` 时，优化器完全可能把容量给少数极热专家的多份副本，
而让其余专家只保留 HBM fallback。是否这样做由实际边际收益决定，不由 `R_brain`
是否达到 1 决定。`R_brain>=1` 也只是“至少可能覆盖全部专家”，并不自然意味着
后续动作一定是热复制。

### 1.3 HBM 语义

第一阶段默认 HBM 中有每个专家的一份完整主副本或 fallback：

```text
every routed expert: at least one GPU-resident copy
selected experts:     one or more brain-SRAM copies
```

因此：

- “冷专家”表示没有类脑副本，只能由 GPU 执行；
- “热专家”表示有类脑副本，但 GPU 路径仍然合法；
- 一个热专家的 assignment 可以根据当前 GPU 队列、类脑队列和网络成本分流到任一可用副本；
- 第一阶段不释放 HBM 容量。

仍需记录：

- HBM 中专家权重总字节；
- attention 权重；
- KV cache；
- activation 和运行时 buffer；
- 每 GPU 的剩余 HBM。

HBM 足够大只是免除了可行性约束，不意味着 HBM 带宽不重要。

### 1.4 类脑侧资源

第一阶段只建模：

```text
brain chip SRAM
chip compute
chip-wide queue
node 内链路
跨 node 链路
GPU ↔ brain bridge
```

第一阶段**不把类脑 DRAM 作为主动 tier**。原因是现有参数中：

- estimated KA200 DRAM 仅 68 GB/s；
- projected HP300 DRAM 为 200 GB/s；
- 对应 SRAM 分别为 2.4 TB/s 和 51.2 TB/s。

一条 42 MiB 专家从 estimated KA200 DRAM 搬到 SRAM 的理论下界已约 0.65 ms；
从 projected HP300 DRAM 搬迁也要约 0.22 ms。它不适合作为高频 promotion/demotion
路径。类脑 DRAM 只能在以后作为低频 staging、overflow 或故障恢复扩展单独研究，
且这类结论不能混入第一阶段的 HBM fallback 主结论。

---

## 2. 系统实体与决策变量

### 2.1 集合

| 符号 | 含义 |
|---|---|
| `L` | MoE 层集合 |
| `E_l` | 第 `l` 层专家集合 |
| `N` | 类脑 node 集合 |
| `C_n` | node `n` 中的类脑 chip 集合 |
| `W` | 时间窗口或 epoch 集合 |
| `T` | 路由 trace 中的 token/request |

### 2.2 HBM 基线变量

```text
g[l,e] = 1
```

第一阶段固定每个专家在 GPU/HBM 至少有一份副本。

### 2.3 类脑放置变量

```text
x[l,e,c,w] ∈ {0,1}
```

表示窗口 `w` 中专家 `(l,e)` 是否在类脑 chip `c` 有副本。

约束：

```text
副本数：
  r[l,e,w] = sum_c x[l,e,c,w]

每 chip 唯一：
  x[l,e,c,w] <= 1

SRAM 容量：
  sum_(l,e) x[l,e,c,w] * size[l,e]
    <= usable_sram[c]

每 chip 专家数上限：
  C_actual[c,w] =
    sum_(l,e) x[l,e,c,w]
    <= C_max
```

当前 v2 有 SRAM 容量约束，但没有显式 `C_max`。完整实验前必须加入。

### 2.4 dispatch 变量

对每个 assignment：

```text
a[τ,e,r] ∈ {0,1}
```

表示 assignment `τ` 绑定到专家 `e` 的实际副本 `r`。每个 assignment 绑定且只绑定一个副本。

第一阶段 dispatcher 应支持：

- `earliest_finish`：主策略；
- `nearest`：通信优先消融；
- `least_loaded`：队列优先消融；
- `balanced_hybrid`：显式控制 GPU 与类脑之间的分流比例，用于机制实验。

这段公式的直白含义是：

> token 在某一层需要专家 `e` 时，虽然 `e` 可能有 GPU 副本和多个类脑副本，
> 但这一次 assignment 最终只能交给其中一份副本执行。dispatcher 负责根据
> 当前 GPU 队列、类脑队列、bridge 和 node 网络选择哪一份。

### 2.5 刷新变量

```text
u[w] ∈ {0,1}
m[l,e,c,w] ∈ {0,1}
```

- `u[w]=1`：窗口边界允许产生新的 placement version；
- `m[l,e,c,w]=1`：专家副本发生迁移。

迁移约束：

```text
迁移源和目标不能造成重复扣费；
旧副本在在途 assignment 完成前不能回收；
迁移流量和 serving 流量共享 bridge、DRAM 和 node 链路；
迁移期间必须满足双缓冲容量约束。
```

这段公式的直白含义是：

- `u[w]=1` 只是表示这个 epoch 允许发布新布局，不代表一定要迁移；
- `m[l,e,c,w]=1` 表示某一份专家副本真的发生移动；
- 只有 net value 足够高的迁移才应被选中；
- 新副本建立完成前，旧副本仍要服务在途请求。

这里的“刷新”不是每个 token 重新选专家，也不是修改模型参数。它是在时间窗口
边界上决定是否更新布局统计、是否重算放置，以及是否搬运已经存在的一份专家副本。

---

## 3. 输入需求统计

### 3.1 连续热度

对层 `l`、来源 node `n`、专家 `e`、窗口 `w`：

```text
lambda[l,n,e,w] =
    sum of gate_weight or assignment count
```

同时保存：

- 原始 assignment count；
- gate weight sum；
- 请求数；
- token 数；
- 不同 request 数。

主热度不应直接使用固定 `hot_frac`。

### 3.2 node affinity

```text
A[l,n,e,w] =
    lambda[l,n,e,w]
    /
    sum_e' lambda[l,n,e',w]
```

用来判断：

- 某个 node 是否对专家有局部偏好；
- 复制是否能带来 node-local hit；
- 全局热度相同但 node affinity 不同的 workload 是否会有不同布局。

### 3.3 共激活

每个 token 在每层产生一个 top-k 集合：

```text
H = {e1, e2, ..., ek}
```

保存：

- 完整 hyperedge 频次；
- pair 频次；
- pair lift；
- Ochiai similarity；
- hyperedge 的 node fanout；
- hyperedge 在 chip 级别的 cut。

主 placement objective 使用完整 hyperedge；pair affinity 只作为对比算法。

### 3.4 时间漂移

每层、每个主要任务族保存：

- marginal distribution；
- top-hot-set Jaccard；
- node-affinity distance；
- pair/hyperedge distance；
- routing entropy；
- request 内稳定性；
- 跨请求稳定性。

---

## 4. 渐进式分配策略

### 4.1 物理放置离散，价值连续

专家最终只有：

```text
没有脑侧副本
有一份脑侧副本
有多份脑侧副本
```

但排序和分配依据是连续价值，不先执行：

```text
p(e) > threshold -> hot
p(e) <= threshold -> cold
```

### 4.2 候选动作

对当前 placement `P` 产生四类动作：

1. 给一个只有 GPU 副本的专家增加第一份 SRAM 副本；
2. 给一个已有 SRAM 副本的专家增加第二份或更多副本；
3. 将现有 SRAM 副本移动到另一个 chip/node；
4. 删除一份低价值 SRAM 副本。

每次选择后重新评估所有候选的边际价值。

### 4.3 候选价值

从 detailed model 或 placement-window event model 估计：

```text
Benefit(action, P) =
    T_P(placement-window)
  - T_P_after_action(placement-window)
```

同时记录：

```text
Cost(action) =
    migration_time
  + migration_bandwidth_interference
  + temporary_SRAM_usage
```

静态阶段先使用：

```text
score = Benefit / action_bytes
```

动态阶段使用：

```text
net_value =
    expected_benefit_over_epoch
  - migration_cost
  - SLO_interference
  - energy_cost
```

### 4.4 阶段扫描

对每个 `R_brain` 重新优化，不把上一个容量点的布局硬搬过来。

不要把以下区域当成固定顺序；它们是由同一组候选动作竞争后可能出现的现象：

```text
少量 SRAM
  - 可能覆盖少量最高价值专家
  - 也可能复制极少数高价值专家
  - 由 first-copy 与 extra-replica 的边际价值决定

中等 SRAM
  - first-copy 与 extra-replica 同时竞争

较大 SRAM
  - 仍可能优先复制头部专家
  - 也可能开始覆盖此前的冷专家

SRAM 富余
  - 冷专家覆盖基本完成
  - 副本主要用于负载均衡和 locality
```

必须同时保留两种极端策略作为对照：`coverage-first` 和 `replica-first`，以及
一个在每个动作后重新排序的 `joint-marginal` 策略。因此 `R_brain<1` 时同样会
出现复制决策。

### 4.5 连续变化实验

至少做三类：

1. **容量连续扫描**：只增加 `R_brain`。
2. **任务混合连续扫描**：

```text
P_alpha(e) = alpha * P_domain_A(e)
           + (1-alpha) * P_domain_B(e)

alpha = 0, 0.1, ..., 1.0
```

3. **batch/load 连续扫描**：固定布局评价不同 batch，或重新优化后比较。

任务是：

- 识别布局相变；
- 判断边界专家是否稳定；
- 测量一次刷新需要多长的持续收益期。

---

## 5. 共激活聚类与 C

### 5.1 聚类目标

同一 token 在某一层选中多个专家时，希望减少：

- 不同目标 node 数；
- 不同目标 chip 数；
- 跨 node dispatch/combine；
- 最忙 chip 的完成时间。

不能只优化 pair affinity。

### 5.2 聚类与 C 的 trade-off

增大 `C_max` 可能：

- 让更多共激活专家位于同一 chip；
- 减少 token fanout；
- 减少要求的 chip 数。

同时可能：

- 增加单 chip 串行专家数；
- 增加 chip queue；
- 减小每个副本的 `M`；
- 使分组 GEMM 效率下降；
- 使热点更集中。

因此 `C_max` 必须作为实验变量，而不是固定常数。

### 5.3 必须比较的算法

| 算法 | 作用 |
|---|---|
| random same-capacity | 清除容量和副本数因素后的负对照 |
| popularity placement | 只使用边际热度 |
| node-affinity placement | 只使用局部性 |
| pair coactivation | 当前 v2 风格 |
| full-hyperedge clustering | 完整 top-k 超边 |
| joint value | 热度 + affinity + hyperedge + queue |
| oracle | 小拓扑上界 |

### 5.4 概率和层级

不预设一个跨模型通用的共激活概率阈值。需要：

- 按层分别拟合；
- 按任务族分别统计；
- 报告 pair support、lift、Ochiai；
- 对阈值做 50/75/90/95 分位或显式 value cutoff 扫描；
- 检查低支持度但高 lift 的 pair 是否真正影响性能。

---

## 6. 刷新设计

### 先说清楚：刷新是三个动作

“刷新”不是每 token 重新摆放专家，而是以下控制链：

```text
1. 统计刷新
   更新最近窗口的热度、node affinity、共激活
   -> 只改变“我们对当前需求的估计”，不改硬件状态

2. 布局重算
   用新统计评价 P0，必要时求 P1
   -> 只产生候选方案，仍可不执行

3. 物理刷新/迁移
   只把 P0 -> P1 中收益足够高的专家副本搬过去
   -> 这里才消耗 bridge/DRAM/网络带宽并影响 serving
```

例如：

```text
最近 200 个请求显示 code 变少、math 变多
→ 统计刷新（无需迁移）
→ 重算后发现 40 个专家的理想位置变化（无需迁移）
→ 预计只搬其中 8 个专家即可获得大部分收益
→ 比较这 8 个专家的搬运成本和服务干扰
→ 只有净收益为正才真正迁移
```

### 冻结的刷新主路径

第一阶段的主路径确定为：

```text
持续统计
  -> 检测到需求变化
  -> 局部候选布局重算
  -> 估计局部迁移的持续收益与完整成本
  -> 若净收益为正，选择少量专家副本
  -> 在安全窗口后台迁移
  -> 新副本完整后按层或 node 发布新版本
  -> 在途请求继续旧版本
  -> 没有旧请求后回收旧副本
```

这不是硬实时保证。它是软实时、可回退的在线重配置：

- 不在每个 token 上搬运专家；
- 不要求一次重算就迁移全部变化；
- 不假设迁移不影响 serving；
- 不允许新副本未完成就切换；
- 不要求每次局部调整后仍是全局最优。

为了避免长期停留在次优布局，系统保留周期性或 drift 触发的全局重算。全局重算只产生候选方案，是否进行大范围迁移仍由同一套成本收益规则决定。

### 不确定项作为策略参数

以下内容不预先宣称哪一种最好，而是进入策略空间：

| 维度 | 可选策略 |
|---|---|
| 触发时机 | 每 prompt、每轮对话、大任务边界、agent subtask 边界、固定周期、drift 阈值 |
| 检测窗口 | 最近 request 数、固定秒数、EMA、任务块 |
| 重算范围 | 单专家、单层、层组、单 node、全局 |
| 迁移粒度 | 单副本、专家集合、层级、node 级 |
| 迁移调度 | 立即逐个迁移、批量迁移、优先级队列、在 compute-bound 窗口迁移 |
| 服务模式 | stop-the-world、background、compute-overlap |
| 版本发布 | 单专家、逐层、node 级、全局 |
| 稳定性控制 | 最小 epoch、冷却时间、迟滞、收益置信下界 |
| 全局兜底 | 从不、固定周期、长期 drift 触发 |

这些维度共同形成动态策略族，而不是在文档阶段猜测唯一答案。

### 6.1 两个时钟

#### 统计时钟

更新：

- 专家热度；
- node affinity；
- 共激活图；
- drift 指标。

可以使用滑动窗口或 EMA。

#### 迁移时钟

只在以下条件成立时迁移：

```text
expected_benefit
  >
migration_bytes / effective_bandwidth
  + interference_penalty
  + safety_margin
```

统计可以频繁更新，迁移不频繁。

### 6.1.1 推荐的触发粒度

不同粒度承担不同职责：

| 粒度 | 做什么 | 不做什么 | 建议 |
|---|---|---|---|
| 每个 token / layer | 产生 assignment，dispatcher 选择已有副本 | 不重新布局、不迁移 | 始终允许 |
| 每个请求/prompt | 收集路由统计；多轮对话中可视为自然边界 | 不保证每个 prompt 都迁移 | 允许检测，不默认执行 |
| 每个 decode step | 观察短期局部热度 | 不适合全局重排 | 只用于统计/调度 |
| 每个对话轮次 | 用户可能切换任务，可用 prefill 激活预测 decode | 不适合无条件迁移 | 适合 task-aware 候选触发 |
| 每个大任务/agent episode | 任务方向稳定，适合重布局 | 不适合每步重排 | 推荐的主触发边界 |
| 周期窗口 | 按固定请求数或时间观察分布 | 不宜太频繁 | 作为对照和兜底 |

建议的主策略是：

```text
每层、每请求持续记录统计
    ↓
在用户轮次、大任务或 agent episode 边界检查 drift
    ↓
只有预期新布局的持续收益 > 迁移成本时才迁移
    ↓
优先迁移变化最大、价值最高的少量专家
    ↓
按层独立切换版本，避免所有层同时产生迁移拥塞
```

对于连续多租户 serving，不应把单个用户的 prompt 直接当成全局刷新点，
因为很多对话会交错。此时应使用：

- request/task family 的滚动窗口；
- 当前 batch 的任务混合变化；
- 稳定时间超过阈值后，在 decode iteration 或层边界做一次候选重算。

对于单个长对话，优先在用户轮次边界判断；对于 agent，优先在 goal/subtask
切换边界判断。token 级只做 assignment，不触发全局迁移。

### 6.2 可迁移动作

第一阶段只考虑：

1. **首次覆盖**：GPU HBM → 某颗脑侧 chip SRAM；
2. **增加副本**：GPU HBM → 另一颗脑侧 chip SRAM；
3. **node 内副本重排**：同 node 不同 chip；
4. **跨 node 重排**：改变 locality 和聚类。

类脑 DRAM 的 promotion/demotion 不在第一阶段主实验内。

### 6.3 对照策略

| 策略 | 含义 |
|---|---|
| Never refresh | 当前静态基线 |
| Periodic | 固定 token/step/time 间隔 |
| Task-boundary | task family 改变时 |
| Drift-triggered | 分布距离超过置信阈值 |
| Local-incremental | 每次只迁移最高净价值副本 |
| Oracle | 使用未来信息的上界 |

### 6.4 迁移服务模式

| 模式 | 语义 |
|---|---|
| Stop-the-world | 暂停相关服务，保守基线 |
| Background | 迁移与推理共享链路 |
| Compute-overlap | 只在安全窗口迁移，乐观上界 |

主实验使用 Background；Stop-the-world 和 Compute-overlap 作为边界。

### 6.5 版本一致性

```text
旧副本服务在途请求
新副本后台建立
完整后发布新 placement version
新请求使用新版本
旧请求完成后回收旧副本
```

如果双缓冲不足，必须产生显式 stall 或 fallback，不能暗中假设免费容量。

---

## 7. 关键对比实验

### E0：语义与实现正确性

**目的**：确认 HBM fallback、GPU/类脑分流和指标定义正确。

对照：

- 全 GPU；
- 全部专家一份 GPU fallback；
- 1 层、2–8 chip 的小拓扑。

检查：

- assignment 守恒；
- GPU 和类脑 assignment 均合法；
- bridge 收费；
- SRAM 容量；
- event 时间单调；
- 小规模 oracle。

### E1：容量前沿

**目的**：在任意 `R_brain` 下，联合比较“给未覆盖专家第一份副本”和
“给已覆盖专家增加副本”，找出实际分配前沿。

固定：

- GPU/HBM 配置；
- 模型；
- trace；
- node 大小；
- batch；
- SLO。

扫描：

- `R_brain`；
- 每 chip `C_max`；
- 类脑 node 数。

对照：

- 无类脑；
- random copies；
- top-p；
- coverage-80/90%；
- `unique-first`；
- `hot-replica-first`；
- 连续边际价值；
- oracle。

输出：

- 覆盖专家数；
- 热副本数；
- GPU/类脑 assignment 比例；
- node fanout；
- P99；
- goodput；
- `tokens/s/node`；
- J/token；
- SRAM 利用率。

通过条件：

- 相同 node 数和同 GPU 下，相对全 GPU 或 `unique-balanced` 的收益显著；
- 同收益下使用更少类脑 node；
- 不是通过增加额外硬件获得。

#### E1b：`R_brain<1` 时的复制能力

单独扫描低容量区域：

```text
R_brain = 0.02, 0.05, 0.1, 0.2, 0.35
```

每个容量点比较：

1. `coverage-first`：尽量覆盖更多不同专家；
2. `replica-first`：尽量给最高价值专家增加副本；
3. `joint-marginal`：每一步直接比较两种动作的边际收益。

输出“唯一覆盖专家数 × 总副本数 × goodput”的三维前沿，并明确：

> 在 `R_brain<1` 时复制是否优于扩大覆盖。

这项实验不能等到 `R_brain>=1` 才开始。

### E2：热复制 vs 冷覆盖

**目的**：验证在相同 SRAM 增量下，哪种动作更值得。

控制：

- 相同模型；
- 相同 GPU/HBM；
- 相同类脑 SRAM 字节；
- 相同 node 数。

对照：

- 只做热复制；
- 只做冷覆盖；
- 热覆盖后热复制；
- 热覆盖后冷覆盖；
- 联合边际价值。

关键变量：

- routing skew；
- GPU 队列；
- batch；
- expert `M`；
- node fanout。

预期不是固定答案，而是找边界：

```text
skew 强、GPU 忙、batch 小时，热复制可能更值；
coverage 低、fallback 多、batch 大时，冷覆盖可能更值。
```

### E3：共激活聚类与 C

**目的**：区分“共同热门”和“真正共激活”。

控制：

- 相同 `p(e)`；
- 相同副本数；
- 相同 SRAM；
- 相同 node 数。

切换：

- pair support；
- lift；
- Ochiai；
- full hyperedge；
- `C_max=1,2,4,8,16`；
- node 大小 4/8/16/32。

比较：

- random placement；
- popularity placement；
- node-affine placement；
- pair clustering；
- hyperedge clustering；
- joint placement。

输出：

- fanout node/chip；
- local assignment fraction；
- hyperedge cut；
- busiest chip queue；
- grouped `M`；
- completion/TPOT；
- `tokens/s/node`。

通过条件：

- 在固定边际热度后，聚类仍带来收益；
- 收益不是仅由副本数或额外 node 造成；
- C 的最优点随 node/batch 改变，且能用事件模型解释。

### E4：node 与 batch 的交互

**目的**：确定 node 和 batch 如何改变最优布局。

扫描：

- batch：1、8、32、128、512、1024；
- node 数：例如 1、2、4、8、16、32，按模型可行性设置；
- chips/node：4、8、16、32；
- ingress：distributed/concentrated；
- `C_max`。

必须报告：

- absolute goodput；
- `tokens/s/node`；
- locality；
- P99；
- 能耗；
- 容量比。

不能把 strong scaling 和 weak scaling 画成同一条曲线。

停止规则：

- 找到满足 SLO 的最小 node 数；
- 再测试两个更大的 node 数；
- 若连续两组收益 CI 上界低于 5%，或 `tokens/s/node` 持续下降，判为饱和。

### E5：漂移与刷新

**目的**：回答刷新是否值得，而不是先假设需要刷新。

构造三类 trace：

1. 稳定单任务；
2. 两个任务族按 `alpha` 连续混合；
3. task family 突变或 burst。

比较：

- Never；
- Periodic；
- Task-boundary；
- Drift-triggered；
- Local-incremental；
- Local-incremental + periodic global fallback；
- Global-only；
- Oracle。

并至少拆开以下策略维度做消融：

- 触发时机：prompt、turn、task、agent subtask、period、drift；
- 重算范围：local vs global；
- 迁移粒度：expert vs layer vs node；
- 迁移调度：immediate vs batched vs compute-window；
- 版本发布：per-layer vs global；
- 最短驻留时间、冷却期和滞回阈值。

记录：

- 迁移字节；
- 迁移完成时间；
- P99；
- SLO violation；
- goodput loss；
- 回本步数/秒数；
- 刷新后净收益；
- 刷新频率；
- 迁移专家比例。

通过条件：

- 在真实或受控漂移下，刷新策略的累计 goodput 显著高于 Never；
- 迁移成本已进入同一条服务时间线；
- 收益在迁移后持续足够久；
- 局部刷新能获得大部分全局重排收益，同时显著降低迁移字节和服务抖动；
- 若局部刷新长期落后于全局刷新，必须报告差距及何时需要全局兜底；
- 若没有正收益，必须报告“静态更优”的边界。

### E6：消融与稳健性

逐项关闭：

- 共激活目标；
- node-affinity；
- 热复制；
- GPU/类脑混合 dispatch；
- earliest-finish；
- dispatch dedup；
- combine reduce；
- multicast；
- 动态刷新；
- detailed network。

敏感性：

- bridge 带宽和延迟；
- 类脑 SRAM 容量和带宽；
- 类脑计算峰值；
- HBM 带宽；
- kernel `t_expert(M)`；
- SRAM reserve；
- `C_max`；
- node 大小；
- trace sampling。

---

## 8. 指标

### 8.1 主指标

- SLO-constrained goodput；
- 满足 SLO 的最小类脑 node 数；
- `tokens/s/node`；
- TPOT P50/P99；
- TTFT P99。

### 8.2 机制指标

- GPU/类脑 assignment 比例；
- 每副本 assignment 数；
- 每副本 grouped `M` 分布；
- fanout node/chip；
- remote/local assignment fraction；
- busiest chip queue；
- hyperedge cut；
- 热专家第一份覆盖比例；
- 第 2 份副本数；
- 冷专家覆盖比例；
- HBM fallback 数。

### 8.3 成本指标

- J/token；
- 类脑静态功耗；
- bridge/scale-up/scale-out 能量；
- GPU/HBM 使用；
- 类脑 node/chip 数；
- migration bytes；
- migration energy；
- 迁移额外 P99 和 goodput loss。

---

## 9. 可信度规则

### 9.1 数据切分

必须按时间切分：

```text
placement window: W0
evaluation window: H0
later drift window: W1/H1
```

随机打乱 token 后的 train/test 不能用于研究漂移。

### 9.2 统计单位

- 以 request 或 task block 作为 bootstrap 单位；
- 不把同一请求的 token 当作独立样本；
- 多随机种子；
- paired comparison；
- 报告 95% CI；
- 固定 trace 使用 block bootstrap，不使用 seed CI 冒充 trace CI。

### 9.3 判定

- `supported`：paired speedup CI 下界 > 1；
- `practically supported`：下界 > 1.05；
- `unsupported`：CI 上界 <= 1；
- `inconclusive`：CI 跨 1。

能耗和延迟 P99 使用各自的单侧或双侧门限，必须预先声明。

### 9.4 证据等级

```text
measured > vendor > paper > derived > projected
```

如果结论由 projected 参数决定，只能称为情景分析。

---

## 10. 当前完成与未完成

### 已完成

- v2 静态专家级仿真骨架；
- placement/evaluation 分离；
- expert/chip/replica/assignment 数据模型；
- SRAM 容量和 reserve；
- uniform/weighted/node-affine synthetic routing；
- strict JSONL trace loader；
- 完整 top-k hyperedge 统计；
- 六种 placement baseline；
- GPU fallback；
- `unique-first` / `hot-replica-first`；
- max-min fluid network；
- grouped `t_expert(M)`；
- continuous batching、SLO capacity、多 seed CI；
- 107 项测试通过。
- GPU/HBM 参数已集中在 `flowgpu/hardware/registry.py`，包括 H100、BI-V100
  的 HBM 容量和带宽；
- 已有 `grounded_published_hw.yaml`、`pd_a_reference.yaml` 和 `regimes.yaml`
  三套 GPU/数据流系统配置；
- KA200 已有 device-level 估计值：48 MB SRAM、2.4 TB/s SRAM 带宽、
  68 GB/s DRAM、50/25 GB/s 片间/板间链路；官方页面公开 24 TFLOPS FP16、
  48 TOPS INT8、30 核和约 12 W；
- 本地 SHIP 论文提供生产级 SRAM LLM 系统的实测证据，可用于机制对照。
- 新增 `flowgpu/sim_v2/research.py`：task-conditioned community trace、
  HBM-fallback 的 coverage/replica/joint-marginal 分配、局部增量计划；
- 新增渐进式 `ranked-replication`：按复制预留比例在覆盖与热复制之间
  分配容量，并提供 `ranked-adaptive` 根据 placement-window 的 top-decile
  流量集中度自动选择复制强度；
- 新增 `scripts/run_moe_research.py`：容量、skew、coactivation/C、
  node×batch、refresh、seed CI 和敏感性实验；
- 新增 `scripts/analyze_moe_research.py` 和可复现结果图；
- 修复 Python 3.8 下 `math.nextafter` 不可用导致的网络事件兼容问题；
- 当前测试总数 112，全部通过。

### 部分完成

- 连续需求排序：有 `assignments/bytes` 排序，但没有完整边际价值；
- 热复制：有 baseline，但没有在同一容量点联合比较 first-copy 与
  extra-replica 的 `joint-marginal` 策略；
- 共激活：保存 hyperedge，但放置使用 pair clique；
- GPU 与类脑共同执行：dispatcher 理论上可选，但 GPU 仍是单个聚合 tier；
- 结果 bundle：对比 API 和顶层 CLI 输出不一致。
- 动态刷新：已实现 epoch 级局部增量布局、并行 bridge 惩罚和收益门控；
  尚未实现后台 migration flow 与 serving 流量的共享争用；
- 完整 hypergraph partition：当前聚类仍使用 affinity 加权的贪心放置；
- 超大拓扑：full detailed sweep 已验证到 64 nodes；更大拓扑在当前
  Python flow scheduler 下出现数小时和多 GB 内存，暂不宣称支持。

### 未完成

- 真实 request-ordered trace；
- request/task block bootstrap；
- 显式 `C_max`；
- 完整 hypergraph partition；
- 多 GPU fallback pool 建模；
- 动态统计更新；
- drift detector；
- 迁移事件；
- 双缓冲和版本切换；
- 迁移与 serving 流量竞争；
- 唯一 v2 主参考配置的选择与固化；
- KA200 原始校准 artifact；当前只有 estimate profile 和空测量模板；
- 当前 registry 与 KA200 官方页面之间计算参数不一致的修正；
- KA200 的 `t_expert(M)`、multicast/fanout、reduce/fanin、bridge 和完整功耗曲线；
- 真机端到端验证。

类脑 DRAM tier 已从第一阶段范围移除，只保留为低频 staging/overflow 的可选后续研究。

准确的现状表述是：

> 已具备并执行静态专家放置、连续覆盖/复制分配、受控共激活、node/batch、
> 局部刷新和参数敏感性的软件模型实验；真实 trace、KA200 实测和完整后台迁移
> 争用仍缺失。

---

## 11. 实施顺序

### P0：冻结本文件

- 以本文件作为 MoE 第一阶段的主计划；
- 旧文档中的固定 `hot_frac` 只作为历史 baseline；
- HBM fallback 和独占分层分开报告。

### P1：最小代码补齐

1. 增加 `max_experts_per_chip`；
2. 新增在任意容量点联合比较 first-copy、extra-replica 和 move/delete 的
   `joint-marginal` allocator；
3. 把 tiered plan 的 GPU fallback 容量计入 attention/KV reserve；
4. 把 GPU fallback 从单端点扩展为多 GPU pool；
5. 输出完整的 per-case result bundle。

### P2：静态主实验

先跑 E0、E1、E2、E3、E4 的 synthetic/projected 版本，定位相变和候选前沿。

### P3：真实 trace

接入 request-ordered trace，重新跑 E1–E4，输出真实热度、node affinity、共激活和漂移。

### P4：动态刷新

实现实时动态局部刷新主路径：

- 统计/EMA 与 drift detector；
- 局部候选布局重算；
- migration event；
- 双缓冲和版本切换；
- background/compute-overlap；
- migration 与 serving 流量竞争；
- 局部迁移策略与全局兜底策略。

随后跑 E5，并用实验决定触发频率、迁移粒度、局部范围和服务重叠方式。

### P5：硬件校准

采集 KA200 kernel、bridge、NoC、并发和功耗，跑 E6 与真机验证。

---

## 12. 最终交付

第一阶段的完整结果应至少包含：

1. 静态容量前沿图；
2. 热覆盖/热复制/冷覆盖的相变；
3. pair vs hyperedge 聚类对照；
4. C × node × batch 的相图；
5. 真实 trace 的热集稳定性和 node affinity；
6. 静态 vs 刷新的累计 goodput 曲线；
7. 迁移成本、SLO violation 和 refreshed epoch 的分布；
8. 完整 placement table、逐层结果和 key-path 证据；
9. measured/projected 参数敏感性和结论边界。

只有这些结果完成，才能决定：

- 第一阶段的类脑异构是否值得；
- 采用 HBM fallback 还是独占分层；
- 是否需要在真实部署中加入动态刷新；
- 是否需要继续投入 KA200 硬件测量。
