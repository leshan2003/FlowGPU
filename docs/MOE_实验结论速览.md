# MoE 实验结论速览

## 一句话结论

在当前的合成 workload 和投影硬件仿真中：

> HBM 保留全部专家、类脑 SRAM 额外存放热点专家，是最稳定的方案。
> 单纯优先复制热点、类脑独占专家、共激活聚类和动态刷新，尚未得到确定收益。

当前实验是：

```text
合成 workload + 投影/估计硬件
```

不是 KA200 真机实验，也不是真实业务 trace。

## 先把术语翻译成人话

| 术语 | 这里到底指什么 |
|---|---|
| HBM | GPU 自带的高带宽显存。本实验默认里面保存全部专家 |
| 类脑 SRAM | 类脑芯片上的高速片上存储，比走 HBM 和 GPU-类脑 bridge 更适合放热点 |
| GPU-only | 对照组：只有 GPU/HBM，没有类脑加速 |
| Overlay | “叠加副本”模式：HBM 仍保存全部专家，同时把少量热点专家额外复制到类脑 SRAM |
| H-exclusive | “专家分层独占”模式：热专家只放类脑，其他专家只放 GPU，不做另一层 fallback |
| Coverage | 先把更多不同的专家各覆盖一份 |
| Replica | 优先给最热的少数专家增加第二份或更多副本 |
| Joint | 每增加一份副本时，比较“新增一个专家”与“给热专家再加副本”哪个更划算 |
| Static 加速 | 固定 workload、固定 placement 下，用总完成时间计算的加速 |
| Serving 加速 | 连续 decode 服务下，用满足 SLO 的请求吞吐计算的加速 |
| SLO goodput | 每秒完成了多少同时满足 TTFT 和 TPOT 时限的请求 |
| C | 每颗类脑芯片最多放多少个专家 |
| Batch | 同时进入 continuous batching 的请求数 |
| Refresh / Migration | 需求变化后重新计算 placement，并搬运专家权重 |
| 95% CI | 20 个 paired seed 给出的均值不确定性区间 |

本表中的 “Overlay serving 加速”具体是：

```text
Overlay 的 SLO goodput
────────────────────────
GPU-only 的 SLO goodput
```

例如 `1.033×` 表示：

> 在相同 workload、相同 100 ms TTFT/TPOT 门槛下，Overlay 每秒完成的合格请求数比 GPU-only 多 3.3%。

它不是“每个 token 的延迟一定降低了 3.3%”，也不是“类脑整机比 GPU 快 3.3%”。

## “场景”到底是什么意思

本文中的“场景”表示：

```text
使用哪个模型
+ 测试了模型的前几层
+ 专家访问有多集中
```

它不是 node 数、batch 数或类脑副本数。那些是另一组实验变量。

| 表中场景 | 模型和层数 | 专家访问分布 |
|---|---|---|
| Mixtral Zipf 0.5 | 完整 32 层 | 比较平均，少数专家只略热 |
| Mixtral Zipf 1.3 | 完整 32 层 | 中等偏斜 |
| Mixtral Zipf 2.0 | 完整 32 层 | 明显偏斜，少数专家非常热 |
| Qwen 四层 | 前 4 个 MoE 层 | Zipf 1.3 |
| DeepSeek 四层 | 前 4 个 MoE 层 | Zipf 1.3 |
| gpt-oss 四层 | 前 4 个 MoE 层 | Zipf 1.3 |

Zipf 越小，专家访问越平均；Zipf 越大，访问越集中在少数专家。
Node × batch 表和 C 扫描表使用的都是 Mixtral Zipf 1.3；表中每一行改变的是
node、batch 或 C，而不是模型场景。

## 实验完成了吗

计划内的仿真实验已经完成：

- 4 个模型：Mixtral、DeepSeek-V3、Qwen3、gpt-oss
- 完整 9 点 node × batch 矩阵
- 9 点 C 隔离实验
- 6 个关键场景，每个 20 个 paired seeds
- HBM overlay、类脑独占、Brain DRAM staging
- 副本策略、聚类、迁移和版本切换
- 140 个测试通过
- 35 个最终结果目录，源码哈希差异为 0

尚未完成的是：

- 真实 request-ordered router trace
- KA200 真机 kernel、链路、并发和功耗测量
- 完整模型、长窗口的动态刷新收益实验
- 真实 TCO

## 控制变量和比较对象

不同表格回答不同问题，不能混在一起解释。

| 实验组 | 保持不变 | 主动变化 | 比较 baseline | 指标 |
|---|---|---|---|---|
| 20-seed 策略比较 | 2 nodes、batch 2、16 copies、C=2、相同模型和 workload | 随机 seed | GPU-only | Coverage 看总完成时间；Overlay 看 SLO 请求吞吐 |
| Node × batch | Mixtral 完整 32 层、5 seeds、16 copies、C=2 | nodes=`1/2/4`，batch=`1/2/4` | 每个实验点各自的 GPU-only | 总完成时间和 SLO 请求吞吐 |
| C 隔离 | workload、copy budget、node/chip 数固定 | 每芯片专家上限 C | 每个实验点各自的 GPU-only | 总完成时间和 SLO 请求吞吐 |
| 聚类隔离 | 相同 copy multiset、C=1、相同 node 和访问序列 | unique、pair、full-hyperedge | unique placement | event-model completion |
| 迁移重放 | 模型、hardware、placement 版本固定 | 迁移流量是否与 serving 共享链路 | 无迁移的 serving flow | 迁移时间、逐 serving-flow 延迟和总 makespan |

必须明确的控制边界：

1. `20-seed` 是 20 个随机 workload，不是 20 个 node。
2. `20-seed` 表固定为 2 个 node、batch 2；node 扫描是另一组实验。
3. Qwen、DeepSeek、gpt-oss 的四层结果不能和 Mixtral 的完整 32 层结果直接比较。
4. 两组 C 扫描的 node/chip/copy 总数不同，不能把两个 panel 的点混成一条曲线。
5. Coverage 列和 Overlay 列使用不同指标，不能直接相减，也不能据此说某一个策略
   在同一指标上更好。
6. GPU-only baseline 只使用 16 个 GPU；默认 Overlay 还增加了 16 个类脑 chips。
   因此两端到端结果表示“增加类脑层后的收益”，不是相同总硬件资源下的效率比较。
7. node/batch 增加时总硬件也会增加。该实验回答系统趋势，不能直接证明
   “每节点效率”或“每瓦效率”提高。

## 主要结果

### 1. “20-seed”是什么意思

一个 seed 就是一套随机生成的专家访问序列。

“20-seed”表示：

> 用 20 套不同的随机 workload 重复同一个实验，每个 seed 同时跑所有策略，最后对 20 个配对结果取均值。

它不是：

- 20 台机器；
- 20 个类脑 node；
- 20 次真实请求 trace。

20-seed 表使用固定配置：

```text
2 个类脑 node
batch = 2
16 份类脑专家副本
C = 2
```

其中 Mixtral 使用完整 32 层；Qwen、DeepSeek、gpt-oss 只使用前 4 个 MoE 层。

### 2. 静态总完成时间：Coverage 与 GPU-only

`总完成时间加速`定义为：

```text
GPU-only 完成时间
──────────────────
Coverage 完成时间
```

| 场景 | Coverage 相对 GPU-only |
|---|---:|
| Mixtral Zipf 0.5 | 1.053× |
| Mixtral Zipf 1.3 | 1.085× |
| Mixtral Zipf 2.0 | 1.081× |
| Qwen 四层 | 1.112× |
| DeepSeek 四层 | 1.081× |
| gpt-oss 四层 | 1.176× |

例如 `1.085×` 表示：

> Coverage 完成同一批固定专家访问所需的时间，是 GPU-only 的约 `1 / 1.085 = 92.2%`。

### 3. 连续 serving：Overlay 与 GPU-only

`SLO 请求吞吐加速`定义为：

```text
Overlay 每秒完成的合格请求数
──────────────────────────
GPU-only 每秒完成的合格请求数
```

| 场景 | Overlay 相对 GPU-only |
|---|---:|
| Mixtral Zipf 0.5 | 1.026× |
| Mixtral Zipf 1.3 | 1.033× |
| Mixtral Zipf 2.0 | 1.036× |
| Qwen 四层 | 1.085× |
| DeepSeek 四层 | 1.074× |
| gpt-oss 四层 | 1.051× |

例如 gpt-oss 的 `1.051×` 表示：

> Overlay 每秒完成的 SLO 合格请求数比 GPU-only 多 5.1%。

这里比较的是**请求吞吐**，不是单个 token 的延迟。

注意：上面的 Coverage 列和 Overlay 列使用不同指标，不能直接相减，也不表示
“Coverage 之后又叠加了 Overlay”。两张表都以 GPU-only 为同一个 baseline。

结论：

- HBM + 少量类脑 SRAM 副本有稳定正收益；
- 收益主要来自减少热点专家的 HBM/bridge 访问；
- 类脑不是替代 HBM，而是作为高速热点层更合理。

### 4. 单纯复制热点收益较小

Replica-first 在大多数场景只有约 `0.3%–3%` 的收益，明显弱于 Coverage。

不能只因为专家很热就一直复制。还要看：

- 复制后能减少多少远端访问；
- 是否造成额外排队；
- 是否挤掉了其他专家的覆盖机会。

### 5. Joint 不一定最好

Joint 理论上更精细，但当前实验中：

- Mixtral：Joint 与 Coverage 基本相同；
- DeepSeek：基本相同；
- gpt-oss：Coverage 更好；
- Qwen：Joint 均值略高，但置信区间重叠。

因此目前不能声称 Joint 一定优于 Coverage。

### 6. 类脑独占不是普遍更快

H-exclusive 表示：

- 热专家只放类脑；
- 其他专家只放 GPU；
- 不再保留另一层 fallback。

20-seed SLO 请求吞吐结果：

| 场景 | H-exclusive 相对 GPU-only |
|---|---:|
| Mixtral Zipf 0.5 | 1.025× |
| Mixtral Zipf 1.3 | 1.015× |
| Mixtral Zipf 2.0 | 1.022× |
| Qwen 四层 | 1.026× |
| DeepSeek 四层 | 1.007× |
| gpt-oss 四层 | 1.041× |

其中只有 Mixtral Zipf 0.5 和 gpt-oss 四层的 95% CI 下界大于 1；其余场景
不能确认性能改善，但也没有足够证据证明会稳定退化。

还需要区分三个不同的“类脑模式”：

1. H-exclusive：热专家在 SRAM，冷专家仍在 GPU/HBM；
2. B-stage：全部专家在类脑 DRAM，热子集 staging 到 SRAM；
3. “全部专家常驻 SRAM”：本轮没有执行完整模型的该模式。

因此，本轮既没有证明“全部权重都放类脑更好”，也没有证明“更差”。
当前能确认的只是：选择性 SRAM 副本最稳定；H-exclusive 只有条件性收益；
B-stage 只完成了单层 grouped staging 语义实验，不是完整模型性能实验。

因此当前最稳妥的选择仍是：

```text
GPU/HBM 保留全部专家
类脑/SRAM 额外存放热点副本
```

### 7. Node 和 Batch 是另一组独立实验

20-seed 表固定为 2 个 node、batch 2。

Node × batch 实验单独测试了全部 9 个组合：

| Nodes / Batch | Coverage 总完成时间 | Overlay SLO 请求吞吐 |
|---|---:|---:|
| 1 / 1 | 1.061× | 1.019× |
| 1 / 2 | 1.087× | 1.026× |
| 1 / 4 | 1.073× | 1.032× |
| 2 / 1 | 1.069× | 1.044× |
| 2 / 2 | 1.103× | 1.032× |
| 2 / 4 | 1.058× | 0.981× |
| 4 / 1 | 1.078× | 1.046× |
| 4 / 2 | 1.059× | 1.025× |
| 4 / 4 | 1.043× | 1.030× |

增加 node 会带来更多远端通信和调度开销；增加 batch 会改变排队、分组计算和副本负载。
因此必须同时比较吞吐、P99、SLO 和 node efficiency。

### 8. C 实验具体测了什么

在相同 64 copies / 64 chips 资源下：

- C=1：总完成时间加速约 `1.275×`；
- C=2 及以上：总完成时间加速约 `1.262×`。

进一步检查 5 个 seed 的实际布局：

| C | 使用多少个类脑 chip | 每 chip 最多专家数 | 与 C=2 的布局 |
|---:|---:|---:|---|
| 1 | 64 | 1 | 不同 |
| 2 | 42–48 | 2 | 基准 |
| 4 | 42–48 | 2 | 完全相同 |
| 8 | 42–48 | 2 | 完全相同 |
| 14 | 42–48 | 2 | 完全相同 |

所以在当前 workload 中：

- planner 实际最多只把 2 个专家放在同一个 chip；
- C=4/8/14 根本没有成为活跃约束；
- C=1 使用 64 个 chips，C≥2 使用 42–48 个 chips，二者布局不同；
- C=1 没有出现额外串行退化，反而略快，但不能外推为所有 workload 都如此。

### 9. 聚类实验具体测了什么

最初 Stage 0 的微基准使用：

```text
8 experts、2 nodes×4 chips、C=1、12 tokens
```

后来审计发现，三种方法在这个微基准中实际生成了完全相同的 chip 布局，
因此 completion 都是 `19.2007 µs`。这只能证明路径可执行，不能证明聚类好坏。

为此新增了有区分度的 C=1 压力实验：

```text
3 nodes × 4 chips
9 experts
3 个互斥的 3-expert hyperedge
30 tokens
固定硬件、副本数、C=1 和 token 数
```

实验分为三类，不能混为一谈：

1. aligned：三个社区固定，并分别由对应 node 发起请求；
2. home permutation：共激活社区固定，只改变三个社区由哪个 node 发起；
3. community permutation：把 expert 重新分成三个同等大小的社区。

在 community 与 home node 完全对齐时：

| 策略 | Node cut | 本地执行比例 | Completion |
|---|---:|---:|---:|
| unique-balanced | 40 | 55.6% | 20.951 µs |
| pair-clustered | 40 | 55.6% | 20.951 µs |
| full-hyperedge | 0 | 100% | 16.151 µs |

此例中 full-hyperedge 的完成时间加速为 `1.297×`。

固定相同社区、只排列 6 种 home-node 对应关系时：

- full-hyperedge 的 node cut 在 6 种情况下都为 0；
- 但本地执行比例分别出现 `100% / 33.3% / 0%` 三种结果；
- 只有 home node 正好对齐的那一种达到 `1.297×`；
- 其余 5 种 completion 基本不变。

这说明：

> “三个专家放在同一个 node”只消除了它们彼此之间的 fanout；
> 如果请求来自另一个 node，所有请求仍然要跨 node。

在 20 个随机 community 划分中：

- pair-clustered 在 `20/20` 个案例中与 unique-balanced 布局完全相同；
- full-hyperedge 把平均 node cut 从 `32.38` 降到 `9.52`；
- 但平均完成时间加速只有 `1.015×`；
- 20 个案例中只有 1 个明显改善；11 个有小于 `0.002%` 的细微下降，其余基本不变；
- 原因是当前 full-hyperedge planner 优化 node fanout，却没有把每个 request
  的 home-node affinity 直接放进目标函数；只把专家聚到一起，不代表聚到了
  正确的 node。

因此当前准确结论是：

> 聚类机制在“同层共激活”且“社区靠近请求来源”同时成立时能产生明确收益；
> 仅把共激活专家放在一起、但忽略 home node 时，node cut 的降低不会稳定转化为
> completion 改善。

详细数据：
[`moe_2026_c1_cluster_stress_v4`](../experiments/moe_2026_c1_cluster_stress_v4/)

### 10. GPU 和类脑可以并行，但粒度有限

在 HBM overlay 模式下：

- 同一专家的不同 token/assignment 可以分别由 GPU 和类脑处理；
- 一个 token 的一次 expert 计算不会拆成 GPU/类脑两半。

也就是说，并行粒度是：

```text
不同 assignment 分到不同设备
```

不是：

```text
同一个 expert kernel 做跨设备张量并行
```

## 刷新实验做到什么程度

已经实现：

- 历史统计窗口；
- drift 检测；
- placement 重算；
- source-aware 迁移；
- migration 和 serving 共享网络带宽；
- 新 placement 完成后再激活；
- 在途请求继续使用旧版本。

当前迁移成本：

| 模型 | 迁移数据量 | 迁移时间 |
|---|---:|---:|
| Mixtral | 2688 MiB | 34.409 ms |
| DeepSeek | 1344 MiB | 13.765 ms |
| Qwen | 576 MiB | 5.900 ms |
| gpt-oss | 403.42 MiB | 4.133 ms |

但目前只重放了前两个 serving layer 的代表流量，没有完成完整模型的长期 refresh 实验。

因此当前只能说：

- 刷新机制已经跑通；
- 迁移成本可以计算；
- 还不能回答多久刷新一次最好；
- 还不能给出完整模型 refresh 的净收益和 P99。

## 最终建议

当前最值得继续投入的方向：

1. 以 `HBM fallback + selective SRAM overlay` 为主方案；
2. 以 Coverage 作为主 baseline；
3. 保留 Joint 作为可调策略，但不能预先假定它最好；
4. 选 node 和 batch 时必须联合看 SLO，不做单调外推；
5. 引入真实 router trace 后，再重点研究聚类与动态刷新；
6. 拿到 KA200 实测数据后，再做最终硬件与 TCO 判断。

## 详细资料

- 完整审计：
  [`MOE_最终实验执行与审计_v1.md`](MOE_最终实验执行与审计_v1.md)
- 实验目录：
  [`experiments/README.md`](../experiments/README.md)
- 结果矩阵：
  [`MOE_仿真完整矩阵结果_v1.md`](MOE_仿真完整矩阵结果_v1.md)
