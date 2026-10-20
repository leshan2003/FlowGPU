# FlowGPU Simulator v2 文档索引

v2 是独立于历史 `flowgpu.sim` 的专家级模拟器。规范中的 `MUST` 表示最终验收要求，
不代表当前代码已经全部实现。

## 规范

1. [范围与声明](00_scope_and_claims.md)
2. [执行语义](01_execution_semantics.md)
3. [拓扑与网络](02_topology_and_network.md)
4. [专家放置与复制](03_expert_placement_replication.md)
5. [路由模型](04_routing_models.md)
6. [校准](05_calibration.md)
7. [验证](06_validation.md)
8. [实验矩阵](07_experiment_matrix.md)
9. [结果契约](08_result_contract.md)
10. [已知限制](09_known_limitations.md)
11. [KA200 校准缺口](10_ka200_measurement_gap.md)
12. [软件侧收口与验收](11_software_readiness.md)

## 当前实现状态

代码位于 `flowgpu/sim_v2/`，不替换 v1。

| 能力 | 状态 | 说明 |
|---|---|---|
| provenance 等级与校验 | foundation 已实现 | `measured` 必须引用原始 artifact |
| expert/chip/replica/assignment schema | foundation 已实现 | 完整副本驻留单芯片 |
| 主副本、去重、SRAM 容量约束 | foundation 已实现 | reserve 在可用权重 SRAM 中扣除 |
| routing trace 与 uniform generator | foundation 已实现 | placement/evaluation/oracle split 显式分离 |
| weighted/Zipf/node-affine routing | foundation 已实现 | 可复现的加权无放回 top-k；不同窗口权重可表达漂移 |
| strict router JSONL loader + YAML 接入 | foundation 已实现 | request/token/step/layer/home/top-k/gates；按 `{case_id}` 选择 placement/evaluation 文件，禁止跨拓扑静默重映射 |
| `lambda[layer,node,expert]` | foundation 已实现 | evaluation trace 禁止用于放置 |
| `unique-balanced` | baseline 已实现 | 每专家一个主副本 |
| `layer-packed` | baseline 已实现 | 同层主副本尽量使用最少芯片 |
| `popularity-replicated` | baseline 已实现 | 只使用全局热度 |
| `ingress-local` | baseline 已实现 | 使用 node 条件需求，每专家每 node 至多一份 |
| `node-load-replicated` | baseline 已实现 | 先补第一份 node-local 副本，再以不同芯片上的同专家副本分摊计算 |
| `coactivation-clustered` | baseline 已实现 | 不增加副本，贪心降低共激活 cut |
| `joint-greedy` | baseline 已实现 | 由调用者提供完整 event-model objective |
| `small-oracle` | Tier-2 已实现 | 小于等于 8 芯片的有界穷举验证 |
| heuristic/oracle gap audit | Tier-2 已实现 | 同目标报告绝对/相对 gap、最优性及搜索是否完整 |
| 显式有向拓扑和层级拓扑生成器 | foundation 已实现 | 固定 chips/node，可扫描 node 数与 ingress |
| shortest-latency / fewest-hop | foundation 已实现 | 稳定 tie-breaker |
| unloaded flow 与逐链路计量 | foundation 已实现 | 可计算时间、字节、能量 |
| serialized-path network | reference 已实现 | 保守串行基线，明确标记 `is_detailed=False` |
| max-min fluid detailed network | foundation 已实现 | flow 开始/完成时动态重分配带宽 |
| injection / receive / half-duplex | foundation 已实现 | 与路径链路共同参与 max-min 共享 |
| fixed-window MoE layer DAG | foundation 已实现 | dispatch、grouped kernel、芯片队列和 combine 可重叠 |
| calibrated `t_expert(M)` | foundation 已实现 | 插值与外推 warning |
| analytic kernel fallback | foundation 已实现 | 自动标记为 derived/extrapolated |
| nearest / least-loaded / earliest-finish | foundation 已实现 | 同副本 assignments 形成一个 `M` batch |
| GPU-origin / home-node-resident ingress | foundation 已实现 | GPU-origin 强制计 bridge，不允许免费进入 node |
| deterministic event queue | foundation 已实现 | 使用规范中的稳定事件键 |
| chip dispatch dedup / chip local combine | foundation 已实现 | combine 必须提供正的校准 reduction 延迟 |
| node dispatch dedup / node local combine | foundation 已实现 | scale-out 与 node 内 fanout/partial flow 分开计量 |
| hardware multicast | foundation 已实现 | 显式最短路树，公共前缀一次、分支逐边传输 |
| `max_inflight` admission | foundation 已实现 | 稳定准入，完成后释放并重算 fair share |
| Tier-1 结果守恒审计 | foundation 已实现 | flow 重建链路字节、能量、关键路径与 assignment |
| result artifact bundle | foundation 已实现 | 原子写入 9 项产物，含输入 SHA-256 |
| benchmark artifact loader | foundation 已实现 | schema、机器身份、重复次数与 SHA-256 强制校验 |
| kernel calibration builder | foundation 已实现 | measured `t_expert(M)` 曲线及缺失 M warning |
| network affine calibration | foundation 已实现 | 拟合 latency/bandwidth、误差门槛与 derived provenance |
| fast/detailed 双跑 | foundation 已实现 | 复用 placement/kernel/path，方向翻转产生强 warning |
| decode continuous batching | foundation 已实现 | 迭代边界准入、逐层 MoE DAG、TTFT/TPOT/throughput/SLO |
| Poisson/trace arrival + multi-seed CI | foundation 已实现 | JSONL、可复现随机种子和小样本 Student-t 95% CI |
| SLO capacity load-grid | foundation 已实现 | 仅以 95% CI 下界连续达标的负载前缀认定容量，并报告右删失/非单调 |
| iteration-boundary admission control | foundation 已实现 | 有界 outstanding queue；拒绝请求计入 offered SLO attainment/goodput |
| projected prefill interference | foundation 已实现 | 外部 compute block windows；decode 在迭代/层边界让行并报告 blocked time |
| fixed-chips/node 的 node-count sweep | foundation 已实现 | 分开构造 distributed/concentrated ingress，记录容量 R 与公平性指纹 |
| capacity-regime gate | foundation 已实现 | `R<1` spill、`R≈1` 唯一副本边界、`R>1` 复制候选；R 不替代碎片检查 |
| GPU–brain tiered placement | foundation 已实现 | 每专家 GPU fallback；对照 unique-first 与 hot-replica-first 的脑侧容量用途 |
| tiered detailed runner | foundation 已实现 | GPU/brain 副本使用各自 kernel 曲线并共享显式 bridge 拓扑 |
| tiered node-count report | foundation 已实现 | 冷专家覆盖优先 vs 热副本优先，报告覆盖数/副本/性能/能耗 |
| held-out fixed-window strategy runner | foundation 已实现 | 同 case/trace/runtime 对照六种 placement，并逐层执行 detailed DAG |
| node-count comparison runner | foundation 已实现 | 汇总 tokens/s、tokens/s/node、R、locality、能耗与相对 unique speedup |
| serving node-count sweep | foundation 已实现 | node × strategy × offered-load × seed 的 SLO capacity/goodput 矩阵 |
| node-count sweep bundle | foundation 已实现 | 校验单一 per-node 硬件指纹并原子保存跨 case 指标 |
| placement/evaluation multi-seed gate | foundation 已实现 | paired speedup Student-t CI；supported/unsupported/inconclusive |
| serving/uncertainty/tiered bundles | foundation 已实现 | 保存全 load-grid CI、goodput/admission、paired-speedup 与冷热覆盖对照；均记录输入 SHA-256 |
| strict YAML + CLI | foundation 已实现 | `v2-node-sweep` / `v2-node-serving` / `v2-node-uncertainty` / `v2-node-tiered`；未知字段失败 |
| comparison result bundle | foundation 已实现 | 原子保存 case/trace/策略/逐层/placement/警告与输入 SHA-256，禁止覆盖 |
| system energy accounting | foundation 已实现 | 显式 profile 下分解 compute dynamic/network/全配置芯片 idle 与 J/token |
| 真机校准与端到端验证 | 未完成 | 当前不得声称真机加速 |

## Ingress 语义

`home_node` 表示该 token 本层进入专家系统时所关联的 node。执行器必须另外声明激活起点：

- `home-node-resident`：router 输出已经在 home node，专家 dispatch 从该 node 开始；
- `gpu-origin`：router 输出仍在 GPU，先沿 bridge 到指定 home/ingress node，再进入
  node 内或 scale-out 网络，combine 反向返回。

两者必须分开报告。仅在拓扑中画出 bridge、但把 flow 直接从 home node 开始，会漏算
GPU↔brain 通信。

## 当前可做与不可做

当前代码已经覆盖 synthetic/projected 条件下的容量、tiering、放置、复制、详细网络、
fixed-window、continuous serving、SLO capacity、能耗账、多 seed 结论门和结果落盘，可用于
生成“软件模型内”的完整实验及反例。

它仍不足以生成 KA200 真机主结论。剩余证据缺口是：

1. KA200 真机 kernel、链路、并发、reduce/multicast 和功耗 artifact；
2. Tier 3 microbenchmark 与 Tier 4 端到端真机验证；
3. 当前 prefill 仅为显式 block-window 敏感性模型，不是实机联合调度；
4. packet/cycle-level NoC、动态迁移、单专家 TP 等明确列为范围外扩展。

## 可执行入口

```text
python -m flowgpu v2-node-sweep CONFIG -o NEW_DIR
python -m flowgpu v2-node-serving CONFIG -o NEW_DIR
python -m flowgpu v2-node-uncertainty CONFIG -o NEW_DIR
python -m flowgpu v2-node-tiered CONFIG -o NEW_DIR
```

完整 synthetic/projected 示例见
[`configs/sim_v2/node_sweep_smoke.yaml`](../../configs/sim_v2/node_sweep_smoke.yaml)。
输出目录必须不存在，写入采用临时目录加原子替换，避免把中断运行误认成完整结果。
