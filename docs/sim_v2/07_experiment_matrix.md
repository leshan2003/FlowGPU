# 实验矩阵

## 1. 两阶段执行

### Stage A：fast sweep

目标是定位相变点、无效区和候选最优策略。

```text
batch:          1, 8, 32, 64, 128
capacity R:     1.0, 1.25, 1.5, 2.0, 4.0
chips/node:     4, 8, 16, 32
ingress ratio:  1/N, 1/4, 1
routing:        uniform, popularity, node-affine, coactivation, drift
strategies:     unique, packed, popularity, ingress, node-load, coactivation, joint
```

完整笛卡尔积不必全部进入 detailed。

### Stage B：detailed verification

选择：

- 每种策略的最好点；
- 结论翻转边界；
- 典型现实参数；
- 最强反例；
- fast/detailed 差异最大点。

运行显式拓扑、flow 争用和 continuous serving。

## 2. 公平比较

在同一个 node 数下：

- 硬件、reserve、routing trace、随机种子相同；
- 主副本和副本都计容量；
- 不允许策略隐式增加设备或 bridge lanes；
- 所有策略使用相同调度能力，除非做明确消融。

跨 node 数只称为 scaling study，并报告：

- absolute throughput；
- `tokens/s/node`；
- `J/token`；
- capex/power（若 profile 提供）。

必须把两种 scaling 语义分开：

- strong scaling：总 token/request 数固定，只增加 node；
- weak scaling：每个 cluster node 的 token/request 数固定，总负载随 node 数增长。

两者不得画在同一条“加 node 的收益”曲线上。每个 case 都使用独立 seed 的 placement
trace 和 held-out evaluation trace；evaluation trace 禁止参与放置。

synthetic routing 必须显式选择 `uniform` 或 `weighted`。weighted 模式分别保存
placement/evaluation expert weights；两组权重不同即为 routing-drift 实验，不得把
evaluation 权重反馈给 placement。

## 3. 机制实验

固定 batch 用于回答：

- touched experts；
- assignments per replica；
- fanout chips/nodes；
- scale-up/scale-out 字节；
- hottest-chip makespan；
- replica locality hit rate；
- coactivation cut；
- batch fragmentation。

这些结果不能直接替代服务容量。

## 4. Serving 实验

至少包含：

- Poisson 与 trace-driven arrivals；
- prompt/output length 分布；
- continuous batching；
- admission control；
- warm-up；
- p50/p90/p99；
- SLO goodput/capacity；
- 多随机种子置信区间。

## 5. Ablations

必须分别关闭：

- node-local replication；
- same-node load replication；
- coactivation objective；
- dispatch chip/node dedup；
- combine chip/node reduce；
- earliest-finish dispatcher；
- network contention；
- kernel `M` 查表；
- routing drift。

## 6. 敏感性

至少扫描：

- SRAM reserve 10/15/20%；
- scale-up/scale-out/bridge 带宽与延迟；
- ingress 数量；
- routing skew；
- node affinity；
- coactivation strength；
- drift interval；
- kernel calibration uncertainty。
