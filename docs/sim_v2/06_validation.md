# 验证

## 1. 验证阶梯

### Tier 1：守恒与单元测试

每次运行验证：

- 每个 routed token 产生恰好 top-k 个 routed assignments；
- shared assignment 数正确；
- assignment 不丢失、不重复执行；
- 放置只引用真实副本；
- SRAM 容量不超限；
- dispatch/combine 字节可还原；
- FLOPs 与 kernel 调用一致；
- 能量分项可加和；
- 事件时间单调且依赖满足。

### Tier 2：小拓扑 oracle

对 2–8 颗芯片的小实例使用手算、穷举或 ILP：

- 最优副本位置；
- 最小跨链路字节；
- 最小 makespan；
- greedy 与 oracle gap。

自动 gap 报告必须使用与启发式完全相同的专家、芯片、副本预算和目标函数。若搜索空间
超过声明上限，应报告 `oracle_available=false, search_complete=false`，不得将未执行或
被截断的搜索称作最优。oracle 目标为零时只报告绝对 gap，避免伪造无穷大的相对百分比。

### Tier 3：Microbenchmark

逐项验证 kernel、p2p、multicast、reduce、bridge 和并发拥塞预测。

### Tier 4：端到端真机

使用小模型或少 node 部署比较：

- per-layer latency；
- TPOT；
- throughput；
- link/chip bottleneck；
- 功耗。

## 2. 必需反例

下列场景中优化器不应虚构收益：

- uniform routing 且无 node affinity；
- `R = 1` 无副本空间；
- scale-up 与 scale-out 成本相同；
- 禁用 dispatch 去重与 local combine；
- 专家计算完全主导且负载均匀；
- 热点变化快于静态放置窗口；
- 副本拆分使 `M` 下降并显著降低 kernel 效率；
- 新增 node 只增加硬件而策略不变。

## 3. Fast/detailed 一致性

fast 模式用于筛选，detailed 模式用于裁决。报告必须给出关键点两者的：

- 延迟差；
- 吞吐差；
- 网络字节差；
- bottleneck 是否一致。

若 fast 改变结论方向，必须扩大 detailed 覆盖，不能继续用 fast 做结论。

## 4. 证伪标准

只有当策略：

1. 同硬件击败 `unique-balanced`；
2. held-out 有效；
3. detailed 有效；
4. 收益超过误差区间；
5. 单位硬件和能耗代价透明；
6. 在合理参数区间稳定；

才可称为 supported。否则分别标记：

```text
unsupported
conditional
inconclusive
```

## 5. 回归

v2 测试不得修改 v1 的历史数值语义。v1 与 v2 通过显式 adapter 连接；任何迁移都必须有
双跑报告，而不是静默替换。
