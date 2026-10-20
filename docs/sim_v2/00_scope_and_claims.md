# FlowGPU Simulator v2：范围与声明

> **MoE 研究口径更新：** 本文是早期通用 v2 规范，其中“所有专家至少一份
> SRAM 主副本”和 MVP 不含迁移的描述不再是当前 MoE 主模式。MoE 设计与实验以
> [`docs/MOE/02_实验方法与可信度.md`](../MOE/02_实验方法与可信度.md)
> 为准；当前主模式是 HBM fallback + selective brain-SRAM overlay。

## 1. 状态

本文档组定义 FlowGPU Simulator v2（下称 v2）的规范。关键词
`MUST`、`SHOULD`、`MAY` 分别表示必须、建议和可选。

v2 与当前解析执行器并存：

- 当前执行器是 `v1-fast`，用于历史结果、粗粒度扫描和回归对照；
- v2 是显式专家副本、显式拓扑和 flow-level 离散事件模拟器；
- v1 的 `shards/groups/ep` 结果不得作为 v2 的 ground truth。

## 2. 核心研究问题

在固定 node 形态、给定 node 数、分层网络和 MoE 路由分布下，当所有专家至少已有一个
SRAM 主副本时，新增 SRAM 应如何用于专家复制与放置，才能改善延迟、吞吐或通信，同时
付出怎样的容量、能耗和单位硬件效率代价？

v2 的目标是判断专家复制和共激活放置**何时有效**，不是证明它们有效。负结果是有效结果。

## 3. 必须区分的数量

v2 MUST 分别表示下列概念，禁止复用一个 `shards`：

| 名称 | 含义 |
|---|---|
| `physical_devices` | 系统真实存在的设备 |
| `allocated_devices` | 分配给一个模型实例/副本的设备 |
| `resident_replicas` | 实际常驻的完整专家副本 |
| `active_devices` | 当前窗口内执行工作或传输的设备 |
| `collective_world` | 某次 collective 的真实参与端点 |
| `token_fanout_chips` | 一个 token 本层到达的不同芯片数 |
| `token_fanout_nodes` | 一个 token 本层到达的不同 node 数 |
| `max_chip_load` | 最忙芯片上的 assignment/服务时间 |

网络扇出、活动设备数和最忙芯片负载不得用同一个聚类因子表示。

## 4. 研究区间

定义容量倍率：

```text
R = 全系统可用于权重的 SRAM / 所有唯一专家权重
```

- `R < 1`：容量不足区间，研究热冷分层、HBM 后备和溢出；
- `R >= 1`：全覆盖区间，研究主副本放置和剩余 SRAM 的副本分配；
- 两个区间 MUST 分开报告，不得把消除 spill 的收益归因于专家复制。

## 5. 证据等级

每个设备、链路、kernel 和工作负载参数 MUST 携带：

```text
measured > vendor > paper > derived > projected
```

建议字段：

```yaml
provenance:
  confidence: measured|vendor|paper|derived|projected
  source: <human-readable citation>
  artifact: <optional local path or URL>
  measured_at: <optional timestamp>
  machine_revision: <optional identifier>
```

若决定结论方向的参数仍为 `projected`，结论只能称为情景分析，不能称为真机预测。
`lynxi_hp300_proj` 在 v2 报告中 MUST 显示为 hypothetical profile。

## 6. MVP 范围

MVP 包含：

- decode 阶段；
- 完整专家驻留于单颗芯片；
- 静态副本；
- 显式 expert-to-chip 放置；
- 逐 token-expert assignment 调度；
- flow-level 分层网络；
- 固定 batch 机制实验和 continuous-batching 服务实验；
- synthetic trace 与真实 trace 接口；
- fast 与 detailed 两种运行模式。

MVP 不包含：

- 训练；
- 每 batch 动态搬移专家权重；
- 故障恢复；
- packet/cycle-level NoC；
- 跨请求权重一致性；
- 单个专家跨芯片 TP 分片；
- prefill 优化；
- speculative decoding/MTP；
- RTL 生成。

## 7. 声明规则

任何“方案有效”的声明 MUST 同时满足：

1. 在同一 node 数、同一硬件下击败 `unique-balanced`；
2. 在 held-out routing trace 上仍有效；
3. 在 detailed 网络模式下仍有效；
4. 收益大于模型误差和参数不确定区间；
5. 报告 `tokens/s/node`、`J/token` 和副本容量；
6. 在合理参数区间成立，而非只在单个理想点成立。

以下旧结论仅是待验证 hypothesis：

- `网络 ÷ C、计算 × C`；
- `C ≈ 3.4`；
- 共激活可压到 1–2 颗芯片；
- 聚类快 43% 或慢 33%；
- 聚合 bridge/all-to-all 可代表真实 node 拓扑；
- `1024 × hp300_proj` 是真机配置。
