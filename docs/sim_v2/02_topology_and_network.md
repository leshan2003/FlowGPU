# 拓扑与网络模型

## 1. 拓扑图

v2 MUST 用显式有向图表示系统。节点类型至少包括：

```text
gpu
brain_chip
local_switch
node_switch
fabric_switch
```

链路至少记录：

```yaml
src: <endpoint>
dst: <endpoint>
bandwidth: <bytes/s>
latency: <seconds>
duplex: full|half
efficiency: <0..1>
max_inflight: <optional>
energy_per_byte: <J/B>
provenance: ...
```

模板生成器 MAY 创建规则拓扑，例如 `N nodes × C chips/node`，但展开后的图 MUST 可检查。

## 2. Node 与 ingress

node 是 scale-up locality domain，不是计算设备。专家必须落到 node 内具体芯片。

配置 MUST 显式给出：

- `chips_per_node`；
- GPU 到 brain node 的连接矩阵；
- ingress node 集合；
- token 的 home-node 规则；
- 是否允许 GPU 直接发送到目标专家 node。

拓扑未知时必须扫描 ingress ratio，不得假定 token 均匀出现在所有 node：

```text
I/N = 1/N, 1/4, 1
```

## 3. Flow-level 模型

网络事件是 flow：

```text
Flow(
  source,
  destinations,
  bytes,
  route,
  startup_latency,
  traffic_class
)
```

并发 flow MUST 竞争：

- 共享链路带宽；
- 发送端 injection bandwidth；
- 接收端 bandwidth；
- half-duplex 方向资源（若适用）。

MVP 不模拟 packet，但 MUST 模拟 flow 的开始、阻塞、带宽重分配和完成。

## 4. 路由

必须支持：

- 静态最短时延路由；
- 静态最少跳路由；
- 显式用户路由表；
- 可选 ECMP。

fast 模式 MAY 用分层聚合近似；detailed 模式 MUST 使用显式路径。关键结果必须在 detailed
模式复核。

## 5. Dispatch 与 combine

以下是独立硬件/runtime 能力开关：

```yaml
dispatch_dedup_chip: false
dispatch_dedup_node: false
combine_reduce_chip: false
combine_reduce_node: false
hardware_multicast: false
```

它们决定通信量，不能从“专家相邻”自动推导。

### 5.1 无去重

一个 token 的 dispatch 字节近似：

```text
top_k * hidden_dim * comm_dtype_bytes
```

### 5.2 Chip 去重

若同一 token 在同一芯片命中多个专家：

```text
distinct_destination_chips * hidden_bytes
```

### 5.3 Node 去重

跨 node 部分为：

```text
distinct_destination_nodes * hidden_bytes
```

进入目标 node 后仍需支付 node 内分发。

combine 只有在相应 local weighted reduction 可用时才能按芯片或 node 减少返回份数。

## 6. Collective

v2 不得把所有 expert dispatch 强制建模成全 world all-to-all。它 MUST 从实际 assignment
生成 point-to-point/multicast flow；collective 是可选实现策略。

若使用 collective，`collective_world` 必须等于真实参与者，并记录空消息、启动延迟和算法。

## 7. 网络守恒

每次运行 MUST 验证：

- 每个非本地 assignment 有对应 dispatch 路径；
- 每个需要返回的结果有对应 combine 路径；
- 所有链路发送字节可由 flow 汇总还原；
- 去重前后字节差能由能力开关和目标集合解释；
- 不存在零成本跨拓扑移动。

