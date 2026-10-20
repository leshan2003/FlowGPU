# 执行语义

## 1. 实体

v2 的最小工作单元是一个 token 在一层中对一个专家的一次 assignment：

```text
Assignment(token_id, layer_id, expert_id, gate_weight, home_node)
```

一个 routed expert 可以有多个物理副本：

```text
Replica(replica_id, layer_id, expert_id, chip_id, weight_bytes)
```

运行时 dispatcher 把每个 assignment 绑定到一个已存在的副本。副本选择不得引用评估
窗口的未来路由。

## 2. 每层事件 DAG

每个 MoE 层 MUST 遵循：

```text
router
  -> dispatch flow(s)
  -> replica-local grouped GEMM(s)
  -> optional chip-local weighted reduction
  -> optional node-local reduction
  -> combine flow(s)
  -> layer completion
```

不同 token、专家、芯片和网络 flow MAY 并发。一个 token 只有在本层所有必需结果完成归约
后才能进入依赖该结果的下一层。

## 3. 同专家多 token

同一调度窗口内绑定到同一副本的 assignment MUST 合并为 grouped/batched GEMM。计算时间
使用：

```text
t_expert(layer, expert_shape, dtype, M)
```

其中 `M` 是该副本收到的 token 数。禁止默认：

```text
t_expert(M) = M * t_expert(1)
t_expert(M / replicas) = t_expert(M) / replicas
```

实测查表优先；缺测量时才使用带 provenance 的解析回退模型。

## 4. 芯片执行资源

每颗芯片至少包含以下逻辑资源：

- compute engine；
- SRAM read/write bandwidth；
- network injection/receive ports；
- 可选 local reduction engine。

同一芯片上多个专家事件是否并发由设备 profile 声明。即使允许并发，总计算吞吐和 SRAM
带宽也 MUST 共享，不能让每个专家独占整颗芯片峰值。

## 5. Shared expert

shared expert MUST 是独立实体：

- 每个 token 都产生 shared assignment；
- 它有独立副本、调度和 `t_expert(M)`；
- 不得再用 `groups = routed_experts_touched + shared_experts` 平均分摊；
- shared expert MAY 采用不同的复制或分片策略，但 MVP 只实现完整副本。

## 6. 关键路径

层完成时间由事件依赖与资源排队共同产生，而不是：

```text
network + compute + SRAM
```

也不是无条件：

```text
max(network, compute, SRAM)
```

v2 MUST 从事件 DAG 中恢复关键路径，并在结果中输出关键路径上的事件、排队时间和资源。

## 7. Runtime dispatcher

若一个专家存在多个副本，dispatcher SHOULD 选择预计完成时间最早的副本：

```text
argmin(
  dispatch_arrival
  + current_replica_queue
  + predicted_t_expert(M_after_assignment)
  + predicted_combine
)
```

必须提供至少三种可比较策略：

- `nearest`：优先通信最近；
- `least_loaded`：优先副本队列最短；
- `earliest_finish`：联合通信与计算。

## 8. 确定性

给定相同配置、routing trace、随机种子和模拟器版本，事件顺序及最终结果 MUST 可重现。
同时间事件必须有稳定 tie-breaker，例如：

```text
(timestamp, event_priority, token_id, layer_id, replica_id, sequence_number)
```

