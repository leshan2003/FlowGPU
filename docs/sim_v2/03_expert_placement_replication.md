# 专家放置与复制

## 1. 放置变量

对层 `l`、专家 `e`、芯片 `c`：

```text
x[l,e,c] in {0,1}
```

表示完整专家副本是否驻留在该芯片。MVP 不允许一个专家副本跨芯片分片。

每个专家至少有一个主副本：

```text
sum_c x[l,e,c] >= 1
```

## 2. 容量约束

每颗芯片：

```text
sum(x[l,e,c] * weight_bytes[l,e])
  + activation_buffers[c]
  + routing_metadata[c]
  + communication_buffers[c]
  + runtime_reserve[c]
<= sram_capacity[c]
```

默认 reserve MAY 为 15%，但 MUST 做 10%、15%、20% 敏感性分析。真实测量到来后由
measured profile 覆盖。

容量跨所有 MoE 层统一结算。禁止每层独占整池 SRAM。

聚合容量比：

```text
R = sum(usable chip SRAM) / sum(unique expert weight bytes)
```

- `R < 1`：必须进入 GPU/HBM spill 或 tiering 模型，不能运行纯驻留副本对照；
- `R ≈ 1`：只处在唯一副本的聚合容量边界；
- `R > 1`：才有复制的聚合余量。

`R >= 1` 只是必要条件，不是充分条件；完整专家不可跨芯片时，单专家大小和装箱碎片仍可
使放置失败。

## 3. 复制粒度

- 同一芯片保存同一专家两份：MVP 禁止；
- 一个 node 的第一份副本：主要提供 locality；
- 同一 node 的额外副本：只提供计算负载均衡；
- 跨 node 副本：可能同时提供 locality 与负载均衡。

全局冷专家默认只有主副本；但 replica planner 使用 node 条件需求，不强制“冷专家永不复制”。

## 4. 需求统计

核心统计量：

```text
lambda[layer,node,expert]
```

表示从某 home node 出发的 assignment 到达率。还需要：

- per-layer marginal popularity；
- pair/hyperedge coactivation；
- burstiness；
- placement window 与 evaluation window 的漂移。

## 5. 边际收益

在 node `n` 新增专家 `e` 副本的候选价值：

```text
delta =
  reduced_scaleout_critical_time
  + reduced_replica_queue_time
  - batch_fragmentation_penalty
  - capacity_shadow_price * expert_weight_bytes
```

正式优化器 SHOULD 使用事件模型估计 `delta`。解析式只用于候选预筛选。

候选排序可采用：

```text
score = delta / expert_weight_bytes
```

当 SRAM 尚未饱和时，capacity shadow price 可以为零；饱和后必须体现被挤出候选的机会成本。

## 6. 必须实现的策略

| 策略 | 定义 |
|---|---|
| `unique-balanced` | 每专家一份，均衡容量和预计计算负载 |
| `layer-packed` | 同层专家紧凑放置 |
| `popularity-replicated` | 按全局热度复制 |
| `ingress-local` | 按 `lambda[layer,node,expert]` 放置 node-local 副本 |
| `node-load-replicated` | 先补 node-local 第一份，再在同 node 不同芯片复制热点以削减队列 |
| `coactivation-clustered` | 共激活专家邻近，不增加副本 |
| `joint-greedy` | 按事件模型边际收益联合选择 |
| `small-oracle` | 小拓扑 ILP/穷举最优，仅用于验证 |

所有策略 MUST 从相同主副本容量、相同 reserve 和相同硬件开始。

## 7. 共激活

共激活关系应表示成带权 hypergraph：

- 顶点：专家；
- hyperedge：一个 token 的 top-k 专家集合；
- 权重：该集合或其近似在 placement window 的频率；
- 容量约束：芯片和 node 的可用 SRAM；
- 平衡约束：预计 assignment/compute load。

减少 hyperedge cut 只是目标之一；它不得覆盖负载均衡和容量目标。

## 8. 输出

每次运行 MUST 保存：

- 完整 expert-to-chip 副本表；
- 主副本与额外副本标记；
- 每 node/芯片容量占用；
- 每个副本收到的 assignment 数；
- replica locality hit rate；
- 因 locality、负载或共激活产生的边际收益分解。
