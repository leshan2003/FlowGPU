# D0 下 C、batch、node 与聚类结果 v1

> 日期：2026-10-20
>
> 证据等级：synthetic routing + projected hardware
>
> dispatch：D0-shared

## 1. C 实验的有效控制

早期 C=1/2/4 扫描使用 8 个唯一 hot experts，最小 8-chip 拓扑仍能放下，
因此 C 不可能成为约束，结果无效。

本轮改为：

```text
copy budget：32
C：1, 2, 4
nodes：1, 2, 4
batch：1, 4, 16
模型：Mixtral 完整 32 层
seeds：101–105
```

有效 hot copy 数：

| Nodes | C=1 | C=2 | C=4 |
|---:|---:|---:|---:|
| 1 | 8 | 16 | 24 |
| 2 | 16 | 24 | 24 |
| 4 | 24 | 24 | 24 |

注意：copy budget 是 32，但 Mixtral 每 chip 容量只能容纳约 3 个完整 expert，
所以 1-node、C=4 实际只能放 24 个 unique hot experts。

## 2. C × node × batch 结果

M2-tier-exclusive，D0-shared，相对 GPU-only 的 completion 加速：

| Nodes | Batch | C=1 | C=2 | C=4 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1.049 | 0.992 | 1.105 |
| 1 | 4 | 1.103 | 1.077 | 1.190 |
| 1 | 16 | 1.095 | 1.136 | 1.188 |
| 2 | 1 | 1.054 | 1.101 | 1.101 |
| 2 | 4 | 1.075 | 1.199 | 1.199 |
| 2 | 16 | 1.121 | 1.212 | 1.212 |
| 4 | 1 | 1.002 | 1.002 | 1.002 |
| 4 | 4 | 1.166 | 1.166 | 1.166 |
| 4 | 16 | 1.231 | 1.231 | 1.231 |

### 结论

1. **C 的主要作用是决定固定 copy budget 下能覆盖多少 hot experts。**
   C 不是简单的串行加速参数。

2. **4 nodes 时 C 完全无效。**
   32 个 Brain chips 足以容纳全部 24 个唯一 hot copies，C=1/2/4 布局相同。

3. **2 nodes 时，C=2 与 C=4 完全相同。**
   二者都能放满 24 个 hot experts；C=1 只能放 16 个，batch 4 下
   C4 比 C1 的时间比 `1.115`，95% CI `[1.084, 1.146]`。

4. **1 node 时 C 出现分段收益。**
   C=1/2/4 分别覆盖 8/16/24 个 hot experts。
   batch 4 下 C1/C4 时间比 `1.078`，95% CI `[1.019, 1.137]`；
   C2 与 C4 的差异 CI 跨 1，仍不显著。

5. **batch 越大，C 带来的覆盖收益通常越明显。**
   因为更多 token 能利用更多 hot experts；但这取决于 node 数和容量。

6. **不能给出“C 越大越好”的统一结论。**
   C 只有在容量约束真实生效、且额外 hot coverage 有价值时才影响结果。

## 3. D0 聚类实验控制

聚类网格：

```text
Mixtral 完整 32 层
nodes：1, 2, 4
batch：1, 4, 16
C：1, 2, 4
copy budget：24
每个 cell 固定：GPU layout、hot set、effective copies、trace、C、node、batch
只改变 Brain hot replicas 的放置
```

hot set 不使用“全局热度前 24 个”，因为那会选满少数层的全部专家，使 unique
布局已经自然同层。当前改为每层取前 2 个热点、跨更多层，保证 hyperedge
有可区分的布局。

比较：

- unique-balanced；
- pair-clustered；
- full-hyperedge。

## 4. 聚类结果

### 4.1 Node-cut 确实下降

例如：

| 设置 | Unique node cut | Full-hyperedge node cut |
|---|---:|---:|
| 2 nodes, C=2/4, batch 1 | 11.8 | 5.4 |
| 4 nodes, C=2/4, batch 1 | 13.4 | 6.4 |

说明 full-hyperedge 确实减少了所选 hot experts 的 node fanout。

### 4.2 Completion 没有稳定改善

full-hyperedge / unique 的 completion 加速比：

| Nodes | Batch | C | Full / Unique | 95% CI |
|---:|---:|---:|---:|---:|
| 2 | 1 | 2/4 | 0.998 | [0.994, 1.002] |
| 2 | 4 | 2/4 | 0.997 | [0.991, 1.003] |
| 2 | 16 | 2/4 | 0.998 | [0.993, 1.003] |
| 4 | 1 | 2/4 | 0.9995 | [0.9981, 1.0009] |
| 4 | 4/16 | 2/4 | 1.000 | [1.000, 1.000] |
| 1 | 任意 | 任意 | 1.000 | [1.000, 1.000] |

结论：

> D0 下 full-hyperedge 虽能降低抽象 node cut，但在当前 workload、拓扑和
> 流量模型中没有转化为稳定 completion 收益。

### 4.3 pair clustering 更差

pair / unique 结果在多 node、C≥2 情况下明显小于 1：

- 2 nodes、batch 16、C2/4：约 `0.963`；
- 4 nodes、batch 16、C2/4：约 `0.962`。

pair 虽然有时降低 node/chip cut，却带来了更差的 completion。这再次说明：

```text
cut 更小 ≠ 实际系统更快
```

## 5. 为什么 D0 聚类意义还没有体现

当前证据指向以下原因：

1. D0 下 GPU 通过 bridge 直接到专家 node，home hop 不再是额外成本；
2. 热点专家虽然分散到多个 node，但总 bridge bytes 基本不变；
3. 网络延迟/带宽不是当前总 completion 的唯一关键路径；
4. unique placement 已经按需求均衡分布，额外聚类可能改变 chip/queue 分布；
5. full-hyperedge 目标只优化 hyperedge fanout，没有直接优化 D0 的
   bridge completion queue、GPU combine 或 load imbalance；
6. synthetic trace 的共激活强度仍不足以让 fanout 减少成为主导收益。

因此当前不能声称“D0 聚类有效”，也不能说聚类永远无效。准确结论是：

> 在当前 D0 synthetic workload 中，聚类降低了 node fanout，但没有形成
> end-to-end completion 收益；pair clustering 甚至明显变差。

## 6. 和 C 的联合解释

C 与聚类不是同一个变量：

- C 决定每 chip 能容纳多少专家，影响可覆盖 hot set；
- 聚类决定这些专家放在哪些 node/chip；
- 当前实验里 C 的改变增加了 coverage，因此 speedup 上升；
- 聚类改变了 fanout，但没有稳定提高 speedup。

不能用“增加 C 后性能更好”来证明聚类有效，也不能用“聚类降低 cut”来证明
C 有用。

## 7. 现在能说什么

可以支持：

- D0 下 C 的收益来自有效 hot coverage 和容量，而不是单个串行步骤；
- node 数为 4 时 C 在 24-copy budget 下完全不活跃；
- 1/2 nodes 下增大 C 可扩大 hot set，但 batch、容量和 node 数共同决定收益；
- full-hyperedge 能降低 D0 node cut；
- 当前 cut 降低没有稳定转化为 completion 改善；
- pair clustering 在当前 D0 场景中可能明显有害。

不能支持：

- C 越大 always better；
- D0 聚类在真实 trace 中有收益；
- D0 聚类在任何 C/node/batch 下都有效；
- full-hyperedge 的 cut 改善可以作为最终性能结论；
- 高 batch 的 GPU 并行收益可以自动叠加到聚类收益上。

## 8. Artifacts

- D0 C=1：
  [`moe_2026_d0_matrix_c1_v2`](../experiments/moe_2026_d0_matrix_c1_v2/)
- D0 C=2：
  [`moe_2026_d0_matrix_c2_v2`](../experiments/moe_2026_d0_matrix_c2_v2/)
- D0 C=4：
  [`moe_2026_d0_matrix_c4_v2`](../experiments/moe_2026_d0_matrix_c4_v2/)
- D0 聚类网格：
  [`moe_2026_d0_cluster_v2`](../experiments/moe_2026_d0_cluster_v2/)
- D0/D1 基础结果：
  [`MOE_D0_D1_执行结果_v1.md`](MOE_D0_D1_执行结果_v1.md)
