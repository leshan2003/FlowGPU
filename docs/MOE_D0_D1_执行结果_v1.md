# MoE D0/D1 与 M2/M3 执行结果 v1

> 日期：2026-10-20
>
> 证据类型：synthetic routing + projected hardware
>
> 状态：D0-shared / D1-shared 已实现并执行；D0-per-GPU 暂停，尚无有效结论

## 1. 模式定义

| 名称 | 数据路径 |
|---|---|
| D0 | GPU → 专家 node → GPU |
| D1 | GPU → token home node → 专家 node → home node → GPU |
| M2 | hot 专家只在 Brain-SRAM；cold 专家只在 GPU/HBM；无跨层 fallback |
| M3 | 保留 M2 的完全相同 Brain hot 副本；另外给 hot 专家增加 GPU 非主副本 |

M3 是专门从 M2 派生出来的：

```text
M2 的 cold GPU 布局不变
M2 的 hot Brain 布局不变
M3 只增加 hot expert 的 GPU 副本
```

因此 M2/M3 的差异只来自“是否允许 hot expert 使用 GPU 并行”，而不是不同
placement。

## 2. 先修复的两个实现问题

### 2.1 实际 flow 曾绕过 D0

第一版 D0 只修改了 Dispatcher 的通信预测，但 `MoELayerEngine` 实际生成
dispatch/combine flow 时仍强制经过 home node。该版本结果无效，已修正：

- D0 的实际 flow 现在是 GPU ↔ expert node；
- D1 的实际 flow 现在是 GPU ↔ home ↔ expert；
- combine 使用对应反向路径；
- 新测试验证 D0 不使用 home bridge。

### 2.2 M2/M3 布局曾不一致

第一版 M3 使用独立 coverage-first 布局，导致它与 M2 的 hot 集或 cold GPU
位置不同。该版本会污染 M3-vs-M2 比较，已修正。

当前 M3 直接继承 M2 的 Brain hot 主副本和 cold GPU 主副本，只增加 hot expert
的 GPU 非主副本。

## 3. 正式实验 A：Mixtral 全模型矩阵

配置：

```text
模型：Mixtral-8x7B 完整 32 层
seeds：101–105，共 5 个 paired seeds
nodes：1, 2, 4
batch：1, 4, 16
SRAM copies：8
C：2
dispatch：D0-shared, D1-shared
tiers：M2, M3
scheduler：earliest_finish, least_loaded, nearest, brain-first, split
```

### 3.1 D0 vs D1

M2 跨全部 45 个 paired matrix points：

| 对比 | 均值 | 95% CI |
|---|---:|---:|
| D1 completion / D0 completion | 1.000031 | [0.999986, 1.000077] |

结论：

> 在 batch 1/4/16 的全模型矩阵中，D0/D1 的 completion 几乎完全相同。

但网络流量不同：

| Variant | 平均 bridge bytes |
|---|---:|
| D1 | 3.90 MiB |
| D0 | 2.14 MiB |

D0 减少约 `45.2%` 的 bridge bytes。

因此当前结果是：

- D0 确实去掉了 home 绕行；
- 在低/中 batch 下，这部分并不是关键路径；
- bridge 流量减少没有转化为稳定 completion 收益。

### 3.2 M3 vs M2：低/中 batch

跨全部 45 个 paired points：

| Variant / scheduler | M2 time / M3 time | 95% CI | 解释 |
|---|---:|---:|---|
| D0 brain-first | 1.000 | [1.000, 1.000] | GPU 热副本完全不用，M3 等价 M2 |
| D0 earliest_finish | 0.9968 | [0.9948, 0.9988] | M3 略慢 |
| D0 least_loaded | 0.9748 | [0.9689, 0.9807] | M3 更慢 |
| D0 nearest | 0.9581 | [0.9506, 0.9657] | M3 明显更慢 |
| D0 split | 0.9760 | [0.9704, 0.9817] | M3 更慢 |

D1 的方向一致。

结论：

> 在低/中 batch 下，当前所有实际使用 GPU hot 副本的启发式都没有产生正收益。
> `brain-first` 可以保证 M3 不差于 M2，因为它等价于不使用并行副本。

`nearest` 最差，因为在 D0 下 GPU 本地距离为 0，它几乎把所有 hot assignment
都送到 GPU，导致 GPU 排队增加。

## 4. 正式实验 B：高 batch 专项

由于 batch 1/4/16 未看到并行收益，增加：

```text
模型：Mixtral 前 4 个 MoE 层
seeds：101–110，共 10 个 paired seeds
nodes：2
batch：64
SRAM copies：8
C：2
```

这是 fast screen，不是完整 32 层结论。

### 4.1 D0 vs D1

M2：

| Variant | 相对 GPU-only completion 加速 | GPU busy fraction |
|---|---:|---:|
| D1-shared | 1.574 | 15.5% |
| D0-shared | 1.634 | 16.1% |

paired `D1 time / D0 time`：

```text
1.03859, 95% CI [1.02441, 1.05277]
```

结论：

> 在 batch 64、2-node 的四层 screen 中，D0 比 D1 快约 3.9%，
> 且 95% CI 下界大于 1。

这与低/中 batch 的结果不同，说明 home 路径是否处于关键路径依赖 batch 和
网络负载。

### 4.2 M3 vs M2：高 batch 并行收益

D0-shared：

| Scheduler | M2 time / M3 time | 95% CI | GPU assignment fraction | GPU busy fraction |
|---|---:|---:|---:|---:|
| brain-first | 1.000 | [1.000, 1.000] | 41.9% | 16.1% |
| earliest_finish | **1.082** | **[1.0085, 1.1556]** | 53.1% | 22.5% |
| least_loaded | 0.942 | [0.828, 1.056] | 71.0% | 23.2% |
| nearest | 0.553 | [0.517, 0.588] | 100% | 17.9% |
| split | 0.945 | [0.830, 1.060] | 70.8% | 23.3% |

D1-shared 的 earliest_finish：

```text
M2 time / M3 time = 1.1158
95% CI = [1.0408, 1.1908]
```

结论：

> 在高 batch 四层 screen 中，`earliest_finish` 确实利用了 GPU 并行 hot
> assignment，并将 M3 completion 比 M2 改善约 8.2%（D0）或 11.6%（D1）。

但这不是通用结论：

- 只测了 Mixtral 前 4 个 MoE 层；
- 只在 2 nodes、batch 64 下成立；
- `nearest`、`least_loaded`、`split` 仍会退化；
- 当前 scheduler 仍不是全局 load-aware 最优解；
- 没有完整模型 serving P99/SLO 结论。

## 5. 对“M3 应优于 M2”的判断

理论上，如果：

- HBM 容量充足；
- M2/M3 有相同 hot 位置；
- GPU 未饱和；
- scheduler 正确分配；

则 M3 拥有 M2 的全部动作，再加 GPU hot fallback/并行，应该不差于 M2。

本轮实际结果显示：

- 小/中 batch 下，GPU 额外动作没有被有效利用；
- 贪婪 scheduler 反而增加 GPU 排队；
- 高 batch 下，`earliest_finish` 找到了可利用的并行区间；
- M3 的理论支配关系需要 **全局 load-aware scheduler**，不能由简单
  `nearest` 或平均分摊实现。

所以当前准确结论是：

```text
M3 在高 batch 条件下有已验证的条件性收益；
M3 不是所有 node/batch 下都自动优于 M2；
当前 scheduler 仍可能导致 M3 退化。
```

## 6. D0-per-GPU 为什么没有结果

当前尝试过 16 个 GPU endpoint，但发现模型不成立：

- 每个专家只有一份 GPU 副本；
- 请求却来自 16 个 GPU ingress；
- 远端 GPU 副本没有有效数据路径；
- 因此该配置会把“无效路由”混入结果。

后续 D0-per-GPU 必须采用以下之一：

1. 每个 GPU 保存全部/分区后的专家副本，并绑定 request owner；
2. 为每个专家在请求可能到达的 GPU 上建立明确副本；
3. 使用真实的 GPU rank/request affinity trace。

在完成之前，D0-per-GPU 结论为空，不能引用。

## 7. 结论等级

可以支持：

- D0 实际 flow 与 D1 已在事件模型中分开；
- 低/中 batch 全模型中 D0 主要减少 bridge 字节，completion 基本不变；
- batch 64、2-node、Mixtral 四层 screen 中 D0 比 D1 快约 3.9%；
- batch 64、2-node、Mixtral 四层 screen 中 M3 earliest-finish 比 M2 快约
  8.2%（D0）；
- 简单 `nearest` 或静态分摊不是有效 GPU/Brain 协作策略。

不能支持：

- D0 在所有 batch/node 下优于 D1；
- M3 在低/中 batch 下优于 M2；
- 完整 32 层在 batch 64 下具有同样并行收益；
- 当前 scheduler 已达到全局最优；
- D0-per-GPU 已有结论；
- 真实 KA200/GPU 性能或 TCO 结论。

## 8. Artifacts

- 全模型矩阵：
  [`moe_2026_d0_matrix_v5`](../experiments/moe_2026_d0_matrix_v5/)
- 高 batch 四层 screen：
  [`moe_2026_d0_highbatch_v5`](../experiments/moe_2026_d0_highbatch_v5/)
- 冻结计划：
  [`MOE_聚类与层级调度重验证计划_v1.md`](MOE_聚类与层级调度重验证计划_v1.md)
