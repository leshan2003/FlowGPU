# MoE 文件与结果索引

> 最后更新：2026-10-20
>
> 这是 MoE 项目的唯一阅读入口。先看本页，再按需进入专题报告。

## 1. 先读这三份

1. [实验结论速览](MOE_实验结论速览.md)
   - 一页理解所有主要结论、指标含义和限制。
2. [最终实验执行与审计](MOE_最终实验执行与审计_v1.md)
   - canonical bundle、完整矩阵、测试和哈希审计。
3. [聚类与层级调度重验证计划](MOE_聚类与层级调度重验证计划_v1.md)
   - 冻结的模式定义、D0/D1、M1/M2/M3、聚类目标和后续矩阵。

## 2. 专题结果

| 主题 | 文档 | 主要内容 |
|---|---|---|
| Stage 0 | [MOE_STAGE0_执行审计](MOE_STAGE0_执行审计.md) | 四模型语义门控、G/H/B/refresh/serving |
| Stage 1 screen | [MOE_STAGE1_SCREEN_RESULTS](MOE_STAGE1_SCREEN_RESULTS.md) | Mixtral/大模型 fast screen |
| 总矩阵 | [MOE_仿真完整矩阵结果_v1](MOE_仿真完整矩阵结果_v1.md) | node/batch、C、20-seed |
| D0/D1 | [MOE_D0_D1_执行结果_v1](MOE_D0_D1_执行结果_v1.md) | GPU-direct、home-node、M2/M3、高 batch |
| D0 C/聚类 | [MOE_D0_C与聚类结果_v1](MOE_D0_C与聚类结果_v1.md) | D0 下 C×node×batch、unique/pair/full |
| M1/per-GPU | [MOE_M1与D0_PerGPU结果_v1](MOE_M1与D0_PerGPU结果_v1.md) | 全类脑 B-stage、合法 owner-GPU |
| 数据/校准 | [MOE_仿真数据与校准接口](MOE_仿真数据与校准接口.md) | 当前输入、估计等级、实测替换点 |
| 前人工作 | [MOE_前人工作全文审计_v2](MOE_前人工作全文审计_v2.md) | 论文证据和使用边界 |
| 启动门控 | [MOE_实验启动门控审计_v2](MOE_实验启动门控审计_v2.md) | 启动前审计记录 |
| 冻结设计 | [MOE_高置信度实验设计_v2_冻结版](MOE_高置信度实验设计_v2_冻结版.md) | 早期冻结的统计与实验规范 |

## 3. 三种层级模式

| 模式 | 权重位置 | 热专家计算 | 冷专家计算 |
|---|---|---|---|
| M1 | Brain DRAM + SRAM | Brain SRAM | Brain DRAM → SRAM |
| M2 | hot 只在 SRAM，cold 只在 HBM | Brain | GPU/HBM |
| M3 | HBM 全备份 + SRAM 热副本 | GPU 或 Brain | GPU/HBM |

关键结果：

- M1 全模型在当前 projected profile 下显著慢于 GPU-only；
- M2 只有条件性收益；
- M3 在低/中 batch 没有自动收益；
- M3 在 high-batch 四层 screen 中有统计正收益，但不能直接外推全模型。

## 4. D0/D1 路由

```text
D0：GPU → expert node → GPU
D1：GPU → home node → expert node → home node → GPU
```

- 当前聚类主实验使用 D1；
- D0 实际 event flow 已单独实现；
- D0 reduce bridge bytes，但 completion 是否改善依赖 batch/node；
- D0-per-GPU 已有合法 owner-GPU 实现，但固定 GPU owner 会造成负载不均。

## 5. Canonical 结果版本

```text
Stage 0：v13
Stage 1 / matrix / C / boundary：v9
Cluster stress：v5
D0 base / high-batch：v7
D0 C：v3
D0 clustering：v3
D0 per-GPU：v1
M1 full：v1
```

对应目录见 [experiments/README](../experiments/README.md)。

除这些目录外，带有更早版本号的目录只用于开发审计，不应在论文或汇报中引用。

## 6. 当前审计状态

```text
146 tests passed
46 canonical bundles
source SHA-256 mismatch = 0
```

当前证据等级：

```text
synthetic workload + projected/estimated hardware simulation evidence
```

不是：

- 真实 router trace；
- KA200 实测；
- 真实 TCO；
- 完整生产 serving 性能。

## 7. 目前最关键的尚未完成项

1. 真实 request-ordered router trace；
2. D0/D1 下完整模型 serving P99/SLO；
3. 全局 load-aware scheduler；
4. D0 cluster 与真实共激活/home-affinity 数据；
5. M1 block streaming/prefetch/overlap；
6. KA200 kernel/NoC/DRAM/功耗实测；
7. acquisition cost、系统功耗和真实 TCO。
