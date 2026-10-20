# MoE 异构计算研究

> 当前证据等级：`synthetic workload + projected/estimated hardware`
>
> 不是 KA200 实测，也不是真实业务 trace 或 TCO。

## 30 秒结论

1. 最稳定的方案是 **HBM 保存全部专家 + 类脑 SRAM 选择性保存热点副本**。
2. Coverage-first 是当前最稳定的规划策略；固定 replica-first 收益很弱。
3. M3 的 GPU/Brain 并行收益只在 **高 batch** 条件得到支持，低/中 batch 不会自动获益。
4. D0 相比 D1 明显减少 bridge 流量，但 completion 收益依赖 batch 和网络瓶颈。
5. C 只有在限制 hot coverage 时才产生影响，不存在“C 越大越好”。
6. 当前 D0 聚类降低了 node cut，但没有稳定转化为 completion 收益。
7. M1 全类脑 DRAM/SRAM 路径在当前 projected profile 下显著慢于 GPU/HBM。

## 文档导航

| 文档 | 内容 |
|---|---|
| [01 研究模型](01_研究模型与设计.md) | M1/M2/M3、D0/D1、SRAM/DRAM/HBM、刷新与聚类定义 |
| [02 实验方法与可信度](02_实验方法与可信度.md) | 控制变量、输入数据、统计、审计与已修复问题 |
| [03 实验结果](03_实验结果.md) | 20-seed、node/batch、C、聚类、D0/D1、M1/per-GPU |
| [04 结论与缺口](04_结论与缺口.md) | 可以支持/不能支持的结论及下一阶段优先级 |
| [05 复现与结果索引](05_复现与结果索引.md) | canonical 目录、运行命令、manifest 与并行执行 |
| [给负责人的项目汇报](给学长的项目汇报_2026-10-20.md) | 上传内容、关键结果和可直接发送的微信短版 |

## 当前 canonical 版本

| 实验族 | 版本 |
|---|---|
| Stage 0 | `v13` |
| Screen / matrix / C / boundary | `v9` |
| C=1 clustering stress | `v5` |
| D0 base / high-batch | `v7` |
| D0 C panel | `v3` |
| D0 clustering | `v3` |
| D0 per-GPU | `v1` |
| M1 full B-stage | `v1` |

完整目录见 [05 复现与结果索引](05_复现与结果索引.md) 和
[experiments/README](../../experiments/README.md)。

## 文件结构

```text
docs/MOE/
├── README.md
├── 01_研究模型与设计.md
├── 02_实验方法与可信度.md
├── 03_实验结果.md
├── 04_结论与缺口.md
└── 05_复现与结果索引.md
```

历史详细报告保存在
[`docs/archive/MOE_v1/`](../archive/MOE_v1/)。
归档内容用于追溯，不应替代本目录的 canonical 结论。
