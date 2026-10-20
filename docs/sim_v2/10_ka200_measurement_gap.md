# KA200 校准缺口与采集契约

## 1. 仓库审计结论

当前仓库中不存在可供 v2 使用的 KA200 原始 benchmark artifact。历史 profile
`lynxi_ka200` 的说明明确指出 SRAM 容量/带宽和 NoC 数值是估计；现有输出文件是这些
解析假设的派生结果，不是真机测量。

下列信息可作为公开资料描述，但仍需保存正式来源后才能升级相应 provenance：

- 30 个类脑计算核心；
- ANN/SNN 异构与存算一体；
- 250k neurons / 25M synapses；
- any-to-any core communication 与 multicast 能力描述；
- 约 12–15 W 功耗描述。

下列决定 MoE 结论的参数当前不能视为 measured：

| 参数 | 仓库当前值 | 当前性质 |
|---|---:|---|
| SRAM 容量 | 48 MB | estimate |
| SRAM 聚合带宽 | 2.4 TB/s | estimate |
| NoC bisection | 350 GB/s | estimate |
| inter-chip | 50 GB/s | estimate |
| inter-board | 25 GB/s | estimate |
| expert `t_expert(M)` | 无 | missing |
| multicast/fanout 曲线 | 无 | missing |
| local reduce/fanin 曲线 | 无 | missing |
| GPU↔brain bridge 曲线 | 无 | missing |
| 可用 SRAM reserve/buffer | 无 | missing |

`lynxi_hp300_proj` 的 768 MB SRAM、51.2 TB/s、200/100 GB/s 链路和 150 W 是
人为构造的 projected scenario，不能通过重命名变成 KA200 或后续芯片的真机 profile。

## 2. 必需采集

使用
[`ka200_measured_template.yaml`](../../configs/calibration/ka200_measured_template.yaml)
记录同一机器 revision、firmware 和 runtime 下的：

1. 每个实际 expert shape/dtype 的 `M=1,2,4,8,16,32,64`；
2. 同芯片多专家并发；
3. chip↔chip、node↔node、GPU↔ingress 的消息大小与并发扫描；
4. multicast fanout 与 reduce fanin；
5. idle、compute、SRAM、scale-up、scale-out 和 bridge 功耗；
6. 可用 SRAM、runtime reserve 和通信 buffer 占用。

每项至少保留 repetitions、中心值、p95/p99 和原始样本路径。模板中的 `null` 不得用
现有 estimate profile 填充。

## 3. 进入结果的门槛

- artifact 通过 v2 schema 与 SHA-256 校验；
- kernel 推荐 M 点缺失会产生 warning；
- 网络拟合保存 message-size 范围和 median relative error；
- 拟合参数标为 `derived`，原始样本保留 `measured`；
- 超出测量范围自动产生 extrapolation warning；
- 未达到单算子 median 10%、p95 20% 和端到端 15% 目标前，只能报告区间/趋势。

在这些条件满足前，KA200 相关实验必须标为 public-estimate scenario，不能写成真机
MoE 加速结论。

