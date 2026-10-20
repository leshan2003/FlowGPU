# MoE 仿真数据与校准接口

> 日期：2026-10-20  
> 用途：登记当前仓库可用于仿真平台构建的数据，并说明后续实测数据如何替换。

## 1. 结论

当前仓库中的资料足以搭建和运行第一阶段 MoE 专家分层、复制、聚类和刷新仿真平台。
真实 router trace 与 KA200 原始测量不是仿真平台启动的前置条件，而是后续提高外部
有效度的 calibration artifact。

当前实验必须标注为：

```text
synthetic workload + projected/estimated hardware
```

取得实测文件后才能升级为：

```text
trace-driven / measured-hardware evidence
```

## 2. 当前输入清单

### 2.1 GPU

来源：`flowgpu/hardware/registry.py`，public vendor datasheet。

| Profile | 主要参数 | 用途 |
|---|---|---|
| `h100_sxm` | 80 GB HBM、3.35 TB/s、FP8 1979 TFLOPS、700 W | 主 GPU/HBM 基线 |
| `iluvatar_bi_v100` | 32 GB、1.2 TB/s | 非 H100 敏感性 |
| `b200_sxm` | 192 GB、8 TB/s | 高带宽敏感性 |

这些容量以 decimal bytes 表示，已经统一为 `80,000,000,000 bytes`。

### 2.2 类脑设备

| Profile | 证据状态 | 当前使用方式 |
|---|---|---|
| `lynxi_ka200` | 公开算力 vendor；SRAM/NoC/DRAM 为 estimate | KA200 estimate scenario |
| `lynxi_hp300_proj` | 容量、带宽、功耗均为 simulator projection | 主 projected simulation profile |

已登记：

- KA200：30 cores、48 TOPS INT8、24 TFLOPS FP16、约 12 W；
- KA200 estimate：48 MB SRAM、2.4 TB/s、68 GB/s DRAM、50/25 GB/s chip/board link；
- HP300 projected：768 MB SRAM、51.2 TB/s、200 GB/s DRAM、200/100 GB/s link、150 W。

任何估计/profile 值都保留 provenance，不得改名为实测值。

### 2.3 模型

来源：`flowgpu/workload/models.py` 的公开模型配置。

| 模型 | 结构 | 用途 |
|---|---|---|
| Mixtral-8x7B | 32 layers、8 experts、top-2 | small/full-model screen |
| Qwen3-235B-A22B | 94 layers、128 experts、top-8 | medium/large screen |
| DeepSeek-V3 | 61 layers、58 MoE layers、256 experts、top-8 | main model |
| gpt-oss-120B | 36 layers、128 experts、top-4、mxfp4 | model-family generality |

Stage 0 runner 已根据 config 自动计算 expert bytes、top-k、共享专家、first MoE layer。
DeepSeek-V3 routed expert 为 `44,040,192 bytes`，总量 `609 GiB`。

### 2.4 链路

来源：

- `flowgpu/hardware/interconnect.py` 的 `FABRICS`；
- device registry 中的 brain inter-chip/inter-board 参数；
- GPU↔brain bridge 使用显式 fabric profile。

当前默认：

```text
bridge      = pcie5_x16
brain chip  = lynxi_hp300_proj
GPU         = h100_sxm
```

## 3. 路由与刷新输入

当前提供：

- task-conditioned community generator；
- ordered request/task phase trace；
- Zipf marginal skew；
- node affinity；
- full top-k hyperedge frequency；
- request_id、stage、timestamp、task_family、window；
- drift hysteresis detector；
- paired block bootstrap。

尚未有真实 request-ordered router file。真实文件以后可通过
`load_routing_trace_jsonl()` 接入，字段契约见
`docs/sim_v2/04_routing_models.md`。

## 4. 论文数据

本地全文目录：`data/stage1/moe_review/`。

已获取并记录 SHA-256 的 9 篇工作：

- Expert Activation Patterns；
- Cobalt；
- Director；
- HCRMap；
- ReXpert；
- Sieve；
- Expert Streaming；
- Not All Models Suit Expert Offloading；
- HybriMoE。

此外有本地 SHIP 论文和项目历史硬件资料。论文数据用于设置 workload、机制和敏感性，
不得移植为项目实测性能。

## 5. 后续校准替换点

实测数据到位后只需替换以下接口，不改实验策略：

1. `HardwareProfile`：SRAM/HBM/DRAM 容量、chip/board 带宽、功耗；
2. `ExpertKernelCurve`：真实 `t_expert(M)`；
3. `LinkProfile`：bridge、chip、board、multicast/reduce 拟合；
4. `RoutingTrace`：真实 request/task trace；
5. `ChipPowerProfile`：idle/compute/link 功耗；
6. acquisition/TCO 输入。

## 6. 当前可用运行入口

```powershell
python scripts/run_moe_stage0.py `
  --model deepseek_v3 `
  --output NEW_STAGE0_DIR

python scripts/run_moe_screen.py `
  --model mixtral_8x7b `
  --output NEW_SCREEN_DIR
```

设备 profile 可选：

```text
--gpu-profile h100_sxm
--brain-profile lynxi_hp300_proj
--bridge-fabric pcie5_x16
```

也可使用 `--brain-profile lynxi_ka200` 运行 estimate 上限/可行性检查。

## 7. 目前仍不代表什么

当前结果不代表：

- 真实 KA200 kernel 性能；
- 真实 router popularity；
- 实测能效或 TCO；
- 最终 node 数和刷新周期。

这些结论需要在后续 calibration 阶段由实测数据替换并重新运行。
