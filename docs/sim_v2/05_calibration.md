# 校准

## 1. 原则

校准数据和模拟实现必须解耦。设备 profile 引用 benchmark artifact，不把拟合常数散落在代码中。

只有具备原始测量或正式规格证据的 profile 才能使用 `measured`。建议命名：

```text
lynxi_ka200_public_estimate
hp300_hypothetical
<real_model>_measured_<date_or_revision>
```

## 2. 最小 benchmark 集

### 2.1 专家 kernel

```text
expert_latency[
  expert_shape,
  dtype,
  M,
  concurrent_experts,
  residency
]
```

`M` 至少覆盖：

```text
1, 2, 4, 8, 16, 32, 64
```

记录 mean、p50、p95、p99、重复次数、时钟/功耗状态。

### 2.2 网络

至少测量：

```text
p2p[src_class,dst_class,bytes,concurrency]
multicast[fanout,bytes,concurrency]
reduce[fanin,bytes,concurrency]
bridge[bytes,concurrency]
```

需要覆盖小消息延迟区和大消息带宽区，并测量并发拥塞，而非只记录峰值带宽。

### 2.3 功耗

至少区分：

- node/chip idle；
- compute active；
- SRAM active；
- scale-up active；
- scale-out active；
- bridge active。

## 3. 数据格式

建议每份 benchmark artifact 包含：

```yaml
schema_version: 1
machine:
  model: ...
  revision: ...
  firmware: ...
  runtime: ...
measurement:
  timestamp: ...
  repetitions: ...
  environment: ...
records: [...]
```

原始样本和聚合结果都应保存。

## 4. 插值与外推

- 插值方法 MUST 记录；
- 超出测量范围必须产生 warning；
- 外推结果自动降级为 `derived`；
- 不得用单个 peak FLOPS 替代所有 `M`；
- 网络并发超出测量范围时必须报告不确定区间。

## 5. 验收目标

建议初始门槛：

```text
单算子/单链路:
  median relative error <= 10%
  p95 relative error <= 20%

小规模端到端:
  TPOT/throughput relative error <= 15%
  bottleneck ranking must match
```

若测量噪声本身超过门槛，可在记录依据后调整。未通过验收时只报告区间和趋势。

## 6. 当前 hypothetical profile

`lynxi_hp300_proj` 的 SRAM、算力、NoC、片间互连和功耗不得被描述为真机测量。
v2 可将其导入为 `hp300_hypothetical` 做敏感性扫描，但所有报告必须显示 decisive
projected parameters。

