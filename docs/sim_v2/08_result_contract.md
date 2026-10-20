# 结果契约

## 1. 必需产物

每次实验目录 MUST 包含：

```text
config.resolved.yaml
manifest.json
summary.json
placement.json
routing_summary.json
chip_metrics.parquet|csv
link_metrics.parquet|csv
critical_paths.json
warnings.json
```

`manifest.json` 至少记录：

- schema version；
- simulator version/Git revision；
- command；
- seed；
- timestamp；
- host/runtime；
- 输入 artifact hashes；
- decisive provenance。

## 2. 摘要指标

### 性能

- TPOT mean/p50/p90/p99；
- throughput；
- goodput；
- SLO capacity；
- admission fraction 与 rejected requests；
- offered-load SLO goodput（拒绝请求按未达标计）；
- `tokens/s/node`。

### 放置与负载

- unique experts；
- replica count/factor；
- per-node/per-chip SRAM；
- assignments per replica；
- busiest-chip queue/compute；
- grouped-GEMM `M` 分布；
- locality hit rate。

### 网络

- fanout chips/nodes；
- bridge/scale-up/scale-out bytes；
- per-link utilization；
- flow queueing；
- dedup/reduction saving；
- busiest link；
- network critical-path share。

### 能耗和成本

- J/token；
- device/network/host energy；
- idle share；
- node count；
- capex（若有可信输入）。

功耗账采用 `idle + incremental compute` 语义：所有已配置芯片在完整观察窗口内均计
idle，compute 只对该芯片忙区间的并集计增量功率，network 由实际 flow 的逐链路
`J/byte` 累加。不得只统计 active chip 而让新增空闲 node 免费，也不得在缺少带
provenance 的功率 profile 时输出系统 `J/token`。

Serving 中必须分别命名 `network_energy_per_token_j` 与
`system_energy_per_token_j`。未提供完整 chip power profile 时后者为缺失值，禁止把前者
重命名为系统 J/token。

## 3. 分解要求

任何速度提升必须能分解为：

```text
spill removal
network locality
network dedup/reduction
compute load balance
batch efficiency change
queueing change
additional hardware
```

不得只输出一个总加速比。

## 4. 不确定性

报告必须包含：

- 多 seed 置信区间；
- decisive parameter sensitivity；
- measured/estimated/projected 参数占比；
- 校准误差传播；
- fast/detailed 差异。

多 seed 连续指标默认使用双侧 95% Student-t 区间，不得在小样本时固定使用 1.96。
概率区间截断到 `[0,1]`，非负物理量的区间下界截断到 0；单 seed 必须明确标记为
`single-sample-no-uncertainty`，不能称为有统计置信度。

如果收益区间跨过 1.0×，结论标为 inconclusive。

## 5. 警告

以下情况 MUST 产生高可见度 warning：

- 容量超限；
- 拓扑不可达；
- 零成本 bridge；
- kernel/network 外推；
- projected 参数决定结论；
- placement 使用 evaluation future；
- fast/detailed 结论方向不一致；
- 守恒检查失败。

守恒失败必须使运行失败，不能只 warning。

## 6. 策略对照 bundle

同一 node-count case 的多策略对照必须原子保存：

- `case.json`：固定 chips/node、总芯片数、ingress、R 和硬件指纹；
- `traces.json`：strong/weak scaling、placement/evaluation token 数及各自 provenance；
- `strategies.csv`：总延迟、相对 unique baseline 的 speedup、副本数、locality、网络能耗；
- `layers.csv`：逐层 assignment、流量、争用和能耗；
- `placements.json`：各策略完整副本与容量表；
- manifest、resolved config、warnings 和输入 artifact SHA-256。

已有目录不得覆盖；不同 ingress/scaling/node count 必须使用不同 case 目录。
