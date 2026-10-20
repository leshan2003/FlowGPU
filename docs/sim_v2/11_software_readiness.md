# Simulator v2 软件侧收口与验收

## 结论

在没有 KA200 真机和真实 router trace 的当前条件下，v2 的软件侧实验闭环已经完成。
它可以给出带完整 provenance 的 synthetic/projected 机制结论、反例和敏感性结果；
不能给出 KA200 实测性能或真机加速结论。

## 已闭环

- 固定 chips/node、扫描 node count，并分离 distributed/concentrated ingress；
- `R<1` tiering、`R≈1` 唯一副本边界、`R>1` 冷覆盖/热复制对照；
- 六种普通 placement、两种 tiered policy、三种 dispatcher；
- placement/evaluation 严格隔离，uniform/weighted/Zipf/node-affine 与 JSONL trace；
- 显式 GPU bridge、node 内/间链路、max-min fluid 争用、multicast/reduce 能力；
- 逐副本 grouped `t_expert(M)`、芯片串行队列和层级 event DAG；
- fixed-window、continuous decode、Poisson arrival、admission、SLO capacity；
- projected prefill block-window、网络/系统能耗、paired-seed Student-t 95% CI；
- fast/detailed 方向审计、小规模 oracle gap 和结果守恒检查；
- 原子结果 bundle、resolved config、warning 和所有输入 artifact SHA-256；
- 四个严格 CLI：fixed-window、serving、uncertainty、GPU/brain tiered。

## 当前验收命令

```text
python -m pytest tests/test_sim_v2_foundation.py -q
python -m pytest tests -q
python -m compileall -q flowgpu

python -m flowgpu v2-node-sweep configs/sim_v2/node_sweep_smoke.yaml -o NEW_DIR
python -m flowgpu v2-node-serving configs/sim_v2/node_sweep_smoke.yaml -o NEW_DIR
python -m flowgpu v2-node-uncertainty configs/sim_v2/node_sweep_smoke.yaml -o NEW_DIR
python -m flowgpu v2-node-tiered configs/sim_v2/node_sweep_smoke.yaml -o NEW_DIR
```

2026-10-19 的收口回归为：v2 foundation 64 项通过、全仓 107 项通过、四条 smoke
命令全部通过。输出目录必须是尚不存在的新目录。

## 唯一剩余的外部证据

1. KA200 kernel、链路、并发、reduce/multicast、容量和功耗原始 artifact；
2. 真实 router trace 的独立 placement/evaluation 窗口；
3. Tier 3 真机 microbenchmark；
4. Tier 4 真机端到端验证。

没有校准重复样本时，不建立虚构的 kernel/link 概率分布。当前 uncertainty CLI 只传播
placement/evaluation 随机种子，硬件参数不确定性应通过明确的 projected 参数扫描报告，
取得真机重复样本后才可增加校准误差 Monte Carlo。

## 明确不属于“真机回来前必须补完”

- packet/cycle-level NoC；
- 动态权重迁移和在线重放置；
- 单专家跨芯片 TP；
- 完整 prefill/decode 联合抢占调度；
- speculative decoding/MTP、多租户和故障恢复。

这些是后续模型扩展，不是当前专家聚类/复制命题的软件验收阻塞项。
