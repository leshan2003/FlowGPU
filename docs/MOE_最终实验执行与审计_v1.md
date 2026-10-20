# MoE 最终实验执行与审计 v1

> 日期：2026-10-20
>
> 证据类型：synthetic workload + projected/estimated hardware
>
> 最终 canonical 版本：Stage 0 `v12`，下游 screen/matrix/C/20-seed 为 `v8`
>
> 结论边界：这是可复现的软件仿真证据，不是 KA200 实测，也不是真实 TCO。

## 1. 本轮实际完成了什么

本轮不是继续做设计，而是执行并审计了完整的 synthetic/projected 实验矩阵：

| 实验组 | 内容 | 规模 | 最终目录版本 |
|---|---|---:|---|
| Stage 0 | 四模型语义与机制门控 | 4 bundles | `stage0_final_v12_*` |
| Stage 1 screen | Mixtral 三种 skew、三种大模型四层 fast screen | 6 bundles | `screen_final_v8_*` |
| Node × batch | 1/2/4 nodes × 1/2/4 request batch | 9 points | `matrix_final_v8_*` |
| C isolation | 64 copies/64 chips、32 copies/16 chips | 9 points | `cscan*_final_v8_*` |
| 20-seed boundary | 6 workload/model boundary points | 120 paired seed runs | `boundary20_v8_*` |

合计 34 个 canonical bundle。34 个 bundle 的 `manifest.json` 均覆盖 runner、
`flowgpu/units.py` 和全部 31 个 `flowgpu/sim_v2/*.py` 文件；
逐文件 SHA-256 审计结果为：

```text
BUNDLES=34
SOURCE_HASH_MISMATCH=0
MISSING_MIGRATION_RUNTIME=0
```

测试及编译检查：

```text
python -m pytest tests -q
138 passed

python -m compileall -q flowgpu scripts
PASS
```

重建后，30 个下游实验的 `aggregate.csv` 与重建前对应文件逐字节一致；
Stage 0 四个模型的 `gate_results.csv` 也逐字节一致。此前矩阵中缺失的
`n2/b2` 点已在本轮补齐。

## 2. 当前模型语义

### 2.1 G：HBM fallback + Brain-SRAM overlay

这是主模式：

- GPU/HBM 保存每个专家的一份 primary 副本；
- Brain-SRAM 只保存选定专家的额外副本；
- 同一个专家的不同 token/assignment 可分别落在 GPU 与 Brain；
- dispatcher 对每个 assignment 选择预测最早完成的副本；
- Brain 不可用或未命中时，可由 GPU/HBM 静默 fallback。

因此，G 模式确实允许 hot expert 的并发执行，但其粒度是“不同 assignment
分布到两个副本”，不是把一个 expert kernel 拆成 GPU 与 Brain tensor parallel。

### 2.2 H：按专家独占层级

H 模式中每个专家恰好有一份执行副本：

- hot expert：Brain-SRAM primary；
- 其余专家：GPU/HBM primary；
- 不保留另一层的静默 fallback。

因此 H 可以判断“是否把某专家整体搬离 HBM”，但它不是“所有专家都只在
Brain 上运行”的模式。

### 2.3 B-stage：Brain DRAM + SRAM staging

B-stage 表示：

- 专家权重位于 Brain DRAM；
- hot 子集预先驻留 Brain-SRAM；
- miss 后按 grouped expert 从 DRAM staging 到 SRAM；
- 不计 GPU/HBM fallback。

它是 grouped full-expert staging，不是 block/fine-grained streaming。

### 2.4 连续冷热与副本

规划器不再要求固定 top-20% hot set。每个候选 copy 使用连续需求、node
affinity、共激活和容量约束评估。`R_unique < 1` 时仍允许极热专家获得第二份
或更多副本；不需要先把全部专家覆盖一遍。

本实验实际比较了：

- coverage-first：先最大化 unique coverage；
- replica-first：固定份额优先复制热点；
- ranked-0.25 / ranked-adaptive：渐进复制份额；
- joint-marginal：在“新增 unique copy”和“新增 extra copy”之间比较边际收益；
- marginal-event：使用 event-model callback 的连续边际动作。

### 2.5 共激活聚类

原始 trace 保存每个 token/layer 的完整 top-k hyperedge。评估使用完整
hyperedge cut；placement 同时提供 pair-clique 和 full-hyperedge 路径。

聚类实验使用完全相同的 copy multiset 和 C=1 资源约束，比较：

- unique-balanced；
- pair-clustered；
- full-hyperedge。

因此 C=1 聚类不会因为增加额外副本而获得虚假收益。

### 2.6 刷新

刷新被拆成四个动作：

1. 从历史窗口更新统计；
2. 重算 candidate placement；
3. source-aware migration；
4. 新版本完整可用后激活，并保护在途请求。

迁移源按实际位置选择：Brain 内部 copy 优先从 Brain 获取，只有缺失时才从
GPU/HBM 经 bridge 搬运。迁移 flow 与 serving flow 进入同一个
`FluidNetwork`，共享链路带宽。

## 3. 关键变量与实验控制

当前可控变量包括：

- 模型：Mixtral、DeepSeek-V3、Qwen3-235B-A22B、gpt-oss-120B；
- workload skew：Zipf `0.5/1.3/2.0`；
- task phase 与 request order；
- Brain node 数；
- request batch；
- total Brain copy budget；
- 每 chip expert 上限 `C`；
- primary coverage 与 extra-copy policy；
- placement/evaluation/refresh 时间窗口；
- seed；
- GPU、Brain 和 bridge profile。

公平性控制：

- placement 与 evaluation 使用不同窗口和 seed offset；
- 所有策略重放同一 demand trace；
- 比较 policy 时使用同一专家字节预算和同一芯片集合；
- 比较 C 时使用同一 copy budget、node/chip 数和 workload；
- 每个 paired seed 同时应用于所有策略；
- 不能用某个 node 数的最优 placement 外推到另一个 node 数。

## 4. 输入数据和证据等级

| 输入 | 当前值/来源 | 等级 |
|---|---|---|
| H100 | 80 GB HBM、3350 GB/s、700 W | vendor |
| PCIe 5.0 x16 bridge | 64 GB/s、1.2 us、0.80 efficiency | projected catalogue |
| HP300 projection | 768 MB SRAM、51.2 TB/s；200 GB/s DRAM；200/100 GB/s link；150 W | simulator projection |
| KA200 estimate | 48 MB SRAM、2.4 TB/s；68 GB/s DRAM；公开 24 TFLOPS FP16/48 TOPS INT8/约 12 W | public estimate |
| 模型结构 | 本地公开 config | public config/derived |
| expert kernel | 硬件相对 synthetic curve | projected |
| routing | task-conditioned synthetic top-k trace | synthetic |

模型几何：

| 模型 | MoE 层 | experts | top-k | expert 大小 | routed weights |
|---|---:|---:|---:|---:|---:|
| Mixtral-8x7B | 32 | 8 | 2 | 168 MiB | 42 GiB |
| DeepSeek-V3 | 58 | 256 | 8 | 42 MiB | 609 GiB |
| Qwen3-235B-A22B | 94 | 128 | 8 | 18 MiB | 211.5 GiB |
| gpt-oss-120B | 36 | 128 | 4 | 12.61 MiB | 56.73 GiB |

当前 GPU fixture 为 16 个 H100，因此 HBM fallback 可以持有四个模型的全部
routed experts。实验比较的是“全量 HBM + selective Brain overlay”“按 expert
分层”和“Brain DRAM staging”，不是 HBM 容量不足导致的被迫 offload。

## 5. 20-seed 结果

Mixtral 使用完整 32 层；Qwen、DeepSeek、gpt-oss 使用前 4 个 MoE 层。
区间为 paired seed 95% CI。

| Model/workload | Coverage | Joint-marginal | Replica-first |
|---|---:|---:|---:|
| Mixtral Zipf 0.5 | 1.053 [1.041, 1.065] | 1.053 [1.041, 1.065] | 1.003 [1.000, 1.006] |
| Mixtral Zipf 1.3 | 1.085 [1.074, 1.097] | 1.085 [1.074, 1.097] | 1.007 [1.003, 1.011] |
| Mixtral Zipf 2.0 | 1.081 [1.067, 1.094] | 1.081 [1.067, 1.094] | 1.011 [1.006, 1.016] |
| Qwen3, 4 layers | 1.112 [1.084, 1.141] | 1.120 [1.092, 1.149] | 1.012 [1.001, 1.022] |
| DeepSeek-V3, 4 layers | 1.081 [1.061, 1.101] | 1.082 [1.056, 1.108] | 1.018 [1.004, 1.032] |
| gpt-oss, 4 layers | 1.176 [1.131, 1.220] | 1.156 [1.123, 1.189] | 1.030 [1.010, 1.050] |

连续 serving：

| Model/workload | Overlay serving | H-exclusive serving |
|---|---:|---:|
| Mixtral Zipf 0.5 | 1.026 [1.017, 1.035] | 1.025 [1.008, 1.042] |
| Mixtral Zipf 1.3 | 1.033 [1.025, 1.040] | 1.015 [0.999, 1.031] |
| Mixtral Zipf 2.0 | 1.036 [1.025, 1.047] | 1.022 [0.997, 1.048] |
| Qwen3, 4 layers | 1.085 [1.068, 1.101] | 1.026 [0.995, 1.057] |
| DeepSeek-V3, 4 layers | 1.074 [1.058, 1.091] | 1.007 [0.980, 1.034] |
| gpt-oss, 4 layers | 1.051 [1.036, 1.067] | 1.041 [1.019, 1.064] |

解释：

- Coverage-first 是当前最稳定的主 baseline；
- Replica-first 的收益很小，不能据此否定“副本”，但固定优先复制的策略不优；
- Qwen 四层的 joint-marginal 均值高于 coverage，但两者 CI 重叠，当前不能声称
  已建立显著优势；
- H-exclusive 明显依赖模型和 workload，DeepSeek/Qwen/Mixtral 的部分 CI 跨 1；
- 四层 fast screen 不能外推为完整模型结论。

## 6. Node × batch 矩阵

Mixtral 完整 32 层、C=2、16 copies、5 seeds。batch 在本 runner 中表示同时
admitted 的 request 数。

| Nodes | Batch | Coverage static | Overlay serving |
|---:|---:|---:|---:|
| 1 | 1 | 1.061 [1.013, 1.108] | 1.019 [1.013, 1.026] |
| 1 | 2 | 1.087 [1.027, 1.147] | 1.026 [1.013, 1.040] |
| 1 | 4 | 1.073 [1.019, 1.126] | 1.032 [1.011, 1.054] |
| 2 | 1 | 1.069 [0.987, 1.152] | 1.044 [1.020, 1.068] |
| 2 | 2 | 1.103 [1.080, 1.127] | 1.032 [1.003, 1.061] |
| 2 | 4 | 1.058 [1.029, 1.087] | 0.981 [0.895, 1.068] |
| 4 | 1 | 1.078 [1.050, 1.106] | 1.046 [1.012, 1.080] |
| 4 | 2 | 1.059 [1.047, 1.071] | 1.025 [0.999, 1.051] |
| 4 | 4 | 1.043 [1.020, 1.065] | 1.030 [1.009, 1.051] |

结论是“非单调”，不是“node 越多越好”或“batch 越大越好”。新增 node 同时
增加远端路径、更多芯片和不同 packing 选择；batch 改变 grouped-M、queueing
和 replica 负载。只有一个 `2/4` serving 点低于 1，但其中仍存在静态 placement
收益，说明 serving 动态调度/争用会吞掉部分静态收益。

## 7. C 隔离

### 7.1 64 copies / 64 chips

该 panel 中 C=1 可行。所有 C 点使用相同的 64-chip、64-copy 资源。

| C | Coverage static | Overlay serving |
|---:|---:|---:|
| 1 | 1.275 [1.262, 1.287] | 1.125 [1.066, 1.185] |
| 2 | 1.262 [1.228, 1.297] | 1.123 [1.063, 1.184] |
| 4 | 1.262 [1.228, 1.297] | 1.123 [1.063, 1.184] |
| 8 | 1.262 [1.228, 1.297] | 1.123 [1.063, 1.184] |
| 14 | 1.262 [1.228, 1.297] | 1.123 [1.063, 1.184] |

### 7.2 32 copies / 16 chips

| C | Coverage static | Overlay serving |
|---:|---:|---:|
| 2 | 1.175 [1.134, 1.217] | 1.051 [1.021, 1.082] |
| 4 | 1.175 [1.134, 1.217] | 1.052 [1.022, 1.082] |
| 8 | 1.175 [1.134, 1.217] | 1.052 [1.022, 1.082] |
| 14 | 1.175 [1.134, 1.217] | 1.052 [1.022, 1.082] |

C 在当前 workload 中不是主瓶颈。C=1 不需要额外串行负担这一判断得到支持；
C≥2 之后 planner 的 copy 分配没有继续改变，因此结果平台化。仍不能据此认为
C 永远无影响：当每 chip 实际候选数超过 C，或 task trace 产生更强 node pressure
时，C 才会重新进入活跃约束。

## 8. 共激活聚类与 H/B

C=1 固定资源微基准中，unique、pair、full-hyperedge 的 completion 都是
`19.2007 us`，没有观察到额外开销，也没有观察到可利用的收益。

这说明：

- full-hyperedge plumbing 和 cut 计算可执行；
- 当前 synthetic community 没有提供足够强的 node-fanout 差异；
- 当前结果不能证明聚类优于 unique placement；
- 也不能否定真实 router trace 中的聚类收益。

当前正收益主要来自 coverage/overlay，而不是聚类。下一轮若要加强机制证据，
应优先引入 task-family conditioned community、node-affinity stress 和真实
hyperedge block bootstrap。

## 9. 刷新与迁移共流

Stage 0 中已执行从 GPU-primary plan 到 overlay plan 的 source-aware migration，
并把代表性 serving prefix（前 2 个 serving layer）的 flow 与 migration flow
放入同一个网络竞争。

| 模型 | Migration bytes | Migration completion | Shared serving delay | Migration energy |
|---|---:|---:|---:|---:|
| Mixtral | 2688 MiB | 34.409 ms | 0.000 us | 0.03946 J |
| DeepSeek-V3 | 1344 MiB | 13.765 ms | 0.000 us | 0.01973 J |
| Qwen3 | 576 MiB | 5.900 ms | 0.000 us | 0.00846 J |
| gpt-oss | 403.42 MiB | 4.133 ms | 0.000 us | 0.00592 J |

这里的 `serving_delay` 是逐 serving flow 比较“独立带宽”和“共享带宽”完成时间
后的最大增量，不是总 workload makespan 与 serving makespan 的差。

当前前缀重放没有观察到 serving flow 被迁移拖慢，但整体完成时间仍由 migration
主导。该结果只验证了共享 flow scheduler、版本切换和资源计量路径；它不能回答
完整 32/58/94 层 serving 下的 P99 和 refresh 净收益。

当前刷新结论必须保持为：

- 已经能够计算 source-aware migration bytes、时间、能耗和版本状态；
- 已经能够检测 drift 并在 safe point 激活新版本；
- 尚未完成“完整模型、长窗口、多任务 phase”的 never/periodic/task/drift/
  incremental/oracle 对比；
- 因而当前不能声称某刷新周期已经最优，也不能给出全模型 refresh ROI。

## 10. 能支持的结论与不能支持的结论

当前可以支持：

- 在 projected HP300 + synthetic trace 中，HBM fallback + selective Brain overlay
  的 coverage-first 是稳健正向机制；
- 热专家不要求固定 0/1 分类，连续边际规划可以改变 copy 选择；
- `R_unique < 1` 时允许 extra copies 已实现；
- H-exclusive 的收益依赖模型和 workload；
- node 数和 batch 存在非单调交互；
- C=1 不会自动造成额外串行成本；
- 迁移和 serving 可以共享同一个 flow-level 网络；
- 所有 final results 可由当前源码和记录参数重建。

当前不能支持：

- KA200/HP300 真机性能或能效；
- 真实 router popularity、task drift 或共激活分布；
- 完整模型的四层 fast-screen 外推；
- 真实 TCO、采购成本或最小 node 数；
- 聚类一定优于 non-clustering；
- 某一固定 refresh 周期；
- 完整模型 refresh 的净收益或 P99；
- 把 synthetic seed CI 当作真实 trace sampling CI。

## 11. Canonical artifacts

Stage 0：

```text
experiments/moe_2026_stage0_final_v12_mixtral/
experiments/moe_2026_stage0_final_v12_deepseek/
experiments/moe_2026_stage0_final_v12_qwen/
experiments/moe_2026_stage0_final_v12_gptoss/
```

Stage 1 / matrix / C / boundary：

```text
experiments/moe_2026_screen_final_v8_*/
experiments/moe_2026_matrix_final_v8_*/
experiments/moe_2026_cscan_final_v8_*/
experiments/moe_2026_cscan32_final_v8_*/
experiments/moe_2026_boundary20_v8_*/
```

`v11`、`v7` 及更早目录保留作审计或开发检查点，不作为最终引用。

## 12. 复现入口

Stage 0：

```powershell
python scripts/run_moe_stage0.py `
  --model deepseek_v3 --dtype fp8 `
  --c-max 4 --replicas 32 --hot-experts 16 `
  --seed 211 --brain-nodes 2 `
  --output NEW_STAGE0_DIR
```

Screen / matrix：

```powershell
python scripts/run_moe_screen.py `
  --model mixtral_8x7b --dtype fp8 `
  --zipf-exponent 1.3 `
  --seeds 101,102,103,104,105 `
  --c-max 2 --replicas 16 --hot-experts 8 `
  --brain-nodes 2 --requests 4 `
  --output NEW_SCREEN_DIR
```

每个最终 bundle 的 `summary.json` 记录决定实验结果的参数，`manifest.json`
记录生成时全部源码的 SHA-256。
