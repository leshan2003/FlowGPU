# MoE 实验启动门控审计 v2

> **历史快照：** 本文记录 Stage 0 执行前的 NO-GO 状态。当前实现和门控结果见
> [`MOE_STAGE0_执行审计.md`](MOE_STAGE0_执行审计.md)。其中“未实现”描述不应再作为
> 当前源码状态引用。

> 审计日期：2026-10-20  
> 审计对象：[`MOE_高置信度实验设计_v2_冻结版.md`](MOE_高置信度实验设计_v2_冻结版.md)  
> 审计结论：**NO-GO（不可启动全部正式实验）**  
> 可以执行：Stage 0 工程收口和小规模正确性验证。  
> 不可执行：Stage 1–5 的最终参数 sweep、刷新实验和成本结论。

## 1. 当前代码健康度

本轮只读验证：

```text
python -m pytest tests -q
113 passed in 5.15s

python -m compileall -q flowgpu scripts
COMPILEALL_OK
```

这只证明现有代码没有编译或回归错误。当前测试不覆盖冻结设计中的主要新增机制。

## 2. 十项门控

| 门控 | 状态 | 依据 |
|---|---|---|
| 问题意义 | 绿 | 研究问题、适用范围和负结果价值明确 |
| 时效性 | 绿 | 已覆盖 2025–2026 MoE placement/cache/migration 方向 |
| 最近工作差异 | 黄绿 | 已获取九篇 PDF 并完成章节级机制核对；ReXpert/HCRMap/Expert Streaming 很接近，仍缺逐页/chunk curation 和 peer-review 确认 |
| 核心洞察 | 黄 | co-serving、渐进复制和漂移刷新是合理假设，尚未获得模型内统一验证 |
| 机制完整性 | **红** | 显式 C、完整超边、H/B、版本迁移等核心机制未实现 |
| 可行性 | **红** | 当前实现只能完成 G 模式中的部分静态合成实验 |
| 主张可识别性 | **红** | 无法将 tiering、co-serving、clustering、refresh 的贡献完全分离 |
| baseline 公平性 | **红** | 当前 C 与拓扑耦合，overlay 增加硬件，尚无统一成本/功率边界 |
| 评价充分性 | **红** | 无真实 trace、无统一 serving/SLO 评价、样本量未做功效分析 |
| 贡献上限 | 黄 | 可形成 synthetic/projected 机制证据；不能形成 KA200 真机或绝对成本结论 |

任一硬门控为红即不能启动最终研究包。当前有四个硬红门。

## 3. 代码级阻塞项

### B1. 显式 `C_max` 不存在

当前 `ChipSpec` 只有 SRAM 容量，没有每 chip 最大 expert 数。现有实验通过
`capacity = C × expert_size` 间接制造 C，只对同尺寸 expert 成立。

影响：

- 多模型 expert 大小不同时，C 不再是真正的 expert 数上限；
- capacity 与 routing/queue concurrency 的 C 无法独立测量；
- 结果不能支持“C 是系统设计变量”的主张。

### B2. 当前 ranked 与设计要求不同

当前：

- `ranked-<fraction>` 先按固定 share 保留 coverage；
- `ranked-adaptive` 使用 top-decile 需求集中度选择固定阈值；
- `joint-marginal` 使用 popularity/load proxy，不是完整 event-model 边际收益。

设计要求：

- 以 bytes budget 为主；
- 每次比较 first/extra copy 的真实 net benefit；
- 允许 marginal surrogate 预筛，但最终候选必须过 event-model 和收益门槛。

结论：现有 ranked 数据不能作为“连续渐进式策略”的最终证据。

### B3. 聚类仍是 pair 与完整超边混合目标

当前 `coactivation_clustered` 将 top-k hyperedge 展开成所有 pair，而
`primary_coactivation_cut` 按完整 hyperedge 评估。没有：

- 完整超边放置目标；
- 固定 C；
- SLO/load no-worse guard；
- C=1 固定资源对照。
- 对 HBM-fallback extra-brain copies 的 cluster placement；当前聚类主要作用于
  primary replicas。

### B4. C=1 未通过设计级验收

现有测试覆盖了“coactivation 策略可在小 fixture 降低 cut”，但没有覆盖：

- 同 node/chip/trace/batch/copy budget；
- C=1；
- 端到端 completion 不倒退；
- rejected move/swap 统计。

### B5. H/B 模式不存在

当前实现：

- GPU HBM 总有 primary fallback；
- Brain 只保存额外副本；
- 没有 active Brain DRAM tier；
- 没有 H-exclusive 的 miss/wait/no-fallback；
- 没有 B-stage 的 DRAM→SRAM 完整 expert 装载。

### B6. 刷新只实现了离线近似

当前 epoch 模型：

- 在固定 epoch boundary 重新生成 planning/evaluation trace；
- migration 时间在 event execution 之外单独相加；
- 只按目标 node 和 bridge 带宽估算移动时间；
- 没有迁移与 serving 共享资源；
- 没有版本发布、在途 request、double buffer、回收或 rollback。

此外，每个 epoch 的 planning/evaluation trace 会重新随机生成 community
assignment，任务比例变化与生成器重新随机化没有被分离。

### B7. Trace 与统计链路不完整

当前严格 JSONL loader 拒绝额外 `task_family`、`stage`、timestamp/window
字段。synthetic trace 的 `request_id` 全部为 0，无法按 request block bootstrap。
当前也没有 `routing_summary.json` 中的 lift、top-set stability、drift CI 等完整产物。

### B8. 评价与公平性不完整

当前 MoE research suite 主要输出：

- MoE completion；
- tokens/s；
- GPU/brain assignment fraction；
- network energy。

尚缺最终主张需要的：

- 全模型 attention/KV；
- P50/P95/P99 SLO goodput；
- 系统 J/token；
- capex/TCO；
- 迁移期间拒绝和尾延迟；
- 相同成本或相同功率下的 G/H/B frontier。
- 无歧义的 `tokens/s/system|GPU|brain-node|normalized-unit` 分母。

当前 full static suite 对 DeepSeek 形状只使用 8 个 placement tokens 和
8 个 evaluation tokens；community suite 还会随 C 改变 node 数；node/batch suite
只到 8 nodes；seed suite 使用 3 pairs。这些是工程筛选配置，不是最终置信实验。

### B9. 结果不可复现

- `final_v3` 是最后相对完整包，但与当前源码哈希不一致；
- `final_v4` 的 ranked 数据早于预算修正；
- `mechanisms_v5` 中断；
- 当前源码尚无新的 canonical bundle。

### B10. 自动化保护不足

- 当前目录没有 `.git`；
- 没有阶段报告和阶段冻结产物；
- 没有“测试失败、守恒失败、provenance 缺失、结果目录冲突即停止”的统一 runner。

### B11. 多模型尚未真正接入 research runner

`scripts/run_moe_research.py` 的 expert geometry 由全局 `EXPERT_BYTES` 和
`make_experts(n_layers, n_experts)` 生成，默认只代表 DeepSeek-V3 形状。
Mixtral、Qwen、gpt-oss 的 expert size/top-k/shared-expert 差异尚未进入同一
placement/evaluation pipeline。

## 4. v2.1 已收紧的设计决定

- H 模式冻结为 H-exclusive，禁止静默 GPU fallback；
- B 模式第一阶段冻结为 B-stage，B-stream 为未来独立扩展；
- C 实验拆成 C1-clustering、C-scan、C-expansion 三个不可混合的 panel；
- 所有 budget 以 bytes 为主，copy count 为辅；
- 聚类使用完整超边目标和 SLO/load no-worse guard；
- primary endpoint 固定为 paired SLO goodput；
- 最终样本量需由 pilot variance 做功效分析；
- 多个确认性比较使用 Holm 校正；
- 无价格/功率输入时只能报告 frontier，不能报告成本优势；
- 容量和带宽输出必须显式标注 decimal/binary 单位。

## 5. 启动前必过清单

1. 创建 Git 初始提交或只读快照；
2. 修复容量单位不一致；
3. 加入 device/memory role、`C_max`、`R_unique/R_copy`；
4. 实现 HBM fallback 的唯一 G 语义；
5. 实现 marginal-event 规划器和异构 expert byte budget；
6. 实现 full-hyperedge clustering 与 no-worse guard；
7. 跑通 C=1 fixed-resource 测试和 small oracle；
8. 实现 H-exclusive 与 B-stage；
9. 实现 task-aware ordered trace、drift、migration DAG、versioning；
10. 将 refresh 接入 unified serving/SLO 评价；
11. 实现 request/task-block bootstrap 和功效分析脚本；
12. 生成与当前源码哈希一致的新 canonical bundle；
13. 通过全部新测试后才允许进入正式 sweep。

## 6. 证据上限

即使上述软件 Gate 全部通过，在没有真实 request-ordered trace 和 KA200 measured
artifact 时，最高结论仍只能是：

```text
software-model, synthetic-workload, projected-hardware evidence
```

不能写成 KA200 真机加速、真实 router 分布结论或实际 TCO 结论。

## 7. 最终判定

**当前为 NO-GO。**

设计方向不需要推翻，但必须先完成 Stage 0 和对应测试。当前冻结设计的价值是：后续
自动编码不需要重新讨论核心语义；不等于当前代码已经满足这份合同。

前人工作全文机制、baseline 接受状态和新增实验约束见
[`MOE_前人工作全文审计_v2.md`](MOE_前人工作全文审计_v2.md)。
