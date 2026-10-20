# 路由模型

## 1. Trace 记录

推荐的最小记录：

```json
{
  "request_id": 0,
  "token_id": 0,
  "step": 0,
  "layer_id": 3,
  "home_node": "node0",
  "experts": [7, 19, 21, 44, 83, 90, 130, 201],
  "gate_weights": [0.2, 0.18, 0.15, 0.13, 0.11, 0.09, 0.08, 0.06]
}
```

真实 trace MAY 匿名化 request/token 标识，但 MUST 保留层、顺序、专家集合、home-node
或可重建 home-node 的信息。

固定窗口和 tiered CLI 可把 `workload.routing_model` 设为 `trace`，并配置：

```yaml
workload:
  routing_model: trace
  placement_trace_template: traces/placement-{case_id}.jsonl
  evaluation_trace_template: traces/evaluation-{case_id}.jsonl
```

相对路径以 YAML 所在目录为基准。`{case_id}` 会展开成当前 node-count/ingress case；
每个实际输入文件都写入结果 manifest 的 SHA-256。loader 会校验 token 数、层集合、
top-k、expert 范围和 home-node 是否属于当前 case。它不会为了让同一 trace 适配不同
node 数而静默改写 home-node。固定 trace 也不能进入 seed uncertainty；serving 使用
在线 router 模型，不能把有限 fixed-window trace 循环冒充在线路由。

## 2. Train/evaluation 隔离

静态放置只能读取 placement window。评估 MUST 使用 held-out window。

至少报告：

- in-distribution held-out；
- 热点漂移；
- node affinity 漂移；
- coactivation 社区漂移。

使用当前评估 batch 计算副本放置只能作为 oracle 上界，必须明确标注。

## 3. Synthetic families

### 3.1 Uniform independent

top-k 从专家集合均匀无放回采样。它是“没有可利用结构”的必要负对照。

### 3.2 Global popularity

使用可控 Zipf 或显式概率向量 `p[e]`。生成器 MUST 支持 load-balancing/capacity
限制，避免产生模型运行时不可能接受的负载。

### 3.3 Node-affine

使用条件概率：

```text
p(expert | home_node)
```

控制 affinity 强度，同时保持指定的全局边际热度。

### 3.4 Coactivation communities

控制专家间相关性或 hyperedge 社区强度，同时允许固定 `p[e]`，以隔离共激活收益。

### 3.5 Temporal drift/burst

热点、affinity 或社区结构随 step/window 改变。必须能控制变化幅度与持续时间。

## 4. 正交对照

实验 MUST 包含：

- 相同 `p[e]`、不同共激活强度；
- 相同共激活、不同 `p[e]`；
- 相同全局 `p[e]`、不同 node affinity；
- 相同长期统计、不同 burstiness。

这样才能区分复制、聚类和排队收益。

## 5. MoE 约束

生成器和 trace loader MUST 支持：

- top-k 内专家不重复；
- shared expert；
- expert capacity factor；
- token overflow/drop；
- load-balancing 规则；
- group/node-limited routing；
- 每层不同分布；
- 不同请求类型的条件分布。

## 6. 统计摘要

每个 trace MUST 输出：

- 每层专家边际命中率；
- distinct experts per batch/window；
- assignments per touched expert；
- 最大/均值/P95 expert load；
- pairwise/hyperedge coactivation；
- node affinity；
- 时间漂移指标；
- shared expert 负载。
