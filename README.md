# FlowGPU

A Python simulator for **many-core dataflow / brain-inspired chips + GPUs**
running large-model inference, built to test a specific claim.

## The claim under test

At the 2026 中国算力大会, Lynxi (灵汐科技), China Mobile (Suzhou) Software, CETC
Nanhu, Iluvatar CoreX (天数智芯), Tsinghua and Peking University announced a
**GPU + brain-inspired-chip heterogeneous hybrid inference system**
([announcement](https://mp.weixin.qq.com/s/Fi4AgXUpkPAkdOyXKRUUyg)).  Its
architecture is a **PD+A split**:

> Prefill 阶段及 Attention 计算模块交由国产GPU承载，FFN（MOE专家）延时敏感模块
> 交由类脑芯片处理
>
> *(Prefill and the attention modules go on the domestic GPU; the FFN / MoE
> experts, which are latency-sensitive, go on the brain-inspired chip.)*

and its headline result is:

> 数据显示（3台天数较早型号的GPU服务器+3台类脑机柜），运行 Deepseek V4 Flash
> 模型，相较同等投入规模的纯 GPU 集群，**推理输出和推理能效均提升 1 倍以上**，
> 业务运营成本下降 40% 以上。
>
> *(3 Iluvatar GPU servers + 3 brain-chip cabinets running DeepSeek V4 Flash:
> more than 2x inference output **and** 2x energy efficiency versus a pure-GPU
> cluster of equivalent investment; >40% lower operating cost.)*

FlowGPU exists to work out, quantitatively, **whether and when that can be
true** — and, where it cannot, what the hardware would have to look like for
it to become true.

## What it models

| Layer | What it captures |
|---|---|
| **Dataflow / brain chip** | core count, SRAM per core, FLOPS per core, NoC topology + bisection bandwidth + hop latency, presence/size/bandwidth of on-package DRAM, inter-chip bandwidth, inter-board bandwidth, static-schedule determinism, spike sparsity for SNN parts |
| **GPU** | SM count, register file / shared memory / L1 / L2 / HBM capacities *and* bandwidths, per-dtype peak, tile & wave quantisation, split-K, L2 hit rate, kernel-launch overhead, scheduling jitter |
| **Interconnect** | on-chip NoC, inter-chip scale-up, inter-board scale-out, and the **GPU↔dataflow bridge** — plus ring/tree all-reduce, all-to-all and all-gather cost models |
| **Workloads** | dense LLM, MoE LLM, MLA (DeepSeek), VLM (ViT + decoder), diffusion (DiT and U-Net), CNN |
| **Placement** | fully configurable rule-based mapping of any op (by role / stage / layer / kind / name regex) to any device pool |
| **Serving** | continuous batching, chunked prefill, paged KV with real capacity limits, concurrent prefill/decode engines, micro-batch pipelining, TP/PP/EP/DP |
| **Power** | per-level memory energy, per-dtype compute energy, NoC and SerDes energy, leakage, idle power of under-used pools, host overhead, PUE — anchored to each device's published TDP and optionally calibrated by real Synopsys DC + PrimeTime PX runs |
| **Results** | TTFT, TPOT, E2E latency (mean/p50/p90/p99), throughput, SLO attainment, goodput, SLO-constrained capacity, power, J/token, tokens/s/kW, tokens/s per unit capex |

## Quick start

```bash
python3 -m flowgpu devices          # the device zoo, with spec-confidence tags
python3 -m flowgpu models           # the model zoo
python3 -m flowgpu device groq_lpu_v1
python3 -m flowgpu model deepseek_v3

# run the headline experiment
python3 -m flowgpu run configs/experiments/grounded_published_hw.yaml

# the number the "2x output" claim is really about
python3 -m flowgpu capacity configs/experiments/grounded_published_hw.yaml

# what would the brain chip have to be?
python3 scripts/run_breakeven.py

# non-LLM workloads
python3 scripts/run_multimodal.py

# energy calibration on real EDA tools (Synopsys DC + PrimeTime PX)
python3 -m flowgpu eda
```

Tests:

```bash
python3 tests/test_flowgpu.py       # or: python3 -m pytest tests/ -q
```

## The physics it gets right

Everything hinges on **where a weight byte is read from, and how many times**.

* A GPU keeps its weight shard in HBM.  Every decode step re-streams the whole
  active shard.  At batch *B* the arithmetic intensity is ~*B*, so decode runs
  at a fraction of a percent of peak FLOPS and is bounded by HBM bandwidth.
* A dataflow chip that holds its shard in distributed core SRAM reads it at
  20–100× the bandwidth and ~10–30× lower energy per byte.
* The price is capacity.  SRAM is ~1000× less dense than HBM per mm², so the
  model spreads over far more chips — and *static* power scales with chip
  count whether or not those chips are busy.

FlowGPU makes both sides of that trade explicit, and refuses to let either
side cheat: a pool whose per-device share does not fit reports a spill
fraction and pays for it; a system with no declared bridge warns that its
crossings are being charged at zero.

Two modelling details that change conclusions by an order of magnitude and are
easy to get wrong:

* **MoE weight traffic scales with *distinct experts touched*, not FLOPs.**
  With *E* experts and top-*k*, *T* tokens touch `E·(1 − (1 − k/E)^T)` experts.
  At *T*=1 that is *k*; by *T*=512 it is all of them.
* **MLA decode uses the absorbed form.**  DeepSeek caches only the 576-element
  latent per token, shared across all 128 heads.  Modelling the naive form
  would overstate its KV traffic ~10× and flatter GQA models correspondingly.

Both are covered by tests against published numbers.

## Honesty about inputs

Every device carries a `source` and a `confidence` tag
(`vendor` / `paper` / `press` / `derived` / `estimate`), surfaced in
`flowgpu devices` and in every system report.  Two things in this study are
**projections, not specs**, and are named `*_proj`:

* `lynxi_hp300_proj` — Lynxi publishes neuron and synapse counts for KA200,
  not SRAM bytes or memory bandwidth, and has not published a successor.  The
  projected part is the simulator author's construction, sized to make the
  announced rack physically coherent.
* `deepseek_v4_flash_proj` — no config for "DeepSeek V4 Flash" has been
  published.

For that reason the study is run **twice**: once with the projected parts
(`configs/experiments/pd_a_reference.yaml`) and once with nothing projected at
all (`configs/experiments/grounded_published_hw.yaml`: DeepSeek-V3's published
config on H100, Groq LPU and Graphcore IPU datasheets).  Conclusions that only
survive in the first run are labelled as such.

## What the simulator concludes

Full detail in [`docs/FINDINGS.md`](docs/FINDINGS.md); the short version:

| Claim | Verdict |
|---|---|
| **低时延** — lower per-token latency | **Supported.** 1.26–1.58x lower TPOT at batch 1. |
| **>2x inference output** at equal investment | **Achievable — but not with the announced split.** A layer-split placement on published Graphcore IPUs reaches **2.56x** SLO-constrained capacity. The announced PD+A split reaches **0.83x**. |
| **>2x energy efficiency** | **Not reproduced in any configuration.** Best case **0.44x** (2.3x *worse*); the 2.56x throughput arm costs 0.21x efficiency. Held across SRAM densities 0.5–32 MB/core, chip TDPs 20–250 W, and bridge bandwidths 0.1–19 TB/s. |
| **>40% lower operating cost** | **Not reproduced.** Best tokens/s per unit capex was 1.31x the pure-GPU baseline. |

The most actionable result is one the announcement does not claim:

> **Pinning *all* prefill to the GPU is what breaks the throughput case.**
> Under PD+A the dataflow pool does **1.6% of the work while drawing 28% of
> the energy**. Splitting by layer instead — same hardware, same cost — takes
> 0.59x to 2.56x.

The energy verdict is robust to how the power model is calibrated: substituting
**measured Synopsys DC + PrimeTime PX coefficients** for the technology table
moves every energy number by under 3%, because the result is driven by chip
count and leakage rather than by per-byte coefficients. The measured 16-bit :
8-bit MAC energy ratio came out at exactly 4.00x — the width² scaling a
multiplier array should show.

The energy result follows from SRAM capacity-per-watt
(0.005 GiB/W for a dense SRAM part vs 0.114 GiB/W for HBM3), so holding a
140 GiB expert stack on-chip costs 200–3400 chips whose *leakage alone*
exceeds the whole pure-GPU system's power.

## Documentation

* [`docs/FINDINGS.md`](docs/FINDINGS.md) — results, with the break-even and
  sensitivity analysis
* [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — the cost model, what it gets
  right, and its known limitations
* [`docs/DEVICES.md`](docs/DEVICES.md) — every device and model with its
  source, confidence tag, calibrated energy coefficients and a die-area
  plausibility check (auto-generated by `docs/gen_devices.py`)

## Layout

```
flowgpu/
  units.py            unit parsing (binary for capacity, decimal for rates)
  hardware/
    base.py           Device, MemoryLevel, ComputeSpec, EnergyModel, OpCost
    gpu.py            SM-based model: tile/wave quantisation, split-K, L2
    dataflow.py       many-core model: weight residency, NoC mapping, SNN
    interconnect.py   links, collectives, fabric catalogue
    registry.py       device zoo (17 GPUs, 17 dataflow/brain chips)
  workload/
    graph.py          op IR + ModelSpec
    llm.py            dense / MoE / MLA prefill and decode graphs
    vision.py         ViT, DiT, U-Net, VAE, CNN
    models.py         model zoo
  mapping/placement.py   configurable op -> pool rules and named strategies
  sim/
    executor.py       analytic costing of one graph on one system
    serving.py        continuous-batching request simulator
    oneshot.py        single-graph workloads (diffusion / CNN / VLM)
  power/
    tech.py           technology-node energy table
    model.py          TDP-anchored calibration
    eda.py            Synopsys DC + PrimeTime PX flow driver
  system.py           pools, parallelism, bridges, capacity checks
  capacity.py         SLO-constrained capacity search
  breakeven.py        inverse analysis
  report/             tables, JSON, matplotlib figures
  cli.py
eda/                  RTL + TCL for the energy calibration flow
configs/              system and experiment definitions
```
