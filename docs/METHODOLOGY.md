# Methodology

How FlowGPU turns a model, a system and a placement into TTFT, TPOT,
throughput and joules — and which parts of that are trustworthy.

---

## 1. Workload representation

A workload is a DAG of coarse ops, each roughly one fused kernel. That is the
granularity at which roofline reasoning is meaningful, and it keeps a 61-layer
MoE model at a few hundred nodes rather than a few hundred thousand.

Each op carries `flops`, `weights`, `inputs`/`outputs`, `state_read_bytes` /
`state_write_bytes` (KV cache), the GEMM shape `(m, n, k)`, a `role`
(`attn` / `ffn` / `moe` / …), and a `stage` (`prefill` / `decode`).

### Three details that dominate the answer

**MoE weight traffic scales with distinct experts touched, not FLOPs.**
Under balanced routing, `T` tokens with top-`k` of `E` experts touch

```
E · (1 − (1 − k/E)^T)
```

distinct experts in expectation. At `T = 1` that is `k`; by `T = 512` it is
essentially all of `E`. A decode step at batch 1 reads 8/256 of DeepSeek-V3's
expert stack; at batch 512 it reads all of it. Modelling MoE weight traffic
as proportional to *active* parameters — the common shortcut — understates
large-batch decode by ~30×.

**MLA decode uses the absorbed form.** DeepSeek-style MLA caches only the
compressed latent (`kv_lora + qk_rope` = 576 elements per token) shared across
all 128 heads, and folds `W_kv_b` into Q and O so attention runs on the latent
directly. Modelling the naive materialised form instead would overstate
DeepSeek's KV traffic by ~10× and flatter GQA models by the same factor.
Tested against the published 70 KB/token figure.

**Prefill attention is causally masked and uses the full context.** FLOPs scale
with `chunk × (ctx_before + chunk) / 2`, not `chunk²`.

---

## 2. Device cost model

Each device prices an op into `(time, energy, breakdown)` via a multi-level
roofline:

```
t_mem      = max over memory levels L of (bytes served by L) / bandwidth(L)
t_compute  = flops / (peak(dtype) · efficiency(op, shards))
t_net      = NoC traffic / bisection bandwidth + hops · hop latency
t          = max(t_compute, t_mem, t_net, latency floor) + fixed overhead
```

Levels run concurrently, so a single level's bandwidth is the serialising
resource; that is why `max` rather than `sum`.

### Residency — the decisive modelling choice

`plan_residency()` decides which level serves each weight byte, and this is
where GPUs and dataflow chips diverge:

* **GPU:** weights → HBM, KV → HBM, activations → L2 at a configurable hit
  rate. L2 is 40–126 MB; a shard is tens of GB; there is no reuse across
  decode steps. So every decode step re-streams the shard.
* **Dataflow:** weights are pinned into core SRAM up to
  `sram_weight_frac · capacity` (default 0.85, leaving room for activations
  and double-buffering); the remainder spills to on-package DRAM if the part
  has any, and is reported as `spill_frac` if it does not.

The system layer checks per-device shares against per-device capacity and
emits a warning naming the shortfall. Nothing is silently truncated.

### GPU efficiency model

Achieved fraction of peak FLOPS is the product of:

* **tile quantisation** — `(M·N·K) / (padded M·N·K)`, with an *adaptive*
  M-tile: real kernels ship M-tiles from the MMA minimum (16 rows for bf16 on
  Hopper) up to 256, and the model picks the smallest that covers `M`. Pinning
  `tile_m = 128` would make every MoE decode look 8× worse than it is.
* **wave quantisation** — `n_tiles / (waves · SMs)`, including **split-K**:
  when the output is too small to give every SM a tile (the normal case for
  decode), a real kernel partitions the reduction dimension and reduces the
  partials. The model applies a `1/(1 + 0.18·log2(split))` penalty for the
  workspace round trip.
* **grouped GEMM** — `op.meta['groups']` (MoE experts touched) multiplies the
  tile count; expert-parallelism divides it.
* **K-pipeline fill** — short reduction chains cannot hide MMA latency.
* **achievable peak** — 0.45–0.80 by architecture maturity.

Tensor-parallel sharding narrows the GEMM's `N` (and attention's batched `M`)
per device, which can push a large GEMM back into the wave-quantisation
regime — a real and commonly-ignored cost of aggressive TP.

### Dataflow efficiency model

Structurally different, on purpose:

* **pipeline fill** `M / (M + pipeline_depth)` instead of M-tile quantisation.
  A weight-stationary array streams tokens through; it does not waste 127 of
  128 output rows at batch 1. This is precisely why small-batch decode hurts
  the architecture less.
* **PE-array tile quantisation** against `pe_rows × pe_cols`.
* **core-count quantisation** — can the op fill every core?
* **NoC mapping** — a GEMM is tiled over an `R × C` core grid, `R` splitting
  the reduction dimension and `C` the output dimension. The mapper picks
  `(R, C)` to minimise traffic subject to per-core SRAM capacity, accounting
  for how many *chips* the op is already spread across. With hardware
  multicast it prefers `R = 1` (broadcast activations, no cross-core
  reduction), which is what a real dataflow compiler does.

### Spiking parts

SNN devices (`TrueNorth`, `Loihi 2`, `SpiNNaker 2`, `BrainScaleS 2`,
`TianjicX`, `Darwin3`) scale work by `spike_rate` and declare a restricted
`supported_dtypes`. Running a dense bf16 transformer on them falls back to the
nearest supported precision and the result should be read as an
order-of-magnitude statement, not a prediction. They are included for
architectural comparison, not because they can serve DeepSeek-V3.

---

## 3. Interconnect

Three tiers plus the one that decides this study:

| tier | modelled as |
|---|---|
| NoC | inside the device model: bisection bandwidth + per-hop latency + topology-dependent average hop count |
| inter-chip (scale-up) | `Link` with bandwidth, one-way latency, protocol efficiency (0.78–0.95), pJ/byte |
| inter-board (scale-out) | same, with realistic RoCE/IB latencies |
| **bridge** | the GPU ↔ dataflow path — charged on *every* pool change between consecutive ops |

Collectives: ring all-reduce `2(n−1)/n · bytes` over `2(n−1)` latency hops,
all-to-all `(n−1)/n · bytes`, all-gather likewise. A configurable
`collective_overlap` models what production engines hide behind compute
(DualPipe, SGLang two-batch overlap); MoE dispatch/combine is fully
overlappable, TP all-reduce only 45 % so, because it sits on the critical path
of the GEMM that produced its input.

**Why the bridge matters so much.** Under `pd_a`, the hidden state crosses the
bridge twice per layer per decode step: 122 crossings for a 61-layer model.
Under `layer_split` it crosses twice per *step*. The simulator charges both
honestly, and warns loudly if a system declares no bridge at all.

---

## 4. Serving simulation

Continuous batching with chunked prefill, modelled as **two engines with
per-pool availability clocks**:

* a prefill iteration and a decode iteration may run concurrently **iff they
  touch disjoint pools** — which is exactly what prefill/decode disaggregation
  buys, and what a single global clock would hide;
* when they share a pool they serialise, prefill winning ties so TTFT is not
  starved;
* an iteration cannot start before the current simulated time, so work never
  executes "before" the request that caused it arrived.

KV cache capacity is computed from the pool that hosts attention, net of that
pool's weight share and a runtime reserve, and enforced as admission control.
A request that cannot fit is never admitted and the run is flagged
`INFEASIBLE` rather than quietly producing a number.

**Data parallelism** is handled by simulating one replica and scaling
throughput and active energy by `dp` — exact for symmetric replicas, and `dp`
times faster than simulating all of them. Pools with inconsistent `dp` are
flagged.

**Micro-batch pipelining** splits a decode step across pools. The executor
refuses it when only one pool is involved: halving the tokens barely halves a
weight-bound decode step, so micro-batching a single pool is strictly worse,
and a real scheduler would not do it.

**Jitter** is a lognormal perturbation per iteration with per-device sigma —
0.06–0.12 for GPUs (CTA launch skew, cache-state variance, dispatch bubbles),
0.001–0.005 for statically scheduled dataflow parts. This is the "调度抖动"
the reference article names, and it is why p99 and mean TPOT can tell
different stories.

Step costs are memoised on `(stage, batch bucket, context bucket)`. Prefill
uses the *exact* batch — rounding it up would inflate FLOPs several-fold since
`chunk_tokens` already caps the work — while decode, whose cost is dominated
by batch-independent weight streaming, is bucketed.

---

## 5. Power model

Two stages, so neither literature nor marketing dominates.

**Stage 1 — apportionment** from a technology table: Horowitz's ISSCC-2014
45 nm energy-per-operation anchors, node-scaled with separate curves for logic
and SRAM (SRAM scaling flattens hard below 7 nm, and that asymmetry is exactly
why on-chip-memory architectures have been gaining ground). Off-chip memory
uses published per-bit figures by DRAM generation (HBM2/2e/3/3e, GDDR6,
LPDDR4/5, DDR4/5). SRAM energy per byte grows with array size along an
interpolated curve.

**Stage 2 — absolute scale pinned to TDP.** A reference operating point is
defined (a heavy kernel at `util_compute` of peak FLOPS while pulling
`util_bw` of peak memory bandwidth), and the dynamic coefficients are scaled
so modelled power at that point equals TDP minus leakage.

This matters more than it looks. Raw scaled Horowitz numbers predict an H100
doing dense bf16 at ~200 W — wrong by 3×, because real chips spend a large
fraction of their power on clock distribution, register-file traffic and
control that a per-FLOP number ignores. Anchoring to TDP absorbs all of that,
and keeps the *comparison* honest because both architectures get identical
treatment. A test asserts the calibration reproduces each device's TDP to
within 2 %.

Total energy = active device energy + **idle energy of under-used pools**
(leakage × idle time × device count) + host overhead (per pool, since a rack
of 16-chip cards does not carry a 400 W x86 host per card) — all × PUE.

The idle term is not a detail. In the PD+A configurations it is 28–68 % of
total energy, because the architecture's whole premise requires provisioning
enough SRAM to hold the model whether or not it is busy.

### EDA calibration

`flowgpu/power/eda.py` drives a real Synopsys flow on the EDA host: five
primitives (INT8 MAC array, 16-bit MAC array, SRAM tile, 5-port NoC router,
integrated PE tile) → Design Compiler → PrimeTime PX, with switching activity
either measured from a gate-level VCS SAIF (where the SAIF system tasks work)
or set from the derived figures below.

**Activity, and a correction.** It is often claimed that quantised activations
switch much less than uniform-random data. `eda/scripts/activity.py` computes
the actual per-bit toggle rate for INT8 activations drawn from the
distribution they really follow (roughly Gaussian, σ ≈ 24 LSB of ±127):

| lag-1 correlation ρ | b0..b7 toggle rates | mean |
|---|---|---|
| 0.00 | .50 .50 .50 .50 .50 .50 .50 .50 | 0.500 |
| 0.50 | .50 .50 .50 .50 .50 .48 .35 .33 | 0.457 |
| 0.90 | .50 .50 .50 .50 .46 .27 .15 .14 | 0.377 |
| 0.98 | .50 .50 .50 .43 .24 .12 .07 .06 | 0.303 |

For *independent* samples every bit toggles at 0.50 — indistinguishable from
uniform random. Quantisation alone does not reduce switching: the sign bit and
its extension dominate the high bits, and those are as random as the data. The
reduction appears only with **temporal correlation**, which streaming
activations do have along the reduction dimension. The flow uses ρ = 0.5
(mean toggle 0.457), deliberately the value that perturbs the measured
MAC : SRAM ratio least, so the result is not an artefact of an optimistic
activity assumption.

**Clock period.** `lsi_10k` is a ~0.5 µm library, so a 64-multiplier array has
~100 ns of combinational depth. The flow runs at 200 ns so every primitive
closes timing with the same low optimisation effort. That matters for the
*comparison*: if one block closes easily and another is aggressively
restructured chasing an impossible target, their energy ratio reflects DC's
effort allocation rather than the circuits. Energy per operation (E = P·T),
which is all the simulator consumes, is unaffected by the choice.

**What transfers and what does not.** The only Liberty library readable on the
available host is Synopsys' shipped `lsi_10k`, a ~0.5 µm educational library.
Absolute joules from it are meaningless for a 5 nm part. What is used is the
**ratio** `E(SRAM byte) / E(MAC FLOP)` and `E(NoC byte·hop) / E(MAC FLOP)`,
measured under identical conditions and identical synthesis effort (plain
`compile`, not `compile_ultra`, so no block gets more optimisation than
another). A documented ×10 correction converts the flip-flop-array scratchpad
into a 6T SRAM macro equivalent, since no memory compiler is available. The
absolute scale is then re-anchored to each device's TDP as above.

If the EDA database is absent, the technology table is used and every report
says `source = tech@Xnm + TDP` instead of `eda:<run> + TDP`.

---

## 6. What is measured, and why capacity is the honest metric

Comparing two systems at a **fixed offered load** measures the load, not the
systems. Offer 2 req/s to a machine that can do 10 and to one that can do 3
and both report ~2 req/s.

`flowgpu capacity` therefore binary-searches the Poisson arrival rate at which
SLO attainment crosses 95 %, discarding a warm-up transient, and reports
sustained throughput, power and energy per token **at that point**. That —
serving capacity subject to a latency SLO — is what "2× inference output"
has to mean.

When the SLO is never met even at the lowest probed rate, the result says so
explicitly rather than presenting the floor as a capacity.

---

## 7. Known limitations

* **Routing skew is not modelled.** Balanced MoE routing is assumed. Real
  routing is skewed, and a hot-expert SRAM cache backed by DRAM would
  materially change the capacity-per-watt arithmetic. This is the single most
  important gap, and the one most likely to move the conclusions.
* **No speculative decoding / MTP.** DeepSeek-V3 ships a multi-token
  prediction head; accepting 1.8 tokens/step would cut effective TPOT by ~45 %
  on both architectures.
* **No prefix caching.** Agent workloads reuse long system prompts; modelling
  that would shrink the prefill fraction and shift the balance toward the
  offload architecture.
* **Ops are priced independently.** Kernel fusion beyond the assumed
  granularity, and cross-op L2 reuse, are not modelled.
* **The bridge is a single aggregated link** with no congestion model beyond
  a fixed efficiency factor.
* **`pp > 1` does not model pipeline bubbles** at the request level; it only
  shards layers.
* **Several device specs are estimates** (`confidence: estimate` /
  `derived`), notably the Iluvatar cache hierarchy and everything about the
  Lynxi parts beyond the published TOPS and neuron counts.

Each of these is listed with its likely direction of effect in
[`FINDINGS.md` §8](FINDINGS.md).
