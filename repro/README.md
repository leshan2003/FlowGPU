# repro/ -- local reproduction and diagnostic workspace

Not part of the package. A workbench for "run it, poke at it, tune it", plus
the raw console output every conclusion in here is drawn from, so a reader can
check the claims without re-running anything.

Everything in this directory only *reads* the `flowgpu` package, except where a
section below says otherwise. Run from the repository root.

## Evidence files

| File | Contents |
|---|---|
| `devices.txt` / `models.txt` / `placements.txt` / `fabrics.txt` | device zoo, model zoo, placement strategies, interconnect fabrics |
| `quickstart.txt` / `multimodal.txt` / `make_figures.txt` | the three shipped scripts |
| `grounded_run.txt` / `grounded_capacity.txt` | `grounded_published_hw.yaml`, fixed load and capacity |
| `regimes_run.txt` / `fixed_regimes.txt` | `regimes.yaml`, before and after the batch-bucketing fix |
| `pda_capacity.txt` / `fixed_grounded_capacity.txt` | capacity searches, before and after |
| `capacity_timing_all_fixed.txt` | per-arm timings for the equal-capex arms |
| `bucket_gpu.txt` | `batch_bucket` sensitivity 1/2/4/8/16 -- the measurement behind the batch-bucketing bug |
| `profile_pda.txt` / `profile_pda192.txt` | cProfile output |
| `eda_dryrun.txt` | `flowgpu eda --dry-run --local-only` |
| `v41_run.txt` | `deepseek_v41_flash.yaml` |
| `hot_cold_probe.txt` | hot/cold expert tiering, sections A-G (below) |
| `hot_cold_capacity.txt` | the same arms through the serving simulator, swept over `collective_overlap` |
| `pipeline_overlap_probe.txt` | micro-batch tax vs overlap gain, and the collective-overlap sweep |

## Scripts

All read-only with respect to the package. Import with `sys.path.insert(0, ".")`.

```
python repro/cache_stats.py             <config> <arm> [requests] [rate]
python repro/bucket_sensitivity.py      <config> <arm>
python repro/capacity_timing.py         <config> <arm|ALL> [requests] [iters]
python repro/profile_probe.py           <config> <arm> [requests] [rate]
python repro/hot_cold_probe.py
python repro/hot_cold_capacity.py
python repro/pipeline_overlap_probe.py
```

- `cache_stats.py` -- how many graph keys one probe builds. Prefill uses the
  exact batch and chunk, so it keys a lot; decode buckets, so it keys few.
  Use it to tell a graph-construction bottleneck from a simulation-loop one.
- `bucket_sensitivity.py` -- clone a load and change only `batch_bucket`, to see
  how TPOT/SLO/throughput move. It is the tool that found the "decode cost comes
  from the bucket, not the batch" defect.
- `capacity_timing.py` -- timestamps every `find_capacity` probe, with
  `FLOWGPU_WATCHDOG=<seconds>` (Windows has no `signal.alarm`, so it kills from
  a daemon thread). Separates "one arm is pathologically slow" from "the whole
  machine is saturated".
- `profile_probe.py` -- sorted cProfile dump.
- `hot_cold_probe.py` -- seven sections, below.
- `hot_cold_capacity.py` -- SLO capacity over the arms, below.
- `pipeline_overlap_probe.py` -- micro-batch decomposition and `co` sweep, below.

## Hot/cold expert tiering (`hot_cold_*`)

The question: if only part of the MoE expert stack is pinned on the brain
chip and the rest is served from somewhere else, does the saving beat the
cost?

### `hot_cold_probe.py`, sections A-G

| Section | Question |
|---|---|
| A | Where does a decode step spend its time (compute / memory / network / bridge / collective)? |
| B | Does the brain pool actually hold the expert stack, vs chip count (`spill_frac`)? |
| C | If the *cold* experts move to the GPU's HBM, does the capacity fit? |
| D | Past the point where the stack fits, does another chip still help? |
| E | What does going multi-node cost, at fixed chip count? |
| F | A **hand-computed** `hot_frac x skew` map (labelled as such -- not a pipeline result) |
| G | The real thing: build the split graph and price `pd_a` against `hot_cold` |

Findings worth keeping:

- **NoC is 0.1-0.3% of the step.** On-chip distance is not a bottleneck, so
  replicating hot experts to shorten NoC routes buys nothing.
- **The MoE all-to-all is the largest term**, 25-57% depending on batch.
- **~1000 brain chips** are needed for a zero-spill DeepSeek-V3 expert stack;
  512 chips still spills 46.7%.
- **Marginal chips are worth almost nothing past the fit point**: 1024 -> 4096
  chips (4x the silicon, 4x the leakage) buys 4% on the step.
- **Multi-node is pure cost**: same chips, same SRAM, but the collective falls
  from `scaleup` to `scaleout` as soon as a node holds fewer than the full
  EP degree.

### `hot_cold_capacity.py`

Same arms through `ServingSimulator`, at SLO `ttft 5 s / tpot 50 ms`, target
95% attainment, 48 requests of prompt 1024 / output 64, seed 11. Four arms:

| Arm | Cold experts go to | Brain chips |
|---|---|---|
| A | (all pinned, nothing spilled) | 1024 |
| B | the brain pool's own DRAM | 256 |
| C | the GPU's HBM | 256 |
| C' | the GPU's HBM | 128 |

`collective_overlap` is swept because it turned out to dominate everything else.
It is **scheduler quality, not hardware**: the fraction of the MoE all-to-all a
production engine hides behind compute (DualPipe and SGLang two-batch overlap
reach ~0.6-0.9; a naive engine is ~0). The package default is 0.0 and the
shipped configs use 0.6.

Throughput, tok/s:

| Arm | co=0.0 | co=0.6 | co=0.9 |
|---|---|---|---|
| A pd_a @1024 | 1070.0 | 1598.4 | 2122.3 |
| B pd_a @256 cold->DRAM | 314.9 | 642.1 | 712.7 |
| C hot_cold @256 cold->HBM | 1140.5 | 1594.8 | 1991.5 |
| C' hot_cold @128 cold->HBM | 1116.4 | 1547.7 | 1918.2 |

Energy per output token, J (lower is better):

| Arm | co=0.0 | co=0.6 | co=0.9 |
|---|---|---|---|
| A | 47.30 | 33.75 | 26.98 |
| B | 55.02 | 27.56 | 25.64 |
| C | 13.68 | 10.39 | 8.74 |
| C' | 8.33 | 6.48 | 5.56 |

Three things fall out of that:

1. **At co=0.9, 128 brain chips (12% of the baseline silicon) deliver 90% of
   the throughput at 4.85x the energy efficiency.**
2. **A worse engine favours `hot_cold`.** At co=0.0 the 256-chip `hot_cold`
   arm actually beats the 1024-chip baseline (1140 vs 1070 tok/s), because
   `pd_a` has the largest collective share and therefore depends most on
   scheduler quality.
3. **At matched chip count, "cold on the GPU's HBM" beats "cold on the brain
   pool's own DRAM" by 2.8-3.6x** (1991 vs 713 tok/s at co=0.9). Arm B hugs
   both SLO limits at co=0.0: TTFT p99 4.54 s against a 5 s budget, TPOT p99
   49.43 ms against 50 ms.

## Micro-batch pipelining and collective overlap (`pipeline_overlap_probe.py`)

Two questions with different answers.

### A. Does micro-batch pipelining pay?

It is two effects multiplied, and reporting only the net hides which one wins:

```
overlap_gain   = (m * one_microbatch) / pipelined_time
microbatch_tax = (m * one_microbatch) / serial_time
net            = overlap_gain / microbatch_tax
```

The tax exists because a weight-bound decode step does not get m times cheaper
when cut into m pieces -- the weights stream in full either way. The gain exists
because with two or more micro-batches the steady-state period is the *busiest*
pool, not the sum.

At `co=0.6`, batch 8 the tax is 1.63-3.48 and every arm loses (net 0.33-0.74).
At batch 256 the tax falls to 1.15 and it turns marginally positive: 1.04 for
`pd_a`, 1.13 for `hot_cold`. `hot_cold` wins the comparison because it balances
the two pools -- the same reason it wins on capacity.

### B. How much is left in collective overlap?

Raising `co` from 0 to 0.9 on a single decode step:

| Arm | batch | speedup |
|---|---|---|
| pd_a @256 | 8 | 1.63x |
| pd_a @256 | 64 | **2.23x** |
| hot_cold @256 | 8 | 1.36x |
| hot_cold @256 | 64 | 1.79x |

By far the largest single lever found in this work -- an order of magnitude
above anything micro-batching offers. `hot_cold` depends on it less, because the
collective is a smaller share of its step (40-54% against `pd_a`'s 53-65%), so
it is both faster and less sensitive to scheduler quality.

### Two scoping notes

- Section A deliberately re-derives `execute_pipelined`'s arithmetic, because
  the package exposes only the product. It imports the private `_scale_graph`,
  so it is a diagnostic, not package API.
- Section B prices a single decode graph, not a serving run -- its numbers bound
  the step, not the SLO capacity. `hot_cold_capacity.py` is the one that answers
  the capacity question.

## Package changes these scripts depend on

Not in `repro/`; each needs separate review.

| File | Change |
|---|---|
| `flowgpu/workload/llm.py` | `_moe_block` honours `hot_frac` / `hot_traffic` and emits `L*.moe_hot` + `L*.moe_cold` + `moe_combine` instead of one `L*.moe_experts` |
| `flowgpu/mapping/placement.py` | new named strategy `hot_cold` |
| `flowgpu/hardware/dataflow.py` | expert weights priced as *indivisible* units (below) |
| `flowgpu/hardware/gpu.py` | same, so the two sides agree |
| `flowgpu/sim/serving.py` | the FFN tail all-reduce is gated on the FFN pool's tensor parallelism (`ffn_tp_world`), not the attention pool's |
| `flowgpu/sim/executor.py` | `_scale_graph` now scales `meta["bytes"]` too (below) |

### Why `_byte_map` changed

The old form was `wb = op.weight_bytes / shards`, i.e. spreading the weights of
a `groups`-way grouped GEMM evenly across every device in the pool -- which lets
a single expert access ride the pool's *aggregate* bandwidth. Expert weights are
indivisible: one expert sits whole on one chip, so a hit only ever gets that one
chip's bandwidth. The new form is

```
wb = ceil(groups / shards) * (weight_bytes / groups)
```

This is load-bearing. On `pd_a` with 47% spill at batch 1 it moves the step from
3.14 ms to 9.01 ms. **It is our own modelling choice and still needs independent
review.** It assumes experts are evenly distributed and that a given hit lands
on its own chip; real routing skew would make the busiest chip busier still,
i.e. it would make tiering look *better*, not worse.

### Why `_scale_graph` changed

It shrank flops, state bytes, input/output shapes and `op.m`, and re-derived
`experts_touched`, but left `op.meta["bytes"]` alone. That field is what
`_collective_cost` actually charges, and for both the TP all-reduce and the MoE
dispatch/combine it is `T * k * d * cb` -- linear in the token count. A
micro-batch was therefore paying the *full-batch* collective, which on an
expert-parallel pool is most of the step, and every micro-batching result read
as "overlap is a 2x loss" regardless of the real trade. Fixing it moved
`pd_a` mb2 from 81% to 87% of baseline throughput.

### Why the tail all-reduce moved

`ar_ffn` exists to undo tensor parallelism. A MoE block running expert-parallel
on a `tp=1` pool produces a complete hidden state, so it needs no all-reduce --
and charging one also invented two extra bridge crossings per layer (4 instead
of 2). Gating on the FFN pool's TP removes both; the `hot_cold` decode step gets
8.9-12.1% faster and the crossing count matches `pd_a` exactly.

## Caveats that must travel with any number from this directory

| # | Caveat |
|---|---|
| 1 | **The routing skew `p=0.9` is hand-filled.** There is no measured routing data behind it, and it decides whether tiering wins or loses. |
| 2 | **`lynxi_hp300_proj` is a projected part, not a vendor spec.** Its 768 MB SRAM and 150 W TDP are the simulator author's construction and set every absolute number. |
| 3 | The `ceil(groups/shards)` form in `_byte_map` is our own modelling choice. |
| 4 | The `a2a_combine` cross-pool attribution is approximate: its `world` is the brain pool's EP degree even when the op is priced on the GPU pool. Under 0.4 ms of effect. |
| 5 | `find_capacity` tops out in these runs (the `req/s` column saturates), so per-arm capacity points are not resolved -- only the fixed-load throughput, p99s and energy are usable. |
| 6 | Arm A's baseline is `pd_a` with everything pinned, not a pure-GPU cluster. |

## Notes

- `capacity_timing.py` monkey-patches `_probe` in-process only, so it can time
  without touching the package.
- These scripts are written plainly and do no argument validation; a bad
  argument raises.
- If an editor or browser is running alongside, long commands sent to a
  background session can starve for CPU. Prefer short foreground runs with `-v`.
