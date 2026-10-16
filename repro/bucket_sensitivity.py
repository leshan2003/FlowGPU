"""How much does the decode batch bucket change the answer?

`ServingSimulator.step_cost` memoises decode on a bucket of the batch size
(default 8) and returns the *bucketed* step cost.  A `_scaled()` helper that
was meant to correct the difference is defined but never called, so a decode
step with 12 sequences is priced as 16 and one with 9 is priced as 8.

This measures the size of that error and the cost of removing the bucketing.

Run:  python repro/bucket_sensitivity.py [config] [arm]
"""

import sys
import time
from dataclasses import replace

sys.path.insert(0, ".")

from flowgpu.experiment import Experiment, _resolve_system, load_config
from flowgpu.power import calibrate_system
from flowgpu.sim.serving import ServingSimulator
from flowgpu.system import build_system


def build(cfg_path, arm_name):
    cfg = load_config(cfg_path)
    exp = Experiment(cfg)
    run = [r for r in cfg["runs"] if r.get("name") == arm_name][0]
    model = exp._model(run)
    syscfg = _resolve_system(run["system"], exp.systems, ".")
    syscfg.setdefault("name", arm_name)
    system = build_system(syscfg)
    calibrate_system(system, eda_db=exp.eda_db)
    policy = exp._placement(run, model, system)
    wl = exp._workload(run)
    opt = dict(exp.defaults)
    keep = ("microbatches", "jitter", "collective_overlap",
            "tp_world", "ep_world", "seed")
    opt.update({k: v for k, v in run.items() if k in keep})
    return model, system, policy, wl, opt


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        "configs/experiments/pd_a_reference.yaml"
    arm = sys.argv[2] if len(sys.argv) > 2 else "pd_a"
    model, system, policy, wl, opt = build(cfg_path, arm)
    base = replace(wl, arrival_rate=1.0, n_requests=64)

    print("arm=%s  (arrival 1.0 req/s, 64 requests, jitter on)" % arm)
    print("bucket  wall(s)  tok/s   TPOT_mean  TPOT_p99  att    keys")
    for bb in (1, 2, 4, 8, 16):
        sim = ServingSimulator(model, system, policy, base,
                               batch_bucket=bb, **opt)
        t0 = time.time()
        res = sim.run()
        dt = time.time() - t0
        from flowgpu.sim.serving import SLO
        print("%6d  %7.2f  %7.1f  %8.3f  %8.3f  %5.1f%%  %d"
              % (bb, dt, res.output_throughput,
                 res.tpot_stats()["mean"] * 1e3,
                 res.tpot_stats()["p99"] * 1e3,
                 100 * res.slo_attainment(SLO(ttft=8.0, tpot=0.05)),
                 len(sim._cache)))


if __name__ == "__main__":
    main()
