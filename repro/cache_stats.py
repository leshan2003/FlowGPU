"""Count how many distinct (stage, batch, ctx, chunk) keys a probe builds.

Every cache miss costs a full graph construction (~0.2-0.5 s for a 40-61
layer MoE model) plus execution.  If the hit rate is poor the capacity search
becomes dominated by graph building rather than by the simulation.
"""

import sys
import time
from collections import Counter
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
    n_req = int(sys.argv[3]) if len(sys.argv) > 3 else 192
    rate = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0

    model, system, policy, wl, opt = build(cfg_path, arm)
    wl = replace(wl, arrival_rate=rate, n_requests=n_req)
    sim = ServingSimulator(model, system, policy, wl, **opt)
    t0 = time.time()
    res = sim.run()
    dt = time.time() - t0
    keys = list(sim._cache)
    stages = Counter(k[0] for k in keys)
    print("%s: %.1f s wall=%.1f s att-eligible reqs=%d  cache keys=%d %s"
          % (arm, dt, res.wall_time,
             sum(1 for r in res.requests if r.finish >= 0),
             len(keys), dict(stages)))
    print("  iters: prefill=%d decode=%d" % (res.n_prefill_iters,
                                             res.n_decode_iters))
    print("  distinct prefill chunk values: %d"
          % len(set(k[4] for k in keys if k[0] == "prefill")))
    print("  distinct prefill batches: %d  decode buckets: %d"
          % (len(set(k[1] for k in keys if k[0] == "prefill")),
             len(set(k[1] for k in keys if k[0] == "decode"))))


if __name__ == "__main__":
    main()
