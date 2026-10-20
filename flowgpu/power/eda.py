"""EDA-backed energy calibration via Synopsys DC + PrimeTime PX.

What this buys you
------------------
The simulator's verdict on "is the brain-inspired chip more energy efficient"
hinges on one number: how much cheaper is reading a weight byte from local
SRAM than from HBM, relative to the cost of the MAC that consumes it.  Taking
that from a table is fine; measuring it on synthesised logic with *real*
switching activity is better, and it is checkable.

What it does
------------
1. Pushes ``eda/rtl`` and ``eda/scripts`` to the EDA host.
2. Runs ``run_all.sh`` there: DC synthesis of five primitives and PrimeTime PX
   power analysis of each, with switching activity either measured from a
   gate-level VCS SAIF or derived analytically (see ``eda/scripts/activity.py``
   and the comment block in ``ptpx.tcl``).  Note that for *independent*
   quantised activations every bit toggles at ~0.5, i.e. the same as uniform
   random -- the often-repeated claim that quantisation reduces switching only
   holds once temporal correlation is accounted for.
3. Pulls back ``energy_raw.json`` and converts raw watts into the two ratios
   the power model consumes:

   * ``ratio_sram_byte_per_flop`` -- joules per byte read from a local
     scratchpad, divided by joules per FLOP in the MAC datapath
   * ``ratio_noc_byte_per_flop``  -- joules per byte per router hop, same
     denominator
   * ``relative_flop_energy``     -- measured int8 vs 16-bit MAC energy

Honesty about the library
-------------------------
The only Liberty library readable on the available host is the Synopsys-
shipped ``lsi_10k``, a ~0.5 um educational library.  Absolute joules from it
mean nothing for a 5 nm part.  Ratios between structurally different datapaths
measured under identical conditions do carry over, imperfectly but usefully;
``flowgpu.power.model`` uses only the ratios and anchors the absolute scale to
published per-node data and each device's TDP.  A second correction is applied
for the fact that the ``sram_tile`` array is synthesised from flip-flops (no
memory compiler is available): a flop array costs roughly ``FLOP_SRAM_FACTOR``
times a 6T SRAM macro per bit accessed, and that factor is applied explicitly
and visibly rather than being buried.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EDA_DIR = os.path.join(ROOT, "eda")
RESULTS = os.path.join(EDA_DIR, "results")

# Where the EDA tools live.  There is no useful default for this -- a
# Synopsys installation is site-specific -- so it comes from the environment
# or from `--host`/`--user`/`--key`.  Set:
#
#     export FLOWGPU_EDA_HOST=eda.example.edu
#     export FLOWGPU_EDA_USER=$USER
#     export FLOWGPU_EDA_KEY=~/.ssh/id_rsa_eda        # optional
#     export FLOWGPU_EDA_REMOTE_DIR=/scratch/$USER/flowgpu_eda   # optional
#
# Everything the flow needs on the far side is `dc_shell` and `pt_shell` on
# PATH plus any readable Liberty library; see eda/scripts/syn.tcl.
DEFAULT_HOST = dict(
    user=os.environ.get("FLOWGPU_EDA_USER", os.environ.get("USER", "")),
    host=os.environ.get("FLOWGPU_EDA_HOST", ""),
    key=os.path.expanduser(os.environ.get("FLOWGPU_EDA_KEY", "")),
    remote_dir=os.environ.get(
        "FLOWGPU_EDA_REMOTE_DIR",
        f"/home/{os.environ.get('FLOWGPU_EDA_USER', os.environ.get('USER', 'user'))}"
        "/flowgpu_eda"),
)

# A synthesised flip-flop array burns far more energy per accessed bit than a
# compiled 6T SRAM macro: no bitline sharing, full clock load on every bit,
# and no sense amps.  Published comparisons put the gap at 8-15x for arrays
# of a few KB.  Applied explicitly so the correction is auditable.
FLOP_SRAM_FACTOR = 10.0

# The testbench drives every DUT every cycle; these are the useful work
# quantities per cycle used to turn watts into joules-per-unit.
PER_CYCLE = {
    "mac_array":   dict(flops=8 * 8 * 2, bytes=0),      # 8x8 MACs = 128 FLOP
    "mac_array16": dict(flops=8 * 8 * 2, bytes=0),
    "sram_tile":   dict(flops=0, bytes=8 + 1),          # 64-bit read + 1/8 wr
    "noc_router":  dict(flops=0, bytes=5 * 8 * 0.75),   # 5 ports x 8 B x 75%
    "pe_tile":     dict(flops=8 * 8 * 2, bytes=16),
}

# The synthesised scratchpad is 64 entries x 64 bits = 512 B.  That is a
# register file, not a multi-megabyte SRAM bank, and read energy per byte is
# strongly size-dependent (long bitlines, sense amps, decode).  The measured
# ratio is therefore anchored at this size and re-scaled to each modelled
# memory level using the size curve in :mod:`flowgpu.power.tech`.  Without
# that step the flow would claim an SRAM byte costs about as much as a MAC,
# which is true for a 512 B register file and badly wrong for a 2 MB bank.
MEASURED_ARRAY_BYTES = 64 * 8


# =========================================================================
def _ssh_base(cfg):
    cmd = ["ssh"]
    if cfg.get("key"):
        cmd += ["-i", cfg["key"]]
    return cmd + ["-o", "StrictHostKeyChecking=accept-new",
                  "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
                  f"{cfg['user']}@{cfg['host']}"]


def _scp_base(cfg):
    cmd = ["scp"]
    if cfg.get("key"):
        cmd += ["-i", cfg["key"]]
    return cmd + ["-o", "StrictHostKeyChecking=accept-new", "-q"]


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def push(cfg, verbose=True) -> bool:
    base = _ssh_base(cfg)
    rd = cfg["remote_dir"]
    r = _run(base + [f"bash -lc {shlex.quote(f'mkdir -p {rd}/rtl {rd}/scripts {rd}/work')}"])
    if r.returncode != 0:
        print(f"[eda] mkdir failed: {r.stderr.strip()}", file=sys.stderr)
        return False
    for sub in ("rtl", "scripts"):
        src = os.path.join(EDA_DIR, sub)
        r = _run(_scp_base(cfg) + ["-r"] +
                 [os.path.join(src, f) for f in sorted(os.listdir(src))] +
                 [f"{cfg['user']}@{cfg['host']}:{rd}/{sub}/"])
        if r.returncode != 0:
            print(f"[eda] scp {sub} failed: {r.stderr.strip()}", file=sys.stderr)
            return False
    _run(base + [f"chmod +x {rd}/scripts/run_all.sh"])
    if verbose:
        print(f"[eda] pushed RTL + scripts to {cfg['host']}:{rd}")
    return True


def execute(cfg, timeout=7200, verbose=True, skip_syn=False) -> str:
    base = _ssh_base(cfg)
    rd = cfg["remote_dir"]
    env = "SKIP_SYN=1 " if skip_syn else ""
    cmd = f"{env}bash {rd}/scripts/run_all.sh"
    if verbose:
        print(f"[eda] running: {cmd}  (this takes several minutes)")
    r = _run(base + [f"bash -lc {shlex.quote(cmd)}"], timeout=timeout)
    out = r.stdout + "\n" + r.stderr
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "run_all.log"), "w") as f:
        f.write(out)
    return out


def pull(cfg, verbose=True) -> dict | None:
    rd = cfg["remote_dir"]
    os.makedirs(RESULTS, exist_ok=True)
    for remote, local in (("work/energy_raw.json", "energy_raw.json"),):
        r = _run(_scp_base(cfg) + [
            f"{cfg['user']}@{cfg['host']}:{rd}/{remote}",
            os.path.join(RESULTS, local)])
        if r.returncode != 0:
            print(f"[eda] pull {remote} failed: {r.stderr.strip()}",
                  file=sys.stderr)
            return None
    # reports are small and useful for auditing
    for sub in ("reports", "power"):
        _run(_scp_base(cfg) + [
            "-r", f"{cfg['user']}@{cfg['host']}:{rd}/work/{sub}", RESULTS])
    with open(os.path.join(RESULTS, "energy_raw.json")) as f:
        return json.load(f)


# =========================================================================
def analyse(raw: dict, verbose=True) -> dict:
    """Convert raw PrimeTime watts into the ratios the power model wants."""
    pw = raw.get("power", {})
    if not pw:
        return {}

    def energy_per_cycle(name):
        row = pw.get(name)
        if not row:
            return None
        clk_ns = float(row.get("clk_ns") or 20.0)
        # dynamic only: leakage is handled separately as static power
        dyn = float(row.get("dynamic_w") or
                    (float(row.get("total_w", 0)) -
                     float(row.get("leakage_w", 0))))
        return dyn * clk_ns * 1e-9      # joules per cycle

    e_mac8 = energy_per_cycle("mac_array")
    e_mac16 = energy_per_cycle("mac_array16")
    e_sram = energy_per_cycle("sram_tile")
    e_noc = energy_per_cycle("noc_router")
    e_pe = energy_per_cycle("pe_tile")
    if not e_mac8 or not e_sram:
        return {}

    j_per_flop8 = e_mac8 / PER_CYCLE["mac_array"]["flops"]
    j_per_flop16 = (e_mac16 / PER_CYCLE["mac_array16"]["flops"]
                    if e_mac16 else None)
    j_per_byte_sram_raw = e_sram / PER_CYCLE["sram_tile"]["bytes"]
    j_per_byte_sram = j_per_byte_sram_raw / FLOP_SRAM_FACTOR
    j_per_byte_noc = e_noc / PER_CYCLE["noc_router"]["bytes"] if e_noc else None

    out = dict(
        run_id=raw.get("run_id", "lsi10k"),
        library="lsi_10k (Synopsys-shipped educational library)",
        saif=bool(raw.get("saif")),
        activity=(next(iter(pw.values())).get("activity", "unknown")
                  if pw else "unknown"),
        array_bytes=MEASURED_ARRAY_BYTES,
        dtype="int8",
        ratio_sram_byte_per_flop=j_per_byte_sram / j_per_flop8,
        ratio_sram_byte_per_flop_uncorrected=(j_per_byte_sram_raw
                                              / j_per_flop8),
        flop_sram_factor=FLOP_SRAM_FACTOR,
        measured=dict(
            j_per_flop_int8=j_per_flop8,
            j_per_flop_16b=j_per_flop16,
            j_per_byte_sram_flopram=j_per_byte_sram_raw,
            j_per_byte_sram_corrected=j_per_byte_sram,
            j_per_byte_noc_hop=j_per_byte_noc,
            j_per_cycle=dict(mac8=e_mac8, mac16=e_mac16, sram=e_sram,
                             noc=e_noc, pe=e_pe),
        ),
        note=("Absolute joules are from a ~0.5 um library and are NOT "
              "transferable; only the ratios are used, and the absolute "
              "scale is re-anchored to each device's TDP."),
    )
    if j_per_byte_noc:
        out["ratio_noc_byte_per_flop"] = j_per_byte_noc / j_per_flop8
    if j_per_flop16:
        out["relative_flop_energy"] = {
            "int8": 1.0,
            "fp8": j_per_flop16 / j_per_flop8 * 0.42,
            "bf16": j_per_flop16 / j_per_flop8,
            "fp16": j_per_flop16 / j_per_flop8,
            "fp32": j_per_flop16 / j_per_flop8 * 3.1,
        }
    if verbose:
        print_summary(out)
    return out


def print_summary(a: dict):  # pragma: no cover
    m = a.get("measured", {})
    print("\n=== EDA-measured energy ratios ===")
    print(f"  library         : {a.get('library')}")
    print(f"  activity        : {a.get('activity', '?')}"
          + ("  (measured)" if a.get("saif") else
             "  (derived from quantised-activation statistics, not a "
             "tool default)"))
    print(f"  J/FLOP  int8    : {m.get('j_per_flop_int8', 0):.4g}")
    print(f"  J/FLOP  16-bit  : {m.get('j_per_flop_16b') or 0:.4g}")
    print(f"  J/byte  SRAM    : {m.get('j_per_byte_sram_corrected', 0):.4g} "
          f"(flop-array {m.get('j_per_byte_sram_flopram', 0):.4g} / "
          f"{a.get('flop_sram_factor')})")
    print(f"  J/byte  NoC hop : {m.get('j_per_byte_noc_hop') or 0:.4g}")
    print(f"  --> SRAM byte / MAC FLOP  = "
          f"{a.get('ratio_sram_byte_per_flop', 0):.3f}   "
          f"(at {a.get('array_bytes', 0)} B array; scaled by array size "
          f"when applied)")
    if "ratio_noc_byte_per_flop" in a:
        print(f"  --> NoC byte  / MAC FLOP  = "
              f"{a['ratio_noc_byte_per_flop']:.3f}")
    if "relative_flop_energy" in a:
        print(f"  --> 16-bit / int8 MAC     = "
              f"{a['relative_flop_energy']['bf16']:.2f}x")


def write_db(analysis: dict, path: str | None = None) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = path or os.path.join(RESULTS, "energy_db.json")
    db = {"default": analysis, "dataflow": analysis, "gpu": analysis}
    with open(path, "w") as f:
        json.dump(db, f, indent=2)
    return path


# =========================================================================
def run_flow(host: str | None = None, user: str | None = None,
             key: str | None = None, dry_run: bool = False,
             local_only: bool = False, out: str | None = None,
             skip_syn: bool = False) -> int:
    cfg = dict(DEFAULT_HOST)
    if host:
        cfg["host"] = host
    if user:
        cfg["user"] = user
        cfg["remote_dir"] = os.environ.get("FLOWGPU_EDA_REMOTE_DIR",
                                           f"/home/{user}/flowgpu_eda")
    if key:
        cfg["key"] = os.path.expanduser(key)
    if not local_only and not dry_run and not cfg["host"]:
        print("[eda] no EDA host configured.\n"
              "      Set FLOWGPU_EDA_HOST (and FLOWGPU_EDA_USER), or pass\n"
              "      --host/--user.  The host needs dc_shell and pt_shell on\n"
              "      PATH and a readable Liberty library.\n"
              "      Use --local-only to just generate the RTL and scripts.",
              file=sys.stderr)
        return 2
    print(f"[eda] RTL   : {EDA_DIR}/rtl")
    print(f"[eda] host  : {cfg['user']}@{cfg['host']}:{cfg['remote_dir']}")
    if local_only:
        print("[eda] local-only: RTL and TCL scripts are ready to submit")
        return 0
    if dry_run:
        print("[eda] dry run: would push, run run_all.sh, and pull results")
        return 0
    if not push(cfg):
        return 2
    log = execute(cfg, skip_syn=skip_syn)
    tail = "\n".join(log.strip().splitlines()[-25:])
    print(tail)
    if "FLOWGPU_EDA_DONE" not in log:
        print("[eda] flow did not complete; see eda/results/run_all.log",
              file=sys.stderr)
        return 3
    raw = pull(cfg)
    if raw is None:
        return 4
    a = analyse(raw)
    if not a:
        print("[eda] no usable power numbers were produced", file=sys.stderr)
        return 5
    p = write_db(a, out)
    print(f"\n[eda] wrote {p}")
    print("[eda] subsequent simulator runs will pick this up automatically "
          "(power model source becomes 'eda:...')")
    return 0
