#!/bin/bash
# =====================================================================
# FlowGPU energy-calibration flow, executed on the EDA host.
#   RTL -> Design Compiler -> gate-level VCS (SAIF) -> PrimeTime PX
# Emits eda/work/energy_raw.json for flowgpu.power.eda to post-process.
# =====================================================================
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
export RTL_DIR=$ROOT/rtl
export OUT_DIR=$ROOT/work
export CLK_PERIOD=${CLK_PERIOD:-200.0}
# Synopsys install root, resolved here where `command -v` is the shell
# builtin.  Both TCL scripts read it from the environment rather than
# calling `which`, which inside dc_shell/pt_shell is a search-path lookup.
export SYN_ROOT=$(dirname "$(dirname "$(command -v dc_shell)")")

mkdir -p "$OUT_DIR"/{netlist,reports,sim,power,logs}
LOG=$OUT_DIR/logs

echo "=== FlowGPU EDA flow ==="
echo "root=$ROOT clk=${CLK_PERIOD}ns"
command -v dc_shell >/dev/null || { echo "FATAL: dc_shell not on PATH"; exit 1; }
command -v pt_shell >/dev/null || { echo "FATAL: pt_shell not on PATH"; exit 1; }

# ---------------------------------------------------------------- 1. synth
# SKIP_SYN=1 reuses the existing netlists -- synthesis is the slow stage and
# iterating on the power setup does not invalidate it.
echo "--- [1/4] Design Compiler ---"
if [ "${SKIP_SYN:-0}" = "1" ] && [ -f "$OUT_DIR/netlist/pe_tile.v" ]; then
  echo "    SKIP_SYN=1: reusing $(ls "$OUT_DIR/netlist"/*.v | wc -l) netlists"
else
  ( cd "$ROOT/scripts" && dc_shell -f syn.tcl ) > "$LOG/dc.log" 2>&1
  if ! grep -q FLOWGPU_SYN_DONE "$LOG/dc.log"; then
    echo "FATAL: synthesis failed; tail of log:"; tail -40 "$LOG/dc.log"; exit 2
  fi
  echo "    ok: $(ls "$OUT_DIR/netlist"/*.v 2>/dev/null | wc -l) netlists"
fi

# ------------------------------------------------- 2. gate-level sim + SAIF
# The SAIF system tasks ($set_toggle_region/$toggle_report) hang under the
# VCS build on this host, so they are compiled out by default and PrimeTime
# uses the derived activity documented in ptpx.tcl instead.  Set
# FLOWGPU_SAIF=1 to try the measured path on a host where it works.
echo "--- [2/4] gate-level VCS + SAIF ---"
SIM_OK=0
if [ "${FLOWGPU_SAIF:-0}" = "1" ] && command -v vcs >/dev/null; then
  ( cd "$OUT_DIR/sim" && \
    vcs -full64 -sverilog +v2k -debug_access+all \
        +define+GATE_SIM +define+SAIF \
        -timescale=1ns/1ps \
        "$RTL_DIR/tb_prims.v" \
        "$OUT_DIR"/netlist/mac_array.v \
        "$OUT_DIR"/netlist/mac_array16.v \
        "$OUT_DIR"/netlist/sram_tile.v \
        "$OUT_DIR"/netlist/noc_router.v \
        "$OUT_DIR"/netlist/pe_tile.v \
        -v "$SYN_ROOT/packages/gtech/src_ver/gtech_lib.v" \
        -o simv_gate ) > "$LOG/vcs_gate.log" 2>&1
  if [ -x "$OUT_DIR/sim/simv_gate" ]; then
    ( cd "$OUT_DIR/sim" && ./simv_gate ) > "$LOG/sim_gate.log" 2>&1
    [ -f "$OUT_DIR/sim/flowgpu.saif" ] && SIM_OK=1
  fi
  if [ "$SIM_OK" = 0 ]; then
    echo "    gate-level sim unavailable; trying RTL sim for activity"
    ( cd "$OUT_DIR/sim" && \
      vcs -full64 -sverilog +v2k -timescale=1ns/1ps +define+SAIF \
          "$RTL_DIR/flowgpu_prims.v" "$RTL_DIR/tb_prims.v" \
          -o simv_rtl ) > "$LOG/vcs_rtl.log" 2>&1
    if [ -x "$OUT_DIR/sim/simv_rtl" ]; then
      ( cd "$OUT_DIR/sim" && ./simv_rtl ) > "$LOG/sim_rtl.log" 2>&1
      [ -f "$OUT_DIR/sim/flowgpu.saif" ] && SIM_OK=2
    fi
  fi
fi
case "$SIM_OK" in
  1) echo "    ok: gate-level SAIF (highest fidelity)";;
  2) echo "    ok: RTL SAIF (activity is real, gate-level glitching is not)";;
  0) echo "    no SAIF; PrimeTime will use the derived activity "\
          "(toggle 0.457, from quantised-activation statistics)";;
esac

# ---------------------------------------------------------- 3. PrimeTime PX
echo "--- [3/4] PrimeTime PX ---"
declare -A INST=( [mac_array]=u_mac8 [mac_array16]=u_mac16 \
                  [sram_tile]=u_sram [noc_router]=u_noc [pe_tile]=u_pe )
for top in mac_array mac_array16 sram_tile noc_router pe_tile; do
  [ -f "$OUT_DIR/netlist/$top.v" ] || { echo "    skip $top (no netlist)"; continue; }
  PT_TOP=$top PT_INST=${INST[$top]} SAIF=$OUT_DIR/sim/flowgpu.saif \
    pt_shell -f "$ROOT/scripts/ptpx.tcl" > "$LOG/pt_$top.log" 2>&1
  if grep -q "FLOWGPU_PTPX $top" "$LOG/pt_$top.log"; then
    echo "    ok: $(grep -m1 "FLOWGPU_PTPX $top" "$LOG/pt_$top.log")"
  else
    echo "    FAILED: $top -- see $LOG/pt_$top.log"; tail -15 "$LOG/pt_$top.log"
  fi
done

# ------------------------------------------------------------- 4. collect
echo "--- [4/4] collecting ---"
python3 - "$OUT_DIR" <<'PYEOF'
import csv, glob, json, os, re, sys
out = sys.argv[1]
rows = {}
for f in glob.glob(os.path.join(out, "power", "*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            def _f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return 0.0
            rows[r["design"]] = {
                k: (v if k in ("design", "activity") else _f(v))
                for k, v in r.items()}
# fall back to DC's own report_power if PrimeTime produced nothing
if not rows:
    for f in glob.glob(os.path.join(out, "reports", "*.power.rpt")):
        name = os.path.basename(f).split(".")[0]
        txt = open(f, errors="ignore").read()
        m = re.search(r"Total Dynamic Power\s*=\s*([\d.eE+-]+)\s*(\w+)", txt)
        l = re.search(r"Cell Leakage Power\s*=\s*([\d.eE+-]+)\s*(\w+)", txt)
        sc = {"W": 1, "mW": 1e-3, "uW": 1e-6, "nW": 1e-9, "pW": 1e-12}
        if m:
            rows[name] = dict(design=name,
                              dynamic_w=float(m.group(1))*sc.get(m.group(2),1),
                              leakage_w=(float(l.group(1))*sc.get(l.group(2),1)
                                         if l else 0.0),
                              total_w=0.0, switching_w=0.0, internal_w=0.0,
                              area=0.0, clk_ns=float(os.environ.get("CLK_PERIOD", 20)))
            rows[name]["total_w"] = rows[name]["dynamic_w"] + rows[name]["leakage_w"]
            rows[name]["source"] = "dc_report_power"

areas = {}
for f in glob.glob(os.path.join(out, "reports", "*.summary.txt")):
    d = {}
    for line in open(f):
        parts = line.split(None, 1)
        if len(parts) == 2:
            d[parts[0]] = parts[1].strip()
    if "design" in d:
        areas[d["design"]] = d

# lsi_10k carries no leakage tables, so PrimeTime reports leakage as 0 and
# `Total Dynamic Power` is absent; total == dynamic there.  Fill it in rather
# than propagating a zero that would make every energy ratio NaN downstream.
for k, v in rows.items():
    if not v.get("dynamic_w"):
        v["dynamic_w"] = max(0.0, v.get("total_w", 0.0) - v.get("leakage_w", 0.0))

payload = dict(power=rows, summary=areas,
               saif=os.path.exists(os.path.join(out, "sim", "flowgpu.saif")))
with open(os.path.join(out, "energy_raw.json"), "w") as fh:
    json.dump(payload, fh, indent=2)
print("wrote", os.path.join(out, "energy_raw.json"))
for k, v in rows.items():
    print(f"  {k:14s} total={v['total_w']:.6g} W  dyn={v['dynamic_w']:.6g} W "
          f"leak={v['leakage_w']:.6g} W  area={v.get('area',0):.6g}")
PYEOF

echo "=== FLOWGPU_EDA_DONE ==="
