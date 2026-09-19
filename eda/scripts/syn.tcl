#=====================================================================
# Design Compiler synthesis of the FlowGPU energy primitives.
#
# Library: the Synopsys-shipped lsi_10k (the only Liberty library readable
# on this host).  It is an old technology, so ABSOLUTE energies from it are
# not meaningful for a 5 nm GPU.  What *is* meaningful, and what the
# simulator consumes, are the RATIOS between primitives measured on the
# same library with the same activity:
#
#     E(SRAM byte read) / E(MAC)       and     E(NoC byte hop) / E(MAC)
#     E(16-bit MAC)     / E(8-bit MAC)
#
# Those ratios are set by the relative amount of switched capacitance in
# each datapath, which is a structural property of the logic, not of the
# node.  flowgpu.power.model anchors them to published per-node absolutes.
#=====================================================================

# NOTE on the clock period: lsi_10k is a ~0.5 um educational library, so a
# 64-multiplier array needs ~100 ns of combinational depth.  CLK_PERIOD is set
# to 200 ns so every primitive closes timing with the same (low) effort spent
# on optimisation.  This matters for correctness of the *comparison*: if one
# block closes easily and another is restructured aggressively to chase an
# impossible target, their energy ratio reflects DC's effort allocation rather
# than the circuits.  Absolute power scales with 1/period, but the energy per
# operation that the simulator consumes does not.

set RTL_DIR   [expr {[info exists ::env(RTL_DIR)]   ? $::env(RTL_DIR)   : "../rtl"}]
set OUT_DIR   [expr {[info exists ::env(OUT_DIR)]   ? $::env(OUT_DIR)   : "../work"}]
set CLK_PERIOD [expr {[info exists ::env(CLK_PERIOD)] ? $::env(CLK_PERIOD) : 200.0}]

file mkdir $OUT_DIR
file mkdir $OUT_DIR/netlist
file mkdir $OUT_DIR/reports

set SYN_ROOT $::env(SYN_ROOT)
set LIBS "$SYN_ROOT/libraries/syn"

set search_path      [concat $search_path $LIBS $RTL_DIR]
set target_library   "$LIBS/lsi_10k.db"
set link_library     "* $LIBS/lsi_10k.db"
set symbol_library   "$LIBS/lsi_10k.sdb"

define_design_lib WORK -path $OUT_DIR/work_lib

set DESIGNS {mac_array mac_array16 sram_tile noc_router pe_tile}

analyze -format verilog [list $RTL_DIR/flowgpu_prims.v]

foreach top $DESIGNS {
    puts "\n### ===== synthesising $top ====="
    elaborate $top
    current_design $top
    link

    create_clock -name clk -period $CLK_PERIOD [get_ports clk]
    set_clock_uncertainty [expr {$CLK_PERIOD * 0.05}] [get_clocks clk]
    set_input_delay  [expr {$CLK_PERIOD * 0.2}] -clock clk \
        [remove_from_collection [all_inputs] [get_ports clk]]
    set_output_delay [expr {$CLK_PERIOD * 0.2}] -clock clk [all_outputs]
    set_driving_cell -lib_cell ND2 \
        [remove_from_collection [all_inputs] [get_ports clk]]
    set_load 0.1 [all_outputs]
    set_max_area 0

    # Keep the comparison fair: identical effort for every primitive.  Plain
    # `compile` rather than `compile_ultra` -- on a 0.5 um library the
    # ultra-effort datapath transforms spend all their time chasing timing
    # that cannot close, and unequal effort between blocks would corrupt
    # exactly the ratios this flow exists to measure.
    compile -map_effort medium -area_effort medium

    change_names -rules verilog -hierarchy
    write -format verilog -hierarchy -output $OUT_DIR/netlist/$top.v
    write_sdf $OUT_DIR/netlist/$top.sdf
    write_sdc $OUT_DIR/netlist/$top.sdc

    redirect $OUT_DIR/reports/$top.area.rpt   { report_area -hierarchy }
    redirect $OUT_DIR/reports/$top.timing.rpt { report_timing -max_paths 3 }
    redirect $OUT_DIR/reports/$top.power.rpt  { report_power -hier -analysis_effort high }
    redirect $OUT_DIR/reports/$top.qor.rpt    { report_qor }

    # machine-readable summary
    set fh [open $OUT_DIR/reports/$top.summary.txt w]
    puts $fh "design $top"
    puts $fh "area [get_attribute [current_design] area]"
    if {[catch {set slk [get_attribute [get_timing_paths -max_paths 1] slack]}]} {
        set slk "na"
    }
    puts $fh "slack $slk"
    puts $fh "clk_period $CLK_PERIOD"
    close $fh

    remove_design -designs
}

puts "\n### FLOWGPU_SYN_DONE"
exit
