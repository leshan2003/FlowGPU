#=====================================================================
# PrimeTime PX: activity-driven power analysis of each primitive.
#
# Uses the SAIF written by the gate-level VCS run, so switching activity is
# measured, not propagated from a default toggle rate.  That distinction is
# the whole reason for doing this rather than reading numbers off a
# datasheet: the MAC:SRAM:NoC energy ratio depends strongly on activity,
# and transformer activations have far lower high-bit toggle rates than the
# uniform-random default PrimeTime would otherwise assume.
#=====================================================================

set OUT_DIR [expr {[info exists ::env(OUT_DIR)] ? $::env(OUT_DIR) : "../work"}]
set TOP     $::env(PT_TOP)
set SAIF    [expr {[info exists ::env(SAIF)] ? $::env(SAIF) : "$OUT_DIR/sim/$TOP.saif"}]
set CLK_PERIOD [expr {[info exists ::env(CLK_PERIOD)] ? $::env(CLK_PERIOD) : 200.0}]

set SYN_ROOT $::env(SYN_ROOT)
set LIBS "$SYN_ROOT/libraries/syn"
set search_path  [concat $search_path $LIBS]
set link_path    "* $LIBS/lsi_10k.db"

set power_enable_analysis true
# `time_based` needs a VCD/FSDB event file.  With propagated switching
# activity (the derived path below) the correct mode is `averaged`, which
# propagates toggle rates and static probabilities through the netlist and
# integrates against the library's internal-power tables.  Selecting
# time_based without an event file makes report_power fail outright rather
# than degrade, which is how this was caught.
set power_analysis_mode   averaged
set power_enable_multi_rail_analysis false

read_verilog $OUT_DIR/netlist/$TOP.v
current_design $TOP
link_design

create_clock -name clk -period $CLK_PERIOD [get_ports clk]
set_propagated_clock [get_clocks clk]

if {[file exists $OUT_DIR/netlist/$TOP.sdf]} {
    read_sdf $OUT_DIR/netlist/$TOP.sdf
}

# ---------------------------------------------------------------------
# Switching activity.
#
# Preferred: a SAIF measured by the gate-level run.  Where that is not
# available, activity is set from a *derived* figure rather than a guess.
#
# scripts/activity.py computes the per-bit toggle rate of INT8 activations
# drawn from the distribution real per-tensor-quantised transformer
# activations follow (roughly Gaussian, sigma ~ 24 LSB of +-127), as a
# function of the lag-1 correlation between consecutive samples:
#
#   rho    b0..b7 toggle rates                          mean
#   0.00   .50 .50 .50 .50 .50 .50 .50 .50              0.500
#   0.50   .50 .50 .50 .50 .50 .48 .35 .33              0.457
#   0.90   .50 .50 .50 .50 .46 .27 .15 .14              0.377
#   0.98   .50 .50 .50 .43 .24 .12 .07 .06              0.303
#
# Two things worth noting, because the second contradicts a common claim:
#  1. For *independent* samples the toggle rate is 0.50 on every bit --
#     indistinguishable from uniform random.  Quantisation alone does not
#     reduce switching; the sign bit and its extension are what dominate the
#     high bits, and those are as random as the data.
#  2. The reduction only appears with *temporal correlation*, which streaming
#     activations do have along the reduction dimension.
#
# rho = 0.5 is used here as a deliberately conservative middle: it is the
# value that changes the MAC:SRAM ratio least relative to the default, so the
# measured ratios are not an artefact of an optimistic activity assumption.
set ACT_TOGGLE [expr {[info exists ::env(ACT_TOGGLE)] ? $::env(ACT_TOGGLE) : 0.457}]
set ACT_PROB   0.5

if {[file exists $SAIF]} {
    puts "### reading measured activity from $SAIF"
    read_saif $SAIF -strip_path "tb_prims/$::env(PT_INST)"
    set ACT_SOURCE "saif"
    set power_analysis_mode time_based
} else {
    puts "### no SAIF -- using derived activity: toggle=$ACT_TOGGLE prob=$ACT_PROB"
    puts "### (see the comment block above; this is derived, not a default)"
    set_switching_activity -toggle_rate $ACT_TOGGLE \
        -static_probability $ACT_PROB \
        [remove_from_collection [all_inputs] [get_ports clk]]
    set ACT_SOURCE "derived-rho0.5"
}

update_power

file mkdir $OUT_DIR/power
redirect $OUT_DIR/power/$TOP.power.rpt {
    report_power -verbose -nosplit
}
redirect $OUT_DIR/power/$TOP.hier.rpt {
    report_power -hierarchy -levels 3 -nosplit
}

# Machine-readable line the harness parses.  The design-level power
# attributes are not populated in every PrimeTime version/mode, so each is
# wrapped and the report is parsed as a fallback -- an empty attribute used
# to propagate silently into the CSV as an unparseable field.
proc pget {attr} {
    if {[catch {set v [get_attribute [current_design] $attr]}]} { return "" }
    if {$v eq ""} { return "" }
    return $v
}
set tot  [pget total_power]
set dyn  [pget total_dynamic_power]
set leak [pget leakage_power]
set sw   [pget switching_power]
set intp [pget internal_power]
set area [pget area]

if {$tot eq "" || $dyn eq ""} {
    # parse the text report: "Total Power ... = 1.234e-03 W"
    set fh2 [open $OUT_DIR/power/$TOP.power.rpt r]
    set txt [read $fh2]
    close $fh2
    set units 1.0
    if {[regexp {Power Units? *= *(\w+)} $txt -> u]} {
        switch -- $u { mW {set units 1e-3} uW {set units 1e-6} \
                       nW {set units 1e-9} pW {set units 1e-12} }
    }
    foreach {pat var} {
        {Net Switching Power *= *([0-9.eE+-]+)}   sw
        {Cell Internal Power *= *([0-9.eE+-]+)}   intp
        {Cell Leakage Power *= *([0-9.eE+-]+)}    leak
        {Total Dynamic Power *= *([0-9.eE+-]+)}   dyn
        {Total Power *= *([0-9.eE+-]+)}           tot
    } {
        if {[regexp $pat $txt -> v]} { set $var [expr {$v * $units}] }
    }
    foreach v {tot dyn leak sw intp area} {
        if {[set $v] eq ""} { set $v 0.0 }
    }
    if {$tot == 0.0} { set tot [expr {$dyn + $leak}] }
}

set fh [open $OUT_DIR/power/$TOP.csv w]
puts $fh "design,total_w,dynamic_w,switching_w,internal_w,leakage_w,area,clk_ns,activity"
puts $fh "$TOP,$tot,$dyn,$sw,$intp,$leak,$area,$CLK_PERIOD,$ACT_SOURCE"
close $fh

puts "### FLOWGPU_PTPX $TOP total=$tot dyn=$dyn leak=$leak area=$area"
exit
