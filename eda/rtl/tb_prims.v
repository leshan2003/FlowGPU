//=====================================================================
// Testbench that drives each primitive with *realistic* activity and
// dumps SAIF for PrimeTime PX.
//
// The stimulus is a Gaussian approximation of per-tensor-quantised
// transformer activations (sigma ~ 24 LSB of +-127), generated internally by
// the CLT sum below.
//
// Worth knowing before drawing conclusions from this: for *independent*
// samples that distribution toggles every bit at ~0.50, exactly like uniform
// random data.  Quantisation alone does not reduce switching -- the sign bit
// and its extension dominate the high bits and are as random as the values.
// A reduction only appears once consecutive samples are correlated.  See
// eda/scripts/activity.py for the numbers.
//
// The SAIF system tasks below are guarded by `SAIF because they hang under
// some VCS builds; when they are compiled out, PrimeTime uses the derived
// activity set in scripts/ptpx.tcl instead.
//=====================================================================
`timescale 1ns / 1ps

module tb_prims;

    localparam CLK_NS   = 2.0;      // 500 MHz -- slow enough for lsi_10k
    localparam N_CYCLES = 4096;

    reg clk = 0, rst_n = 0;
    always #(CLK_NS/2.0) clk = ~clk;

    // ---------------- stimulus generation ----------------------------
    integer seed = 32'hC0FFEE;

    // Approximate a clipped Gaussian with the sum of 4 uniforms (CLT),
    // scaled so sigma ~= 24 LSB out of +-127.  This reproduces the
    // low-toggle-rate high bits of real quantised activations.
    function signed [7:0] gauss8;
        input integer dummy;
        integer s, i;
        begin
            s = 0;
            for (i = 0; i < 4; i = i + 1)
                s = s + ($random(seed) % 64) - 32;
            s = (s * 3) / 4;
            if (s > 127)  s = 127;
            if (s < -127) s = -127;
            gauss8 = s[7:0];
        end
    endfunction

    function signed [15:0] gauss16;
        input integer dummy;
        integer s, i;
        begin
            s = 0;
            for (i = 0; i < 4; i = i + 1)
                s = s + ($random(seed) % 4096) - 2048;
            s = (s * 3) / 4;
            gauss16 = s[15:0];
        end
    endfunction

    // ---------------- DUT 1: INT8 MAC array --------------------------
    reg  [63:0]  act8;
    reg  [511:0] w8;
    reg          mac_en = 0, mac_wl = 0;
    wire [191:0] mac_out;

    mac_array #(.ROWS(8), .COLS(8), .AW(8), .WW(8), .ACCW(24)) u_mac8 (
        .clk(clk), .rst_n(rst_n), .w_load(mac_wl), .en(mac_en),
        .act_in(act8), .w_in(w8), .acc_out(mac_out));

    // ---------------- DUT 2: 16-bit MAC array ------------------------
    reg  [127:0]  act16;
    reg  [1023:0] w16;
    reg           mac16_en = 0, mac16_wl = 0;
    wire [319:0]  mac16_out;

    mac_array16 #(.ROWS(8), .COLS(8)) u_mac16 (
        .clk(clk), .rst_n(rst_n), .w_load(mac16_wl), .en(mac16_en),
        .act_in(act16), .w_in(w16), .acc_out(mac16_out));

    // ---------------- DUT 3: SRAM tile -------------------------------
    reg         sr_we = 0, sr_re = 0;
    reg  [5:0]  sr_wa = 0, sr_ra = 0;
    reg  [63:0] sr_wd = 0;
    wire [63:0] sr_rd;

    sram_tile #(.DEPTH(64), .DW(64), .AWID(6)) u_sram (
        .clk(clk), .rst_n(rst_n), .we(sr_we), .re(sr_re),
        .waddr(sr_wa), .raddr(sr_ra), .wdata(sr_wd), .rdata(sr_rd));

    // ---------------- DUT 4: NoC router ------------------------------
    reg  [4:0]   nv = 0;
    reg  [319:0] nf = 0;
    reg  [14:0]  nd = 0;
    wire [4:0]   ov;
    wire [319:0] of;

    noc_router #(.FW(64), .PORTS(5)) u_noc (
        .clk(clk), .rst_n(rst_n),
        .in_valid(nv), .in_flit(nf), .in_dest(nd),
        .out_valid(ov), .out_flit(of));

    // ---------------- DUT 5: integrated PE ---------------------------
    reg        pe_en = 0, pe_wl = 0, pe_we = 0;
    reg [5:0]  pe_ra = 0, pe_rw = 0, pe_wa = 0;
    reg [63:0] pe_wd = 0;
    wire [191:0] pe_out;

    pe_tile #(.ROWS(8), .COLS(8), .DEPTH(64)) u_pe (
        .clk(clk), .rst_n(rst_n), .en(pe_en), .w_load(pe_wl),
        .raddr_a(pe_ra), .raddr_w(pe_rw), .we(pe_we),
        .waddr(pe_wa), .wdata(pe_wd), .acc_out(pe_out));

    // ---------------- drive ------------------------------------------
    integer i, j, phase;

    task drive_mac8;
        begin
            for (j = 0; j < 8; j = j + 1)
                act8[j*8 +: 8] = gauss8(0);
            mac_en = 1'b1;
        end
    endtask

    task drive_mac16;
        begin
            for (j = 0; j < 8; j = j + 1)
                act16[j*16 +: 16] = gauss16(0);
            mac16_en = 1'b1;
        end
    endtask

    task drive_sram;
        begin
            sr_re = 1'b1;
            sr_ra = $random(seed);
            sr_we = (($random(seed) % 8) == 0);   // 1 write per 8 reads
            sr_wa = $random(seed);
            for (j = 0; j < 8; j = j + 1)
                sr_wd[j*8 +: 8] = gauss8(0);
        end
    endtask

    task drive_noc;
        begin
            for (j = 0; j < 5; j = j + 1) begin
                nv[j]            = (($random(seed) % 4) != 0);  // 75% load
                nd[j*3 +: 3]     = ($random(seed) % 5);
                for (i = 0; i < 8; i = i + 1)
                    nf[j*64 + i*8 +: 8] = gauss8(0);
            end
        end
    endtask

    task drive_pe;
        begin
            pe_en = 1'b1;
            pe_ra = $random(seed);
            pe_rw = $random(seed);
            pe_we = (($random(seed) % 16) == 0);
            pe_wa = $random(seed);
            for (j = 0; j < 8; j = j + 1)
                pe_wd[j*8 +: 8] = gauss8(0);
        end
    endtask

    initial begin
`ifdef SAIF
        $set_toggle_region(tb_prims);
`endif
        rst_n = 0;
        // load weights once (weight-stationary)
        for (i = 0; i < 64; i = i + 1) w8[i*8 +: 8]   = gauss8(0);
        for (i = 0; i < 64; i = i + 1) w16[i*16 +: 16] = gauss16(0);
        repeat (8) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        mac_wl = 1; mac16_wl = 1;
        @(posedge clk);
        mac_wl = 0; mac16_wl = 0;

`ifdef SAIF
        $toggle_start;
`endif
        for (i = 0; i < N_CYCLES; i = i + 1) begin
            @(posedge clk);
            drive_mac8;
            drive_mac16;
            drive_sram;
            drive_noc;
            drive_pe;
        end
        @(posedge clk);
`ifdef SAIF
        $toggle_stop;
        $toggle_report("flowgpu.saif", 1.0e-9, tb_prims);
`endif
        $display("FLOWGPU_TB_DONE cycles=%0d clk_ns=%f", N_CYCLES, CLK_NS);
        $finish;
    end
endmodule
