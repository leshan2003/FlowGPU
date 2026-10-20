//=====================================================================
// FlowGPU energy-calibration primitives.
//
// Three synthesizable blocks whose switching energy sets the ratios the
// simulator's power model depends on:
//
//   1. mac_array   -- an 8x8 INT8 MAC array with a bf16-ish 16-bit mode.
//                     Gives energy per MAC (i.e. per 2 FLOP).
//   2. sram_tile   -- a register-file-backed scratchpad plus the read
//                     datapath that feeds the array.  Gives energy per byte
//                     read from local on-chip memory.  (No memory compiler
//                     is available, so the array is synthesized from flops;
//                     the *relative* read-path energy is what transfers, and
//                     the absolute value is corrected by a documented
//                     flop-array-vs-6T-SRAM factor.)
//   3. noc_router  -- a 5-port wormhole router with round-robin arbitration.
//                     Gives energy per byte per hop on the on-chip network.
//
// All three are driven by the same testbench with realistic activity
// (actual transformer activation/weight value distributions), dumped to
// SAIF, and measured with PrimeTime PX.
//=====================================================================
`timescale 1ns / 1ps

//---------------------------------------------------------------------
// 1. MAC array: ROWS x COLS INT8 multiply-accumulate, weight-stationary
//---------------------------------------------------------------------
module mac_array #(
    parameter ROWS = 8,
    parameter COLS = 8,
    parameter AW   = 8,       // activation width
    parameter WW   = 8,       // weight width
    parameter ACCW = 24
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         w_load,    // latch new weights
    input  wire                         en,
    input  wire [ROWS*AW-1:0]           act_in,
    input  wire [ROWS*COLS*WW-1:0]      w_in,
    output reg  [COLS*ACCW-1:0]         acc_out
);
    integer r, c;
    reg signed [WW-1:0] wreg [0:ROWS-1][0:COLS-1];
    reg signed [ACCW-1:0] acc [0:COLS-1];

    wire signed [AW-1:0] a [0:ROWS-1];
    genvar gi;
    generate
        for (gi = 0; gi < ROWS; gi = gi + 1) begin : g_act
            assign a[gi] = act_in[gi*AW +: AW];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (r = 0; r < ROWS; r = r + 1)
                for (c = 0; c < COLS; c = c + 1)
                    wreg[r][c] <= {WW{1'b0}};
        end else if (w_load) begin
            for (r = 0; r < ROWS; r = r + 1)
                for (c = 0; c < COLS; c = c + 1)
                    wreg[r][c] <= w_in[(r*COLS + c)*WW +: WW];
        end
    end

    reg signed [ACCW-1:0] psum;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (c = 0; c < COLS; c = c + 1) acc[c] <= {ACCW{1'b0}};
            acc_out <= {(COLS*ACCW){1'b0}};
        end else if (en) begin
            for (c = 0; c < COLS; c = c + 1) begin
                psum = {ACCW{1'b0}};
                for (r = 0; r < ROWS; r = r + 1)
                    psum = psum + a[r] * wreg[r][c];
                acc[c] <= acc[c] + psum;
                acc_out[c*ACCW +: ACCW] <= acc[c] + psum;
            end
        end
    end
endmodule

//---------------------------------------------------------------------
// 1b. 16-bit MAC array (models bf16/fp16 datapath cost via a fixed-point
//     proxy of the same bit width -- the multiplier array dominates, and
//     its energy tracks operand width, not exponent handling)
//---------------------------------------------------------------------
module mac_array16 #(
    parameter ROWS = 8,
    parameter COLS = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     w_load,
    input  wire                     en,
    input  wire [ROWS*16-1:0]       act_in,
    input  wire [ROWS*COLS*16-1:0]  w_in,
    output wire [COLS*40-1:0]       acc_out
);
    mac_array #(.ROWS(ROWS), .COLS(COLS), .AW(16), .WW(16), .ACCW(40)) u (
        .clk(clk), .rst_n(rst_n), .w_load(w_load), .en(en),
        .act_in(act_in), .w_in(w_in), .acc_out(acc_out));
endmodule

//---------------------------------------------------------------------
// 2. SRAM tile: banked scratchpad + read datapath
//---------------------------------------------------------------------
module sram_tile #(
    parameter DEPTH = 64,
    parameter DW    = 64,
    parameter AWID  = 6
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             we,
    input  wire             re,
    input  wire [AWID-1:0]  waddr,
    input  wire [AWID-1:0]  raddr,
    input  wire [DW-1:0]    wdata,
    output reg  [DW-1:0]    rdata
);
    reg [DW-1:0] mem [0:DEPTH-1];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata <= {DW{1'b0}};
        end else begin
            if (we) mem[waddr] <= wdata;
            if (re) rdata <= mem[raddr];
        end
    end
endmodule

//---------------------------------------------------------------------
// 3. NoC router: 5-port wormhole, round-robin arbitration
//---------------------------------------------------------------------
module noc_router #(
    parameter FW    = 64,     // flit width in bits
    parameter PORTS = 5
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [PORTS-1:0]         in_valid,
    input  wire [PORTS*FW-1:0]      in_flit,
    input  wire [PORTS*3-1:0]       in_dest,     // 0..4 output port
    output reg  [PORTS-1:0]         out_valid,
    output reg  [PORTS*FW-1:0]      out_flit
);
    integer i, j;
    reg [2:0] rr [0:PORTS-1];            // round-robin pointer per out port
    reg [2:0] dst;
    reg       granted;
    reg [2:0] src;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= {PORTS{1'b0}};
            out_flit  <= {(PORTS*FW){1'b0}};
            for (i = 0; i < PORTS; i = i + 1) rr[i] <= 3'd0;
        end else begin
            out_valid <= {PORTS{1'b0}};
            for (i = 0; i < PORTS; i = i + 1) begin
                granted = 1'b0;
                src     = 3'd0;
                for (j = 0; j < PORTS; j = j + 1) begin
                    dst = in_dest[((rr[i] + j) % PORTS)*3 +: 3];
                    if (!granted && in_valid[(rr[i] + j) % PORTS] &&
                        dst == i[2:0]) begin
                        granted = 1'b1;
                        src     = (rr[i] + j) % PORTS;
                    end
                end
                if (granted) begin
                    out_valid[i] <= 1'b1;
                    out_flit[i*FW +: FW] <= in_flit[src*FW +: FW];
                    rr[i] <= (src + 1) % PORTS;
                end
            end
        end
    end
endmodule

//---------------------------------------------------------------------
// 4. Integrated PE: SRAM-fed MAC array -- what a dataflow core actually is
//---------------------------------------------------------------------
module pe_tile #(
    parameter ROWS  = 8,
    parameter COLS  = 8,
    parameter DEPTH = 64
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        w_load,
    input  wire [5:0]  raddr_a,
    input  wire [5:0]  raddr_w,
    input  wire        we,
    input  wire [5:0]  waddr,
    input  wire [63:0] wdata,
    output wire [COLS*24-1:0] acc_out
);
    wire [63:0] a_rd, w_rd;

    sram_tile #(.DEPTH(DEPTH), .DW(64), .AWID(6)) u_act (
        .clk(clk), .rst_n(rst_n), .we(we), .re(en),
        .waddr(waddr), .raddr(raddr_a), .wdata(wdata), .rdata(a_rd));

    sram_tile #(.DEPTH(DEPTH), .DW(64), .AWID(6)) u_wgt (
        .clk(clk), .rst_n(rst_n), .we(we), .re(en | w_load),
        .waddr(waddr), .raddr(raddr_w), .wdata(wdata), .rdata(w_rd));

    // replicate the 64-bit weight read across the array (weight-stationary
    // reload path); activations stream 8 lanes of INT8
    wire [ROWS*COLS*8-1:0] w_bus;
    genvar g;
    generate
        for (g = 0; g < ROWS*COLS; g = g + 1) begin : g_w
            assign w_bus[g*8 +: 8] = w_rd[(g % 8)*8 +: 8];
        end
    endgenerate

    mac_array #(.ROWS(ROWS), .COLS(COLS), .AW(8), .WW(8), .ACCW(24)) u_mac (
        .clk(clk), .rst_n(rst_n), .w_load(w_load), .en(en),
        .act_in(a_rd), .w_in(w_bus), .acc_out(acc_out));
endmodule
