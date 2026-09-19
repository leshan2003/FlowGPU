"""Device zoo.

Every entry carries a ``source`` string and a ``confidence`` tag so results
can be read with the right amount of salt:

``vendor``     published vendor datasheet / spec page
``paper``      peer-reviewed publication
``press``      vendor press release or conference talk
``derived``    computed from published numbers (e.g. per-core = total / cores)
``estimate``   not published; inferred from architecture and comparable parts

Anything tagged ``estimate`` is flagged in reports.  Do not quote those numbers
as vendor specs.

Units follow :mod:`flowgpu.units`: ``GB`` is binary, ``GB/s`` is decimal,
``TFLOPS`` are *dense* (no 2:1 structured-sparsity doubling) unless the note
says otherwise.
"""

from __future__ import annotations

import copy

from .gpu import make_gpu
from .dataflow import make_dataflow

# =========================================================================
# GPUs
# =========================================================================
GPUS = {
    # --------------------------------------------------------------- NVIDIA
    "a100_80gb_sxm": dict(
        vendor="NVIDIA", sms=108, clock="1.41 GHz", process_nm=7, year=2020,
        peak={"fp64": "19.5 TFLOPS", "tf32": "156 TFLOPS", "bf16": "312 TFLOPS",
              "fp16": "312 TFLOPS", "int8": "624 TOPS"},
        memory={
            "regfile": dict(capacity="27 MB", bandwidth="230 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="17.3 MB", bandwidth="19.5 TB/s",
                            latency="23 ns", per_unit_capacity="164 KB"),
            "l2":      dict(capacity="40 MB", bandwidth="5.3 TB/s",
                            latency="200 ns"),
            "hbm":     dict(capacity="80 GB", bandwidth="2039 GB/s",
                            latency="450 ns"),
        },
        tdp="400 W", achievable_peak=0.78,
        links={"scaleup": dict(name="NVLink3", bandwidth="300 GB/s",
                               latency="1.5 us", world=8),
               "scaleout": dict(name="IB HDR", bandwidth="25 GB/s",
                                latency="5 us")},
        source="vendor A100 datasheet", confidence="vendor",
    ),
    "h100_sxm": dict(
        vendor="NVIDIA", sms=132, clock="1.76 GHz", process_nm=5, year=2022,
        peak={"fp64": "67 TFLOPS", "tf32": "495 TFLOPS", "bf16": "989 TFLOPS",
              "fp16": "989 TFLOPS", "fp8": "1979 TFLOPS", "int8": "1979 TOPS"},
        memory={
            "regfile": dict(capacity="33 MB", bandwidth="330 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="30 MB", bandwidth="33 TB/s",
                            latency="25 ns", per_unit_capacity="228 KB"),
            "l2":      dict(capacity="50 MB", bandwidth="8.5 TB/s",
                            latency="230 ns"),
            "hbm":     dict(capacity="80 GB", bandwidth="3350 GB/s",
                            latency="480 ns"),
        },
        tdp="700 W", achievable_peak=0.80,
        links={"scaleup": dict(name="NVLink4", bandwidth="450 GB/s",
                               latency="1.3 us", world=8),
               "scaleout": dict(name="IB NDR", bandwidth="50 GB/s",
                                latency="4 us")},
        source="vendor H100 datasheet", confidence="vendor",
    ),
    "h200_sxm": dict(
        vendor="NVIDIA", sms=132, clock="1.76 GHz", process_nm=5, year=2023,
        peak={"fp64": "67 TFLOPS", "tf32": "495 TFLOPS", "bf16": "989 TFLOPS",
              "fp16": "989 TFLOPS", "fp8": "1979 TFLOPS", "int8": "1979 TOPS"},
        memory={
            "regfile": dict(capacity="33 MB", bandwidth="330 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="30 MB", bandwidth="33 TB/s",
                            latency="25 ns", per_unit_capacity="228 KB"),
            "l2":      dict(capacity="50 MB", bandwidth="8.5 TB/s",
                            latency="230 ns"),
            "hbm":     dict(capacity="141 GB", bandwidth="4800 GB/s",
                            latency="480 ns"),
        },
        tdp="700 W", achievable_peak=0.80,
        links={"scaleup": dict(name="NVLink4", bandwidth="450 GB/s",
                               latency="1.3 us", world=8),
               "scaleout": dict(name="IB NDR", bandwidth="50 GB/s",
                                latency="4 us")},
        source="vendor H200 datasheet", confidence="vendor",
    ),
    "b200_sxm": dict(
        vendor="NVIDIA", sms=148, clock="1.86 GHz", process_nm=4, year=2024,
        peak={"fp64": "40 TFLOPS", "tf32": "1100 TFLOPS",
              "bf16": "2250 TFLOPS", "fp16": "2250 TFLOPS",
              "fp8": "4500 TFLOPS", "int8": "4500 TOPS",
              "fp4": "9000 TFLOPS", "mxfp4": "9000 TFLOPS"},
        memory={
            "regfile": dict(capacity="37 MB", bandwidth="400 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="33 MB", bandwidth="40 TB/s",
                            latency="25 ns", per_unit_capacity="228 KB"),
            "l2":      dict(capacity="126 MB", bandwidth="16 TB/s",
                            latency="250 ns"),
            "hbm":     dict(capacity="192 GB", bandwidth="8000 GB/s",
                            latency="480 ns"),
        },
        tdp="1000 W", achievable_peak=0.78,
        links={"scaleup": dict(name="NVLink5", bandwidth="900 GB/s",
                               latency="1.2 us", world=72),
               "scaleout": dict(name="IB XDR", bandwidth="100 GB/s",
                                latency="4 us")},
        notes="dense rates; NVIDIA markets 2x these with 2:4 sparsity",
        source="vendor Blackwell datasheet", confidence="vendor",
    ),
    "gb200_gpu": dict(
        vendor="NVIDIA", sms=148, clock="1.96 GHz", process_nm=4, year=2024,
        peak={"fp64": "40 TFLOPS", "tf32": "1250 TFLOPS",
              "bf16": "2500 TFLOPS", "fp16": "2500 TFLOPS",
              "fp8": "5000 TFLOPS", "int8": "5000 TOPS",
              "fp4": "10000 TFLOPS", "mxfp4": "10000 TFLOPS"},
        memory={
            "regfile": dict(capacity="37 MB", bandwidth="420 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="33 MB", bandwidth="42 TB/s",
                            latency="25 ns", per_unit_capacity="228 KB"),
            "l2":      dict(capacity="126 MB", bandwidth="17 TB/s",
                            latency="250 ns"),
            "hbm":     dict(capacity="192 GB", bandwidth="8000 GB/s",
                            latency="470 ns"),
        },
        tdp="1200 W", achievable_peak=0.80,
        links={"scaleup": dict(name="NVLink5", bandwidth="900 GB/s",
                               latency="1.0 us", world=72),
               "scaleout": dict(name="IB XDR", bandwidth="100 GB/s",
                                latency="4 us")},
        source="vendor GB200 NVL72 datasheet", confidence="vendor",
    ),
    "l40s": dict(
        vendor="NVIDIA", sms=142, clock="2.52 GHz", process_nm=5, year=2023,
        peak={"tf32": "183 TFLOPS", "bf16": "362 TFLOPS", "fp16": "362 TFLOPS",
              "fp8": "733 TFLOPS", "int8": "733 TOPS"},
        memory={
            "regfile": dict(capacity="36 MB", bandwidth="180 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="18 MB", bandwidth="18 TB/s",
                            latency="25 ns", per_unit_capacity="128 KB"),
            "l2":      dict(capacity="96 MB", bandwidth="3.5 TB/s",
                            latency="240 ns"),
            "hbm":     dict(capacity="48 GB", bandwidth="864 GB/s",
                            latency="500 ns"),
        },
        tdp="350 W", achievable_peak=0.70,
        links={"scaleout": dict(name="PCIe5", bandwidth="64 GB/s",
                                latency="8 us")},
        notes="GDDR6 not HBM; listed under 'hbm' level for uniformity",
        source="vendor L40S datasheet", confidence="vendor",
    ),
    # --------------------------------------------------------------- Jetson
    "jetson_agx_orin_64": dict(
        vendor="NVIDIA", sms=16, clock="1.3 GHz", process_nm=8, year=2022,
        peak={"fp32": "5.3 TFLOPS", "bf16": "85 TFLOPS", "fp16": "85 TFLOPS",
              "int8": "170 TOPS"},
        memory={
            "regfile": dict(capacity="4 MB", bandwidth="20 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="2.6 MB", bandwidth="2 TB/s",
                            latency="25 ns", per_unit_capacity="164 KB"),
            "l2":      dict(capacity="4 MB", bandwidth="0.8 TB/s",
                            latency="220 ns"),
            "hbm":     dict(capacity="64 GB", bandwidth="204.8 GB/s",
                            latency="600 ns"),
        },
        tdp="60 W", achievable_peak=0.62,
        links={"scaleout": dict(name="10GbE", bandwidth="1.25 GB/s",
                                latency="30 us")},
        notes="LPDDR5 unified memory; 275 INT8 TOPS figure is 2:4-sparse",
        source="vendor Jetson AGX Orin datasheet", confidence="vendor",
    ),
    "jetson_agx_thor": dict(
        vendor="NVIDIA", sms=20, clock="1.57 GHz", process_nm=4, year=2025,
        peak={"fp32": "10 TFLOPS", "bf16": "250 TFLOPS", "fp16": "250 TFLOPS",
              "fp8": "500 TFLOPS", "fp4": "1000 TFLOPS"},
        memory={
            "regfile": dict(capacity="5 MB", bandwidth="30 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="4.5 MB", bandwidth="3 TB/s",
                            latency="25 ns", per_unit_capacity="228 KB"),
            "l2":      dict(capacity="12 MB", bandwidth="1.5 TB/s",
                            latency="230 ns"),
            "hbm":     dict(capacity="128 GB", bandwidth="273 GB/s",
                            latency="580 ns"),
        },
        tdp="130 W", achievable_peak=0.65,
        links={"scaleout": dict(name="100GbE", bandwidth="12.5 GB/s",
                                latency="20 us")},
        notes="2070 FP4 TFLOPS vendor figure is sparse; dense halved here",
        source="vendor Jetson AGX Thor announcement", confidence="press",
    ),
    "jetson_orin_nx_16": dict(
        vendor="NVIDIA", sms=8, clock="0.92 GHz", process_nm=8, year=2023,
        peak={"fp32": "1.88 TFLOPS", "bf16": "25 TFLOPS", "fp16": "25 TFLOPS",
              "int8": "50 TOPS"},
        memory={
            "regfile": dict(capacity="2 MB", bandwidth="8 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="1.3 MB", bandwidth="1 TB/s",
                            latency="25 ns", per_unit_capacity="164 KB"),
            "l2":      dict(capacity="2 MB", bandwidth="0.4 TB/s",
                            latency="220 ns"),
            "hbm":     dict(capacity="16 GB", bandwidth="102.4 GB/s",
                            latency="620 ns"),
        },
        tdp="25 W", achievable_peak=0.58,
        links={"scaleout": dict(name="1GbE", bandwidth="0.125 GB/s",
                                latency="60 us")},
        source="vendor Jetson Orin NX datasheet", confidence="vendor",
    ),
    # ------------------------------------------------------------ Huawei
    "ascend_910b": dict(
        vendor="Huawei", sms=32, clock="1.6 GHz", process_nm=7, year=2023,
        peak={"fp32": "94 TFLOPS", "bf16": "376 TFLOPS", "fp16": "376 TFLOPS",
              "int8": "752 TOPS"},
        memory={
            "regfile": dict(capacity="8 MB", bandwidth="100 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="32 MB", bandwidth="12 TB/s",
                            latency="30 ns", per_unit_capacity="1 MB"),
            "l2":      dict(capacity="192 MB", bandwidth="4 TB/s",
                            latency="250 ns"),
            "hbm":     dict(capacity="64 GB", bandwidth="1600 GB/s",
                            latency="500 ns"),
        },
        tdp="400 W", achievable_peak=0.62,
        links={"scaleup": dict(name="HCCS", bandwidth="196 GB/s",
                               latency="2.5 us", world=8),
               "scaleout": dict(name="RoCE 200G", bandwidth="25 GB/s",
                                latency="8 us")},
        notes="'SM' count = AI Core count; L2 is the unified on-chip buffer",
        source="press + third-party measurement", confidence="press",
    ),
    "ascend_910c": dict(
        vendor="Huawei", sms=64, clock="1.6 GHz", process_nm=7, year=2025,
        peak={"fp32": "188 TFLOPS", "bf16": "752 TFLOPS", "fp16": "752 TFLOPS",
              "int8": "1504 TOPS"},
        memory={
            "regfile": dict(capacity="16 MB", bandwidth="200 TB/s",
                            latency="1 ns", per_unit_capacity="256 KB"),
            "smem":    dict(capacity="64 MB", bandwidth="24 TB/s",
                            latency="30 ns", per_unit_capacity="1 MB"),
            "l2":      dict(capacity="384 MB", bandwidth="8 TB/s",
                            latency="250 ns"),
            "hbm":     dict(capacity="128 GB", bandwidth="3200 GB/s",
                            latency="500 ns"),
        },
        tdp="600 W", achievable_peak=0.60,
        links={"scaleup": dict(name="HCCS", bandwidth="392 GB/s",
                               latency="2.5 us", world=8),
               "scaleout": dict(name="RoCE 400G", bandwidth="50 GB/s",
                                latency="8 us")},
        notes="dual-die 910B; public specs are approximate",
        source="press", confidence="estimate",
    ),
    # ------------------------------------------------- Iluvatar 天数智芯
    "iluvatar_bi_v100": dict(
        vendor="Iluvatar CoreX 天数智芯", sms=64, clock="1.5 GHz",
        process_nm=7, year=2021,
        peak={"fp32": "37 TFLOPS", "bf16": "147 TFLOPS", "fp16": "147 TFLOPS",
              "int8": "295 TOPS"},
        memory={
            "regfile": dict(capacity="8 MB", bandwidth="90 TB/s",
                            latency="1 ns", per_unit_capacity="128 KB"),
            "smem":    dict(capacity="8 MB", bandwidth="9 TB/s",
                            latency="30 ns", per_unit_capacity="128 KB"),
            "l2":      dict(capacity="32 MB", bandwidth="2.4 TB/s",
                            latency="300 ns"),
            "hbm":     dict(capacity="32 GB", bandwidth="1200 GB/s",
                            latency="550 ns"),
        },
        tdp="250 W", achievable_peak=0.52, vector_frac=0.04,
        l2_act_hit=0.78, jitter_sigma=0.10,
        links={"scaleup": dict(name="PCIe4 p2p", bandwidth="64 GB/s",
                               latency="6 us", world=8),
               "scaleout": dict(name="RoCE 200G", bandwidth="25 GB/s",
                                latency="9 us")},
        notes="天垓100 / BI-V100 training card. 24 Bn transistors, 2.5D CoWoS. "
              "This is the '较早型号' used in the reference deployment. "
              "SM count and cache sizes are estimates.",
        source="vendor spec page + third-party comparison",
        confidence="derived",
    ),
    "iluvatar_mr_v100": dict(
        vendor="Iluvatar CoreX 天数智芯", sms=64, clock="1.4 GHz",
        process_nm=7, year=2022,
        peak={"fp32": "24 TFLOPS", "bf16": "96 TFLOPS", "fp16": "96 TFLOPS",
              "int8": "384 TOPS"},
        memory={
            "regfile": dict(capacity="8 MB", bandwidth="80 TB/s",
                            latency="1 ns", per_unit_capacity="128 KB"),
            "smem":    dict(capacity="8 MB", bandwidth="8 TB/s",
                            latency="30 ns", per_unit_capacity="128 KB"),
            "l2":      dict(capacity="32 MB", bandwidth="2.0 TB/s",
                            latency="300 ns"),
            "hbm":     dict(capacity="32 GB", bandwidth="800 GB/s",
                            latency="550 ns"),
        },
        tdp="250 W", achievable_peak=0.50, vector_frac=0.04,
        l2_act_hit=0.78, jitter_sigma=0.10,
        links={"scaleup": dict(name="PCIe4 p2p", bandwidth="64 GB/s",
                               latency="6 us", world=8),
               "scaleout": dict(name="RoCE 200G", bandwidth="25 GB/s",
                                latency="9 us")},
        notes="智铠100 / MR-V100 inference card",
        source="vendor spec page", confidence="derived",
    ),
    # ------------------------------------------------ Moore Threads 摩尔线程
    "mthreads_s4000": dict(
        vendor="Moore Threads 摩尔线程", sms=128, clock="1.6 GHz",
        process_nm=7, year=2023,
        peak={"fp64": "0.8 TFLOPS", "fp32": "25 TFLOPS", "tf32": "50 TFLOPS",
              "bf16": "100 TFLOPS", "fp16": "100 TFLOPS", "int8": "200 TOPS"},
        memory={
            "regfile": dict(capacity="16 MB", bandwidth="60 TB/s",
                            latency="1 ns", per_unit_capacity="128 KB"),
            "smem":    dict(capacity="8 MB", bandwidth="6 TB/s",
                            latency="35 ns", per_unit_capacity="64 KB"),
            "l2":      dict(capacity="32 MB", bandwidth="1.8 TB/s",
                            latency="320 ns"),
            "hbm":     dict(capacity="48 GB", bandwidth="768 GB/s",
                            latency="650 ns"),
        },
        tdp="450 W", achievable_peak=0.45, vector_frac=0.035,
        l2_act_hit=0.75, jitter_sigma=0.12,
        links={"scaleup": dict(name="MTLink", bandwidth="120 GB/s",
                               latency="4 us", world=8),
               "scaleout": dict(name="RoCE 200G", bandwidth="25 GB/s",
                                latency="9 us")},
        notes="GDDR6 (not HBM). MUSA arch. Cache sizes estimated.",
        source="vendor product page", confidence="derived",
    ),
    "mthreads_s5000": dict(
        vendor="Moore Threads 摩尔线程", sms=160, clock="1.8 GHz",
        process_nm=6, year=2026,
        peak={"fp32": "50 TFLOPS", "bf16": "250 TFLOPS", "fp16": "250 TFLOPS",
              "fp8": "500 TFLOPS", "int8": "500 TOPS"},
        memory={
            "regfile": dict(capacity="20 MB", bandwidth="90 TB/s",
                            latency="1 ns", per_unit_capacity="128 KB"),
            "smem":    dict(capacity="10 MB", bandwidth="9 TB/s",
                            latency="35 ns", per_unit_capacity="64 KB"),
            "l2":      dict(capacity="64 MB", bandwidth="3 TB/s",
                            latency="300 ns"),
            "hbm":     dict(capacity="80 GB", bandwidth="1600 GB/s",
                            latency="550 ns"),
        },
        tdp="500 W", achievable_peak=0.50, vector_frac=0.04,
        jitter_sigma=0.10,
        links={"scaleup": dict(name="MTLink2", bandwidth="392 GB/s",
                               latency="3 us", world=8),
               "scaleout": dict(name="RoCE 400G", bandwidth="50 GB/s",
                                latency="8 us")},
        notes="announced specs; 1000 TFLOPS FP8 vendor figure is sparse",
        source="press", confidence="estimate",
    ),
    # ------------------------------------------------------- MetaX 沐曦
    "metax_c500": dict(
        vendor="MetaX 沐曦", sms=104, clock="1.5 GHz", process_nm=7, year=2023,
        peak={"fp32": "35 TFLOPS", "tf32": "140 TFLOPS", "bf16": "280 TFLOPS",
              "fp16": "280 TFLOPS", "int8": "560 TOPS"},
        memory={
            "regfile": dict(capacity="13 MB", bandwidth="70 TB/s",
                            latency="1 ns", per_unit_capacity="128 KB"),
            "smem":    dict(capacity="13 MB", bandwidth="8 TB/s",
                            latency="30 ns", per_unit_capacity="128 KB"),
            "l2":      dict(capacity="48 MB", bandwidth="2.5 TB/s",
                            latency="300 ns"),
            "hbm":     dict(capacity="64 GB", bandwidth="1800 GB/s",
                            latency="520 ns"),
        },
        tdp="450 W", achievable_peak=0.52, vector_frac=0.04,
        l2_act_hit=0.78, jitter_sigma=0.10,
        links={"scaleup": dict(name="MetaXLink", bandwidth="224 GB/s",
                               latency="3.5 us", world=8),
               "scaleout": dict(name="RoCE 200G", bandwidth="25 GB/s",
                                latency="9 us")},
        notes="曦云 C500. No FP8 support in current stack. Caches estimated.",
        source="vendor page + third-party survey", confidence="derived",
    ),
    "amd_mi300x": dict(
        vendor="AMD", sms=304, clock="2.1 GHz", process_nm=5, year=2023,
        peak={"fp64": "81 TFLOPS", "fp32": "163 TFLOPS", "tf32": "654 TFLOPS",
              "bf16": "1307 TFLOPS", "fp16": "1307 TFLOPS",
              "fp8": "2615 TFLOPS", "int8": "2615 TOPS"},
        memory={
            "regfile": dict(capacity="153 MB", bandwidth="500 TB/s",
                            latency="1 ns", per_unit_capacity="512 KB"),
            "smem":    dict(capacity="20 MB", bandwidth="40 TB/s",
                            latency="30 ns", per_unit_capacity="64 KB"),
            "l2":      dict(capacity="256 MB", bandwidth="17 TB/s",
                            latency="280 ns"),
            "hbm":     dict(capacity="192 GB", bandwidth="5300 GB/s",
                            latency="520 ns"),
        },
        tdp="750 W", achievable_peak=0.62,
        links={"scaleup": dict(name="Infinity Fabric", bandwidth="448 GB/s",
                               latency="1.8 us", world=8),
               "scaleout": dict(name="IB NDR", bandwidth="50 GB/s",
                                latency="4 us")},
        notes="'SM' = CU count; 256 MB L2 is the Infinity Cache",
        source="vendor MI300X datasheet", confidence="vendor",
    ),
}

# =========================================================================
# many-core dataflow / brain-inspired chips
# =========================================================================
DATAFLOW = {
    # ------------------------------------------------------------- Groq
    "groq_lpu_v1": dict(
        vendor="Groq", arch="dataflow", process_nm=14, year=2020,
        cores=320, clock="900 MHz",
        sram_per_core="0.719 MB",          # 230 MB / 320 slices
        sram_bw_per_core="250 GB/s",       # 80 TB/s / 320
        peak={"fp16": "188 TFLOPS", "int8": "750 TOPS",
              "bf16": "188 TFLOPS"},
        pe_rows=320, pe_cols=320,
        memory={"sram": dict(capacity="230 MB", bandwidth="80 TB/s",
                             latency="1.5 ns", per_unit_capacity="0.719 MB")},
        noc=dict(topology="crossbar", bisection_bandwidth="80 TB/s",
                 hop_latency="1 ns", multicast=True),
        has_on_package_dram=False,
        inter_chip_bw="200 GB/s", inter_chip_latency="0.6 us",
        inter_board_bw="100 GB/s", inter_board_latency="2 us",
        chips_per_board=8,
        tdp="275 W", achievable_peak=0.90, jitter_sigma=0.001,
        kernel_launch_overhead="50 ns",
        supported_dtypes=("int8", "fp16", "bf16"),
        notes="TSP: functionally-sliced, software-scheduled, no caches, fully "
              "deterministic. 320x320 fused dot-product unit, 5120 vector "
              "ALUs. NO DRAM AT ALL -- a 70B model needs hundreds of chips. "
              "Core count here models the 320 SIMD slices.",
        source="Groq ISCA'20 TSP paper + vendor", confidence="paper",
        extra=dict(sram_weight_frac=0.82, deterministic=True),
    ),
    "groq_lpu_v2": dict(
        vendor="Groq", arch="dataflow", process_nm=4, year=2025,
        cores=320, clock="1.4 GHz",
        sram_per_core="1.6 MB", sram_bw_per_core="600 GB/s",
        peak={"fp16": "375 TFLOPS", "int8": "1500 TOPS",
              "bf16": "375 TFLOPS", "fp8": "1500 TFLOPS"},
        pe_rows=320, pe_cols=320,
        memory={"sram": dict(capacity="512 MB", bandwidth="192 TB/s",
                             latency="1.2 ns")},
        noc=dict(topology="crossbar", bisection_bandwidth="192 TB/s",
                 hop_latency="0.8 ns"),
        inter_chip_bw="400 GB/s", inter_chip_latency="0.5 us",
        inter_board_bw="200 GB/s", inter_board_latency="1.5 us",
        chips_per_board=8,
        tdp="350 W", achievable_peak=0.90, jitter_sigma=0.001,
        kernel_launch_overhead="40 ns",
        notes="Samsung 4nm LPU v2. Capacity/BW figures are projections.",
        source="press", confidence="estimate",
        extra=dict(sram_weight_frac=0.82, deterministic=True),
    ),
    # --------------------------------------------------------- Graphcore
    "ipu_gc200": dict(
        vendor="Graphcore", arch="dataflow", process_nm=7, year=2020,
        cores=1472, clock="1.33 GHz",
        sram_per_core="624 KB",
        sram_bw_per_core="32.3 GB/s",      # 47.5 TB/s / 1472
        peak={"fp16": "250 TFLOPS", "bf16": "250 TFLOPS",
              "fp32": "62.5 TFLOPS"},
        pe_rows=16, pe_cols=16,
        memory={
            "sram": dict(capacity="900 MB", bandwidth="47.5 TB/s",
                         latency="6 ns", per_unit_capacity="624 KB"),
            "dram": dict(capacity="64 GB", bandwidth="20 GB/s",
                         latency="2 us"),
        },
        noc=dict(topology="crossbar", bisection_bandwidth="8 TB/s",
                 hop_latency="15 ns", multicast=False),
        has_on_package_dram=False,
        inter_chip_bw="320 GB/s", inter_chip_latency="1 us",
        inter_board_bw="128 GB/s", inter_board_latency="3 us",
        chips_per_board=4,
        tdp="300 W", achievable_peak=0.55, jitter_sigma=0.002,
        kernel_launch_overhead="100 ns",
        notes="MK2 IPU: 1472 tiles x 6 threads, BSP execution model. "
              "'dram' is IPU-M2000 Streaming Memory (DDR4) -- note the very "
              "low 20 GB/s: spilling weights off-chip is catastrophic here.",
        source="vendor GC200 datasheet + Graphcore papers", confidence="vendor",
        extra=dict(sram_weight_frac=0.80, bsp_sync_cost="2 us"),
    ),
    "ipu_bow": dict(
        vendor="Graphcore", arch="dataflow", process_nm=7, year=2022,
        cores=1472, clock="1.85 GHz",
        sram_per_core="624 KB", sram_bw_per_core="44.8 GB/s",
        peak={"fp16": "350 TFLOPS", "bf16": "350 TFLOPS",
              "fp32": "87.5 TFLOPS"},
        pe_rows=16, pe_cols=16,
        memory={
            "sram": dict(capacity="900 MB", bandwidth="65 TB/s",
                         latency="5 ns", per_unit_capacity="624 KB"),
            "dram": dict(capacity="128 GB", bandwidth="20 GB/s",
                         latency="2 us"),
        },
        noc=dict(topology="crossbar", bisection_bandwidth="11 TB/s",
                 hop_latency="12 ns", multicast=False),
        inter_chip_bw="320 GB/s", inter_chip_latency="1 us",
        inter_board_bw="128 GB/s", inter_board_latency="3 us",
        chips_per_board=4,
        tdp="350 W", achievable_peak=0.55, jitter_sigma=0.002,
        notes="Bow: wafer-on-wafer power delivery, +40% clock over GC200",
        source="vendor Bow datasheet", confidence="vendor",
        extra=dict(sram_weight_frac=0.80),
    ),
    # ------------------------------------------------------ Lynxi 灵汐
    "lynxi_ka200": dict(
        vendor="Lynxi 灵汐科技", arch="brain", process_nm=28, year=2021,
        cores=30, clock="1 GHz",
        sram_per_core="1.6 MB",            # ESTIMATE: 48 MB total
        sram_bw_per_core="80 GB/s",        # ESTIMATE
        peak={"fp16": "16 TFLOPS", "bf16": "16 TFLOPS", "int8": "32 TOPS",
              "spike": "64 TOPS"},
        pe_rows=32, pe_cols=32,
        memory={
            "sram": dict(capacity="48 MB", bandwidth="2.4 TB/s",
                         latency="3 ns", per_unit_capacity="1.6 MB"),
            "dram": dict(capacity="16 GB", bandwidth="68 GB/s",
                         latency="900 ns"),
        },
        noc=dict(topology="mesh2d", link_bandwidth="32 GB/s",
                 bisection_bandwidth="350 GB/s", hop_latency="8 ns",
                 multicast=True),
        has_on_package_dram=False,
        inter_chip_bw="50 GB/s", inter_chip_latency="1.5 us",
        inter_board_bw="25 GB/s", inter_board_latency="5 us",
        chips_per_board=2,
        tdp="15 W", achievable_peak=0.70, jitter_sigma=0.01,
        kernel_launch_overhead="200 ns",
        spiking=False, spike_rate=0.05,
        supported_dtypes=("int8", "fp16", "bf16", "spike"),
        notes="领启 KA200: 30 brain-computing cores, heterogeneous ANN+SNN, "
              "compute-in-memory, any-to-any core comms with multicast. "
              "250k neurons / 25M synapses dense. 12-15 W. "
              "SRAM capacity/bandwidth and NoC figures are ESTIMATES -- "
              "Lynxi publishes neuron/synapse counts, not SRAM bytes. "
              "LPDDR4 is on the HP300/HM100 module, not on package.",
        source="vendor press + Science Robotics Tianjic lineage",
        confidence="estimate",
        extra=dict(sram_weight_frac=0.85, deterministic=True),
    ),
    "lynxi_hp300_proj": dict(
        vendor="Lynxi 灵汐科技", arch="brain", process_nm=12, year=2025,
        cores=256, clock="1.2 GHz",
        sram_per_core="3 MB", sram_bw_per_core="200 GB/s",
        peak={"fp16": "128 TFLOPS", "bf16": "128 TFLOPS", "int8": "256 TOPS",
              "fp8": "256 TFLOPS"},
        pe_rows=64, pe_cols=64,
        memory={
            "sram": dict(capacity="768 MB", bandwidth="51.2 TB/s",
                         latency="3 ns", per_unit_capacity="3 MB"),
            "dram": dict(capacity="64 GB", bandwidth="200 GB/s",
                         latency="800 ns"),
        },
        noc=dict(topology="mesh2d", link_bandwidth="64 GB/s",
                 bisection_bandwidth="2 TB/s", hop_latency="6 ns",
                 multicast=True),
        has_on_package_dram=True,
        inter_chip_bw="200 GB/s", inter_chip_latency="1 us",
        inter_board_bw="100 GB/s", inter_board_latency="3 us",
        chips_per_board=8,
        tdp="150 W", achievable_peak=0.80, jitter_sigma=0.003,
        kernel_launch_overhead="150 ns",
        supported_dtypes=("int8", "fp8", "fp16", "bf16"),
        notes="PROJECTED next-generation Lynxi part sized to make the "
              "GPU+brain-chip rack in the 2026 中国算力大会 announcement "
              "physically consistent (KA200 at 16 TFLOPS / 48 MB cannot host "
              "a DeepSeek-class MoE FFN). ALL NUMBERS ARE THE SIMULATOR "
              "AUTHOR'S CONSTRUCTION, NOT VENDOR SPECS. 768 MB at 12 nm is "
              "~478 mm2 of SRAM array alone (60% of a reticle) -- buildable "
              "but aggressive; 150 W reflects that. Sweep sram_per_core and "
              "tdp with `flowgpu sweep` to test how sensitive the "
              "conclusions are to these two numbers.",
        source="projection", confidence="estimate",
        extra=dict(sram_weight_frac=0.85, deterministic=True, projected=True),
    ),
    # ----------------------------------------------------- Tsinghua Tianjic
    "tianjic": dict(
        vendor="Tsinghua 清华", arch="brain", process_nm=28, year=2019,
        cores=156, clock="300 MHz",
        sram_per_core="40 KB", sram_bw_per_core="1.2 GB/s",
        peak={"int8": "1.28 TOPS", "fp16": "0.32 TFLOPS",
              "spike": "0.65 TOPS"},
        pe_rows=16, pe_cols=16,
        memory={"sram": dict(capacity="6.1 MB", bandwidth="190 GB/s",
                             latency="4 ns", per_unit_capacity="40 KB")},
        noc=dict(topology="mesh2d", link_bandwidth="1.2 GB/s",
                 bisection_bandwidth="15 GB/s", hop_latency="10 ns"),
        inter_chip_bw="4 GB/s", inter_chip_latency="3 us",
        inter_board_bw="2 GB/s", inter_board_latency="10 us",
        chips_per_board=4,
        tdp="1.2 W", achievable_peak=0.60, jitter_sigma=0.005,
        spiking=False, supported_dtypes=("int8", "spike"),
        notes="Nature 2019 cover chip. 156 FCores, 40k neurons, 10M synapses. "
              "1.28 TOPS/W ANN, 649 GSOPS/W SNN. Research part -- far too "
              "small for LLM inference; included for completeness.",
        source="Pei et al., Nature 572 (2019)", confidence="paper",
        extra=dict(sram_weight_frac=0.9),
    ),
    "tianjicx": dict(
        vendor="Tsinghua 清华", arch="brain", process_nm=28, year=2022,
        cores=16, clock="500 MHz",
        sram_per_core="80 KB", sram_bw_per_core="4 GB/s",
        peak={"int8": "0.5 TOPS", "spike": "0.25 TOPS"},
        pe_rows=16, pe_cols=16,
        memory={"sram": dict(capacity="1.25 MB", bandwidth="64 GB/s",
                             latency="4 ns")},
        noc=dict(topology="mesh2d", link_bandwidth="4 GB/s",
                 bisection_bandwidth="16 GB/s", hop_latency="10 ns"),
        inter_chip_bw="2 GB/s", inter_chip_latency="3 us",
        inter_board_bw="1 GB/s", inter_board_latency="10 us",
        tdp="0.9 W", achievable_peak=0.55, jitter_sigma=0.004,
        spiking=True, spike_rate=0.05, supported_dtypes=("int8", "spike"),
        notes="TianjicX: spatiotemporal-elastic neuromorphic chip for robots "
              "(Science Robotics 2022). Optimised for latency-critical "
              "multi-task robotics, not datacentre LLM serving.",
        source="Ma et al., Science Robotics 7 (2022)", confidence="paper",
        extra=dict(sram_weight_frac=0.9),
    ),
    "tianjic2_proj": dict(
        vendor="Tsinghua 清华", arch="brain", process_nm=12, year=2024,
        cores=320, clock="1 GHz",
        sram_per_core="256 KB", sram_bw_per_core="64 GB/s",
        peak={"int8": "128 TOPS", "fp16": "64 TFLOPS", "spike": "256 TOPS"},
        pe_rows=32, pe_cols=32,
        memory={
            "sram": dict(capacity="80 MB", bandwidth="20.5 TB/s",
                         latency="3 ns"),
            "dram": dict(capacity="16 GB", bandwidth="120 GB/s",
                         latency="850 ns"),
        },
        noc=dict(topology="mesh2d", link_bandwidth="32 GB/s",
                 bisection_bandwidth="1 TB/s", hop_latency="7 ns"),
        inter_chip_bw="100 GB/s", inter_chip_latency="1.2 us",
        inter_board_bw="50 GB/s", inter_board_latency="4 us",
        chips_per_board=8,
        tdp="35 W", achievable_peak=0.75, jitter_sigma=0.004,
        notes="PROJECTED Tianjic-2 class part. Not a vendor spec.",
        source="projection", confidence="estimate",
        extra=dict(sram_weight_frac=0.85, projected=True),
    ),
    "darwin3": dict(
        vendor="Zhejiang Univ 浙大", arch="snn", process_nm=22, year=2023,
        cores=240, clock="333 MHz",
        sram_per_core="64 KB", sram_bw_per_core="1.3 GB/s",
        peak={"spike": "1.2 TOPS", "int8": "0.6 TOPS"},
        pe_rows=8, pe_cols=8,
        memory={"sram": dict(capacity="15 MB", bandwidth="320 GB/s",
                             latency="4 ns")},
        noc=dict(topology="mesh2d", link_bandwidth="1.3 GB/s",
                 bisection_bandwidth="20 GB/s", hop_latency="12 ns"),
        inter_chip_bw="2 GB/s", inter_chip_latency="4 us",
        inter_board_bw="1 GB/s", inter_board_latency="12 us",
        tdp="2 W", achievable_peak=0.50, jitter_sigma=0.004,
        spiking=True, spike_rate=0.03, supported_dtypes=("int8", "spike"),
        notes="Darwin3: 2.35M neurons, 100M+ synapses, custom SNN ISA. "
              "SRAM/NoC figures partly estimated.",
        source="Ma et al. 2023 + press", confidence="estimate",
        extra=dict(sram_weight_frac=0.9),
    ),
    "truenorth": dict(
        vendor="IBM", arch="snn", process_nm=28, year=2014,
        cores=4096, clock="1 MHz",
        sram_per_core="1.3 KB", sram_bw_per_core="1.3 MB/s",
        peak={"spike": "0.058 TOPS"},
        pe_rows=256, pe_cols=1,
        memory={"sram": dict(capacity="5.4 MB", bandwidth="5.4 GB/s",
                             latency="10 ns")},
        noc=dict(topology="mesh2d", link_bandwidth="0.03 GB/s",
                 bisection_bandwidth="0.4 GB/s", hop_latency="100 ns"),
        inter_chip_bw="0.05 GB/s", inter_chip_latency="20 us",
        inter_board_bw="0.02 GB/s", inter_board_latency="50 us",
        tdp="0.07 W", achievable_peak=0.40, jitter_sigma=0.002,
        spiking=True, spike_rate=0.02, supported_dtypes=("spike",),
        notes="1M neurons / 256M synapses at 70 mW. Binary spikes only, "
              "no multiply. CANNOT run a transformer -- the simulator will "
              "report an unsupported-dtype fallback if you try.",
        source="Merolla et al., Science 345 (2014)", confidence="paper",
        extra=dict(sram_weight_frac=0.95),
    ),
    "loihi2": dict(
        vendor="Intel", arch="snn", process_nm=7, year=2021,
        cores=128, clock="1.05 GHz",
        sram_per_core="192 KB", sram_bw_per_core="8 GB/s",
        peak={"spike": "1.0 TOPS", "int8": "0.5 TOPS"},
        pe_rows=8, pe_cols=8,
        memory={"sram": dict(capacity="24 MB", bandwidth="1 TB/s",
                             latency="3 ns")},
        noc=dict(topology="mesh2d", link_bandwidth="8 GB/s",
                 bisection_bandwidth="90 GB/s", hop_latency="6 ns"),
        inter_chip_bw="6 GB/s", inter_chip_latency="2 us",
        inter_board_bw="3 GB/s", inter_board_latency="8 us",
        chips_per_board=8,
        tdp="1 W", achievable_peak=0.45, jitter_sigma=0.003,
        spiking=True, spike_rate=0.04, supported_dtypes=("int8", "spike"),
        notes="Intel 4 process, 2.1 Bn transistors, 1M neurons, 120M "
              "synapses, programmable neuron models. Throughput figures "
              "estimated from published spike rates.",
        source="Intel Loihi 2 tech brief", confidence="estimate",
        extra=dict(sram_weight_frac=0.9),
    ),
    "spinnaker2": dict(
        vendor="TU Dresden / SpiNNcloud", arch="snn", process_nm=22, year=2024,
        cores=152, clock="300 MHz",
        sram_per_core="128 KB", sram_bw_per_core="2.4 GB/s",
        peak={"int8": "4.6 TOPS", "spike": "9.2 TOPS", "fp16": "0.3 TFLOPS"},
        pe_rows=8, pe_cols=8,
        memory={
            "sram": dict(capacity="19 MB", bandwidth="365 GB/s",
                         latency="5 ns"),
            "dram": dict(capacity="2 GB", bandwidth="10 GB/s",
                         latency="1.2 us"),
        },
        noc=dict(topology="mesh2d", link_bandwidth="2.4 GB/s",
                 bisection_bandwidth="30 GB/s", hop_latency="15 ns"),
        inter_chip_bw="6 GB/s", inter_chip_latency="2.5 us",
        inter_board_bw="3 GB/s", inter_board_latency="9 us",
        chips_per_board=48,
        tdp="3 W", achievable_peak=0.35, jitter_sigma=0.02,
        spiking=True, spike_rate=0.05, supported_dtypes=("int8", "spike"),
        notes="152 ARM Cortex-M4F PEs + ML accelerators, 22nm FDSOI. "
              "Software-defined neurons -- flexible but low density.",
        source="Mayr/Hoeppner SpiNNaker2 papers", confidence="paper",
        extra=dict(sram_weight_frac=0.85),
    ),
    "brainscales2": dict(
        vendor="Heidelberg", arch="snn", process_nm=65, year=2020,
        cores=4, clock="125 MHz",
        sram_per_core="32 KB", sram_bw_per_core="0.5 GB/s",
        peak={"spike": "0.01 TOPS", "int8": "0.005 TOPS"},
        pe_rows=8, pe_cols=8,
        memory={"sram": dict(capacity="128 KB", bandwidth="2 GB/s",
                             latency="8 ns")},
        noc=dict(topology="crossbar", bisection_bandwidth="2 GB/s",
                 hop_latency="4 ns"),
        inter_chip_bw="1 GB/s", inter_chip_latency="3 us",
        inter_board_bw="0.5 GB/s", inter_board_latency="10 us",
        tdp="1 W", achievable_peak=0.30, jitter_sigma=0.05,
        spiking=True, spike_rate=0.1, supported_dtypes=("int8", "spike"),
        notes="Analog/mixed-signal accelerated neuromorphic (1000x biological "
              "real time). 512 neurons, 130k synapses. Fundamentally not an "
              "LLM inference part; included for architectural comparison.",
        source="Pehle et al., Front. Neurosci. (2022)", confidence="paper",
        extra=dict(sram_weight_frac=0.9),
    ),
    # ----------------------------------------------------------- Cerebras
    "cerebras_wse3": dict(
        vendor="Cerebras", arch="wafer", process_nm=5, year=2024,
        cores=900000, clock="1.1 GHz",
        sram_per_core="48 KB", sram_bw_per_core="23 GB/s",
        peak={"fp16": "62500 TFLOPS", "bf16": "62500 TFLOPS",
              "fp32": "15600 TFLOPS", "fp8": "125000 TFLOPS"},
        pe_rows=4, pe_cols=4,
        memory={
            "sram": dict(capacity="44 GB", bandwidth="21000 TB/s",
                         latency="1 ns", per_unit_capacity="48 KB"),
            "dram": dict(capacity="1200 GB", bandwidth="150 GB/s",
                         latency="3 us"),
        },
        noc=dict(topology="mesh2d", link_bandwidth="4 TB/s",
                 bisection_bandwidth="26750 TB/s", hop_latency="0.9 ns",
                 multicast=True),
        inter_chip_bw="1200 GB/s", inter_chip_latency="2 us",
        inter_board_bw="600 GB/s", inter_board_latency="6 us",
        chips_per_board=1,
        tdp="23000 W", achievable_peak=0.35, jitter_sigma=0.002,
        kernel_launch_overhead="100 ns",
        notes="Whole-wafer engine. Vendor headline 125 PFLOPS is FP16 with "
              "sparsity; dense halved here. 'dram' models MemoryX weight "
              "streaming. TDP is the full CS-3 system including cooling.",
        source="Cerebras WSE-3 datasheet", confidence="vendor",
        extra=dict(sram_weight_frac=0.75, deterministic=True),
    ),
    # ---------------------------------------------------------- SambaNova
    "sambanova_sn40l": dict(
        vendor="SambaNova", arch="dataflow", process_nm=5, year=2023,
        cores=1040, clock="1.25 GHz",
        sram_per_core="512 KB", sram_bw_per_core="14 GB/s",
        peak={"bf16": "638 TFLOPS", "fp16": "638 TFLOPS",
              "int8": "1276 TOPS", "fp8": "1276 TFLOPS"},
        pe_rows=32, pe_cols=32,
        memory={
            "sram": dict(capacity="520 MB", bandwidth="15 TB/s",
                         latency="4 ns", per_unit_capacity="512 KB"),
            "hbm":  dict(capacity="64 GB", bandwidth="2000 GB/s",
                         latency="500 ns"),
            "dram": dict(capacity="1536 GB", bandwidth="200 GB/s",
                         latency="1.5 us"),
        },
        noc=dict(topology="mesh2d", link_bandwidth="128 GB/s",
                 bisection_bandwidth="4 TB/s", hop_latency="5 ns"),
        has_on_package_dram=True,
        inter_chip_bw="400 GB/s", inter_chip_latency="1 us",
        inter_board_bw="200 GB/s", inter_board_latency="3 us",
        chips_per_board=8,
        tdp="700 W", achievable_peak=0.70, jitter_sigma=0.004,
        notes="RDU with three-tier memory (SRAM/HBM/DDR). 'cores' models the "
              "PCU+PMU array. The DDR tier is what lets one node hold a "
              "trillion-parameter model -- at 200 GB/s.",
        source="SambaNova SN40L paper + vendor", confidence="paper",
        extra=dict(sram_weight_frac=0.8),
    ),
    # --------------------------------------------------------------- Dojo
    "tesla_dojo_d1": dict(
        vendor="Tesla", arch="dataflow", process_nm=7, year=2021,
        cores=354, clock="2 GHz",
        sram_per_core="1.25 MB", sram_bw_per_core="113 GB/s",
        peak={"bf16": "362 TFLOPS", "fp16": "362 TFLOPS",
              "fp32": "22.6 TFLOPS", "fp8": "362 TFLOPS"},
        pe_rows=8, pe_cols=8,
        memory={"sram": dict(capacity="442 MB", bandwidth="40 TB/s",
                             latency="3 ns", per_unit_capacity="1.25 MB")},
        noc=dict(topology="mesh2d", link_bandwidth="512 GB/s",
                 bisection_bandwidth="10 TB/s", hop_latency="2 ns"),
        has_on_package_dram=False,
        inter_chip_bw="4000 GB/s", inter_chip_latency="0.3 us",
        inter_board_bw="900 GB/s", inter_board_latency="1.5 us",
        chips_per_board=25,
        tdp="400 W", achievable_peak=0.65, jitter_sigma=0.003,
        notes="D1 die: 354 training nodes, 1.25 MB SRAM each, no DRAM. "
              "25 D1 dies per training tile. CFP8 native.",
        source="Tesla AI Day / Hot Chips 34", confidence="press",
        extra=dict(sram_weight_frac=0.8),
    ),
}


# =========================================================================
# lookup
# =========================================================================
def list_devices() -> dict:
    return {"gpu": sorted(GPUS), "dataflow": sorted(DATAFLOW)}


def get_spec(name: str) -> tuple:
    if name in GPUS:
        return "gpu", copy.deepcopy(GPUS[name])
    if name in DATAFLOW:
        return "dataflow", copy.deepcopy(DATAFLOW[name])
    raise KeyError(f"unknown device {name!r}. Known: "
                   f"{sorted(GPUS) + sorted(DATAFLOW)}")


def make_device(name: str, overrides: dict | None = None):
    """Instantiate a device by registry name, with optional spec overrides."""
    kind, spec = get_spec(name)
    if overrides:
        spec = _deep_update(spec, overrides)
    dev = make_gpu(name, spec) if kind == "gpu" else make_dataflow(name, spec)
    dev.extra.setdefault("source", spec.get("source", ""))
    dev.extra.setdefault("confidence", spec.get("confidence", ""))
    return dev


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base
