# MasterRTL: Pre-synthesis PPA Estimation Framework

This project implements the pre-synthesis data preparation and **Simple Operator Graph (SOG)** generation engine of **MasterRTL**, corresponding to the research paper:

> **Wenji Fang, Yao Lu, Shang Liu, Qijun Zhang, Ceyu Xu, Lisa Wu Wills, Hongce Zhang, and Zhiyao Xie**,  
> *"Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques,"*  
> **IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)**, Vol. 44, No. 1, January 2025.

---

## Overview

MasterRTL addresses the turnaround time bottleneck of commercial logic synthesis and placement by directly evaluating RTL designs before synthesis. It transforms raw Verilog HDL ($H$) into a bit-level representation called the **Simple Operator Graph (SOG)** ($R$), which canonicalizes arbitrary RTL styles into single-bit registers and five primitive logic operations:

- **Single-bit registers**: `DFF`
- **Five primary logic operators**: `AND`, `OR`, `XOR`, `NOT`, `MUX`

From the SOG, MasterRTL extracts:
1. **Analytical node delays** using a linear resistance-capacitance (RC) fan-out model.
2. **Critical path features** ($P^R_{* \to j}$) using an acyclic Static Timing Analysis (STA) DAG.
3. **Toggle rates** (switching activities) propagated topologically using Boolean logic formulas.
4. **Structural, timing, power, and area proxy statistics** exported to CSV, JSON, and Markdown reports.

---

## Project Structure

```
MasterRTL_Project/
├── data/
│   ├── raw_rtl/         # 41 synthesizable Verilog benchmark designs (.v)
│   ├── yosys_synth/     # Generic synthesis outputs (*.synth.v, *.json ASTs)
│   ├── sog_graphs/      # SOG graph objects (*.pkl), features (*_features.pkl), stats (*_stats.json)
│   └── reports/         # Aggregated stats: sog_statistics.csv, sog_statistics.json, sog_summary.md
├── docs/
│   └── SOG_PIPELINE.md  # Detailed technical documentation and mathematical formulation
├── scripts/
│   ├── run_pipeline.ps1 # One-click execution script for Windows PowerShell
│   ├── run_pipeline.sh  # Execution script for Bash / Linux / macOS
│   └── generate_all_raw_rtl.py # Generator script for the 40+ raw RTL designs
├── src/
│   └── data_prep/
│       ├── yosys_synthesis.py # Generic bit-level synthesis with toolchain auto-discovery
│       ├── sog_converter.py   # SOG builder, STA DAG, delay modeling, toggle propagation
│       └── batch_parse.py     # End-to-end batch processing and statistics aggregation
├── tests/
│   ├── test_sog_converter.py  # Unit tests for SOG converter
│   ├── test_sog_pipeline.py   # Comprehensive pipeline integration tests
│   └── test_yosys_synthesis.py# Unit tests for Yosys synthesis scripts
├── requirements.txt
└── README.md
```

---

## Benchmark RTL Dataset (`data/raw_rtl/`)

The repository includes **41 diverse, synthesizable Verilog designs** spanning multiple circuit scales and application domains:

- **Arithmetic & Mathematical Units**: Ripple-carry adder, carry-lookahead adder (CLA), Kogge-Stone parallel prefix adder, 16-bit subtractor, 4x4 array multiplier, Booth multiplier, sequential multiplier, restoring divider, MAC unit, CORDIC rotational step.
- **Data Path & Bit Manipulation**: 4-bit ALU, 16-bit RISC ALU (with Z/N/C/V flags), 32-bit bitwise unit, 8-bit barrel shifter, 16-to-4 priority encoder, 16-bit parity generator, 32-bit popcount.
- **Sequential & Counters**: 8-bit synchronous counter, up/down loadable counter, Gray-code counter, Johnson/ring counter, 16-bit LFSR pseudo-random generator.
- **Cryptography & Error Detection**: CRC-8 CCITT, Ethernet CRC-32, AES Rijndael S-Box LUT, SHA-256 compression round function.
- **State Machines & Control**: 4-way traffic light controller, multi-coin vending machine, 4-floor elevator scheduler, RISC-V RV32I instruction decoder, 4-way round-robin arbiter.
- **Communication & Storage**: UART transmitter, UART receiver, SPI master, I2C bus controller, synchronous FIFO (8x8), LIFO stack (8x8), multi-port register file (8x8), PWM generator.

---

## Quick Start

### 1. Requirements

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Ensure Yosys is available:
- **Windows**: The pipeline auto-detects OSS CAD Suite at `D:\oss-cad-suite-...` or paths specified by `$env:OSS_CAD_SUITE`.
- **Linux/macOS**: `sudo apt install yosys` or `brew install yosys`.

### 2. End-to-End Execution

#### Windows PowerShell:
```powershell
.\scripts\run_pipeline.ps1
```

#### Linux / macOS:
```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

#### Python CLI:
```bash
python -m src.data_prep.batch_parse data/raw_rtl -o data/yosys_synth --sog-dir data/sog_graphs --report-dir data/reports
```

### 3. Single Design Execution

To synthesize and extract the SOG for an individual Verilog file:

```bash
# Step 1: Synthesize to technology-independent bit-level AST
python -m src.data_prep.yosys_synthesis data/raw_rtl/adder_cla_16bit.v -o data/yosys_synth

# Step 2: Build SOG, compute path delays, propagate toggle rates, and save stats
python -m src.data_prep.sog_converter data/yosys_synth/adder_cla_16bit.json -o data/sog_graphs --name adder_cla_16bit
```

### 4. Running Verification Tests

Run the complete test suite:
```bash
pytest -v
```

---

## Generated Outputs & Statistics

For each design `<name>`, the pipeline generates:
- `data/sog_graphs/<name>.pkl`: NetworkX `DiGraph` containing the full bit-level SOG.
- `data/sog_graphs/<name>_features.pkl`: List of critical paths, path delays, hop counts, and operator counts.
- `data/sog_graphs/<name>_stats.json`: Extracted metrics for the design.

Aggregated reports across all designs are saved in `data/reports/`:
- `sog_statistics.csv`: Complete spreadsheet of 28 structural, timing, power, and area metrics per design.
- `sog_statistics.json`: Machine-readable JSON records for downstream ML feature preparation.
- `sog_summary.md`: Clean Markdown summary table comparing all designs.

For detailed formulas, operator mappings, and metric descriptions, refer to [docs/SOG_PIPELINE.md](docs/SOG_PIPELINE.md).
