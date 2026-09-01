# MasterRTL: Simple Operator Graph (SOG) Pipeline & Dataset Documentation

This document provides a comprehensive technical guide to the pre-synthesis data preparation and **Simple Operator Graph (SOG)** generation engine of MasterRTL, implemented in alignment with:

> **Wenji Fang, Yao Lu, Shang Liu, Qijun Zhang, Ceyu Xu, Lisa Wu Wills, Hongce Zhang, and Zhiyao Xie**,  
> *"Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques,"*  
> **IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)**, Vol. 44, No. 1, January 2025.

---

## 1. Motivation & The SOG Representation

### 1.1 The Challenge of RTL-Stage PPA Evaluation
In modern ASIC design, evaluating the Power, Performance, and Area (PPA) of Register-Transfer Level (RTL) code traditionally requires running full logic synthesis (e.g., Synopsys Design Compiler) and physical layout (e.g., Cadence Innovus). This process takes hours or days for medium-to-large designs, creating an immense bottleneck during architectural exploration and code optimization.

Prior ML-based methods attempted to predict PPA directly from word-level Abstract Syntax Trees (ASTs). However:
- AST representations retain high-level word widths, complex operators, and diverse coding styles, hindering model generalization across designs.
- ASTs lack a direct one-to-one register mapping with the post-synthesis gate-level netlist.

### 1.2 The Simple Operator Graph (SOG)
To overcome AST limitations without running time-consuming logic optimization or technology mapping, MasterRTL transforms raw RTL into a technology-independent, bit-level **Simple Operator Graph (SOG)**.

The SOG consists strictly of:
1. **Single-bit registers**: `DFF` (D flip-flops)
2. **Five primary single-bit logic operations**:
   - 2-input `AND`
   - 2-input `OR`
   - 2-input `XOR`
   - 1-input `NOT`
   - 2-to-1 `MUX`
3. **Primary Inputs (PI)** and **Primary Outputs (PO)**

```
             +---------------------+
             |   Raw RTL (.v)      |
             +---------------------+
                        |
                        v
          [Yosys Generic Bit-Synthesis]
          (proc; opt; fsm; techmap; abc)
                        |
                        v
          +-----------------------------+
          | Technology-Independent JSON |
          +-----------------------------+
                        |
                        v
           [SOG Converter & Feature Engine]
            * Single-bit graph (V, E)
            * STA Timing DAG (Launch/Capture)
            * Linear RC Node Delay Model
            * Critical Path Extraction
            * Boolean Toggle Propagation
                        |
       +----------------+----------------+
       |                                 |
       v                                 v
[Graph Artifacts (.pkl)]      [Aggregated Reports]
  - <name>.pkl (Graph)          - sog_statistics.csv
  - <name>_features.pkl         - sog_statistics.json
  - <name>_stats.json           - sog_summary.md
```

### 1.3 Key Advantages of SOG
- **Gate-Level Structural Similarity**: Bridges the abstraction gap between behavioral RTL and final synthesized netlists.
- **Cross-Design Generalization**: Canonical 5-operator boolean primitives unify disparate coding styles and domain designs.
- **One-to-One Register Mapping**: Every physical register in the post-synthesis netlist corresponds directly to a single-bit `DFF` in the SOG.

---

## 2. Mathematical & Algorithmic Foundations

### 2.1 Technology-Independent Bit-Level Synthesis
The generic synthesis script executed by Yosys converts word-level HDL into single-bit primitives:
```tcl
read_verilog -sv <design.v>
hierarchy -check -top <top_module>
proc
opt_expr
wreduce
alumacc
share
opt
fsm
memory -nomap
techmap
abc -g AND,OR,XOR,MUX
clean
write_verilog -noattr <synth.v>
write_json <synth.json>
```

### 2.2 Analytical Node Delay Modeling
To assess delays at the RTL stage without library technology mapping, an analytical linear resistance-capacitance (RC) fan-out model is applied to every node $v$:

$$\text{delay}(v) = \text{base\_delay}(\text{op}(v)) + k_{\text{fanout}} \cdot \text{fan\_out}(v)$$

Default calibrated coefficients (based on NanGate 45nm cell characteristics with $k_{\text{fanout}} = 0.15$):
| Operator | Base Delay | Description |
| :--- | :---: | :--- |
| `NOT` | 0.8 | Single inverter stage |
| `AND` | 1.0 | 2-input AND primitive |
| `OR` | 1.1 | 2-input OR primitive |
| `XOR` | 1.2 | 2-input XOR primitive |
| `MUX` | 1.6 | 2-to-1 multiplexer cell |
| `DFF` | 2.4 | Flip-flop clock-to-Q propagation |
| `NET` / `IO` | 0.1 / 0.0 | Interconnect / boundary pin |

### 2.3 Static Timing Analysis (STA) DAG Construction
In sequential circuits, register feedback (e.g., $Q \to \text{adder} \to D$) creates directed cycles across clock cycles. To allow exact topological analysis:
1. Each physical register $R_k$ is decomposed into:
   - **Launch Point** ($R_k.Q$): timing path startpoint, launching signals into combinational gates.
   - **Capture Point** ($R_k.D$): timing path endpoint, capturing signals from combinational logic.
2. Primary Inputs (`PI`) are registered as launch startpoints.
3. Primary Outputs (`PO`) are registered as capture endpoints for combinational circuits.
4. The resulting graph $G_{\text{dag}}$ is guaranteed to be a **Directed Acyclic Graph (DAG)** ($nx.is\_directed\_acyclic\_graph(G) = \text{True}$).

### 2.4 Critical Path Identification & Slack Proxies
For each capture endpoint $j \in [1, N]$:
1. Dynamic programming tracks the longest path arrival time:
   $$\text{arrival}(v) = \text{delay}(v) + \max_{u \in \text{pred}(v)} \text{arrival}(u)$$
2. Backtracking from $R_j.D$ identifies the critical path $P^R_{* \to j}$.
3. For each critical path, MasterRTL records:
   - Start node and end register
   - Total accumulated path delay
   - Total hop count (logic depth)
   - Specific counts of `AND`, `OR`, `XOR`, `NOT`, `MUX`, `DFF` on the path
   - Accumulated fan-out sum

From these $N$ endpoint critical paths, timing proxies are derived:
$$\text{CritDelay} = \max_{j} \text{delay}(P^R_{* \to j})$$
$$\text{MeanDelay} = \frac{1}{N} \sum_{j=1}^N \text{delay}(P^R_{* \to j})$$
Percentiles (10th, 50th, 90th) are extracted for downstream calibration models.

### 2.5 Boolean Toggle-Rate Propagation
Dynamic power is governed by switching activities $\alpha$:
$$P_{\text{dyn}} \propto \sum_{v} \alpha(v) \cdot C_{\text{load}}(v) \cdot V_{\text{dd}}^2 \cdot f$$

Toggle rates are propagated in topological order across the DAG:
- **`NOT`**: $\alpha_{\text{out}} = \alpha_{\text{in}}$
- **`AND` / `OR`**: $\alpha_{\text{out}} = \frac{1}{2}\bar{\alpha} \cdot (1 - \frac{1}{4}\bar{\alpha})$
- **`XOR`**: $\alpha_{\text{out}} = \alpha_A(1 - \alpha_B) + \alpha_B(1 - \alpha_A)$
- **`MUX`**: $\alpha_{\text{out}} = 0.5\alpha_{D0} + 0.5\alpha_{D1} + 0.1\alpha_S$
- **`DFF` / `IO`**: Initialized with standard baseline activity ($\alpha = 0.10$) or SAIF switching activity.

Weighted switching activity is computed as:
$$\text{WeightedToggle} = \sum_{v \in V} \text{fan\_out}(v) \cdot \alpha(v)$$

---

## 3. Benchmark Dataset Catalog (41 Designs)

The dataset in `data/raw_rtl/` contains 41 diverse, synthesizable designs spanning 6 major domains:

### Arithmetic & DSP Units (10 designs)
| File | Top Module | Description |
| :--- | :--- | :--- |
| `adder_ripple_8bit.v` | `adder_ripple_8bit` | 8-bit ripple-carry adder with input/output registers |
| `adder_cla_16bit.v` | `adder_cla_16bit` | 16-bit carry-lookahead adder with registered stages |
| `adder_kogge_stone_16bit.v` | `adder_kogge_stone_16bit` | 16-bit parallel prefix Kogge-Stone fast adder |
| `subtractor_16bit.v` | `subtractor_16bit` | 16-bit adder/subtractor with borrow, overflow, and zero flags |
| `multiplier_array_4x4.v` | `multiplier_array_4x4` | 4x4-bit combinational array multiplier with output registers |
| `multiplier_booth_8bit.v` | `multiplier_booth_8bit` | 8-bit signed Radix-4 Booth multiplier |
| `multiplier_sequential_8bit.v` | `multiplier_sequential_8bit` | 8-bit sequential shift-and-add multiplier with controller |
| `divider_restoring_8bit.v` | `divider_restoring_8bit` | 8-bit restoring division unit with division-by-zero detection |
| `mac_unit_8bit.v` | `mac_unit_8bit` | 8-bit Multiply-Accumulate (MAC) pipeline unit |
| `cordic_step_8bit.v` | `cordic_step_8bit` | CORDIC rotational iteration step operator |

### Data Path & Bit Manipulation (7 designs)
| File | Top Module | Description |
| :--- | :--- | :--- |
| `alu_4bit.v` | `alu_4bit` | 4-bit classic 74181-style arithmetic logic unit |
| `alu_16bit.v` | `alu_16bit` | 16-bit RISC ALU with Zero, Negative, Carry, and Overflow flags |
| `alu_bitwise_32bit.v` | `alu_bitwise_32bit` | 32-bit wide multi-function bitwise operator |
| `barrel_shifter_8bit.v` | `barrel_shifter_8bit` | 8-bit logarithmic barrel shifter (SLL, SRL, SRA, ROR) |
| `priority_encoder_16to4.v` | `priority_encoder_16to4` | 16-to-4 priority encoder with valid indicator |
| `parity_generator_16bit.v` | `parity_generator_16bit` | 16-bit even and odd parity generation unit |
| `popcount_32bit.v` | `popcount_32bit` | 32-bit population count (Hamming weight) adder tree |

### Sequential & Counter Circuits (5 designs)
| File | Top Module | Description |
| :--- | :--- | :--- |
| `counter_8bit.v` | `counter_8bit` | 8-bit synchronous up-counter with enable |
| `counter_updown_8bit.v` | `counter_updown_8bit` | 8-bit loadable up/down counter with terminal count |
| `counter_gray_8bit.v` | `counter_gray_8bit` | 8-bit Gray-code counter with binary-to-gray conversion |
| `counter_ring_8bit.v` | `counter_ring_8bit` | 8-bit selectable Johnson / Ring shift counter |
| `lfsr_16bit.v` | `lfsr_16bit` | 16-bit maximal-length Fibonacci LFSR pseudo-random generator |

### Cryptography, Error Detection & Hashing (4 designs)
| File | Top Module | Description |
| :--- | :--- | :--- |
| `crc8_ccitt.v` | `crc8_ccitt` | 8-bit CRC-CCITT serial/parallel calculation engine |
| `crc32_ethernet.v` | `crc32_ethernet` | 32-bit Ethernet polynomial CRC-32 generator |
| `aes_sbox_lut.v` | `aes_sbox_lut` | AES Rijndael S-Box substitution lookup logic |
| `sha256_round_fn.v` | `sha256_round_fn` | SHA-256 compression round function logic ($Ch, Maj, \Sigma_0, \Sigma_1$) |

### State Machines & Control Units (5 designs)
| File | Top Module | Description |
| :--- | :--- | :--- |
| `fsm_traffic_light.v` | `fsm_traffic_light` | 4-way intersection traffic controller with timer & emergency override |
| `fsm_vending_machine.v` | `fsm_vending_machine` | Multi-coin vending machine with item dispensing & change |
| `fsm_elevator_control.v` | `fsm_elevator_control` | 4-floor elevator scheduler with door and motor controls |
| `instruction_decoder_rv32.v` | `instruction_decoder_rv32` | RISC-V RV32I instruction field & control signal decoder |
| `arbiter_round_robin_4way.v` | `arbiter_round_robin_4way` | 4-request round-robin arbiter with fair token passing |

### Communication & Storage Interfaces (8 designs)
| File | Top Module | Description |
| :--- | :--- | :--- |
| `uart_tx.v` | `uart_tx` | UART transmitter with baud rate generator & shift register |
| `uart_rx.v` | `uart_rx` | UART receiver with start bit detection and 8x oversampling |
| `spi_master.v` | `spi_master` | SPI master bus interface with CPOL/CPHA configuration |
| `i2c_master_bit_ctrl.v` | `i2c_master_bit_ctrl` | I2C byte/bit-level bus timing controller |
| `fifo_sync_8x8.v` | `fifo_sync_8x8` | 8-word deep, 8-bit wide synchronous FIFO with full/empty flags |
| `lifo_stack_8x8.v` | `lifo_stack_8x8` | 8-word deep LIFO stack with push/pop and pointer flags |
| `register_file_8x8.v` | `register_file_8x8` | 8x8 dual-read single-write multi-port register bank |
| `pwm_generator_8bit.v` | `pwm_generator_8bit` | 8-bit pulse width modulation generator with duty cycle register |
| `example.v` | `counter_8bit` | Reference 8-bit synchronous counter |

---

## 4. Statistics Reference & Schema

For every design, the converter extracts 28 metrics saved in `data/reports/sog_statistics.json` and `sog_statistics.csv`:

| Metric Key | Type | Description |
| :--- | :---: | :--- |
| `module_name` | string | Resolved top module identifier |
| `total_nodes` | int | Total number of vertices in SOG |
| `total_edges` | int | Total directed edges in SOG |
| `density` | float | Directed graph density $|E| / (|V|(|V|-1))$ |
| `num_registers` | int | Single-bit DFF flip-flops ($N$) |
| `num_comb_nodes` | int | Total combinational logic gates |
| `num_and` | int | Count of 2-input AND operators |
| `num_or` | int | Count of 2-input OR operators |
| `num_xor` | int | Count of 2-input XOR operators |
| `num_not` | int | Count of NOT inverters |
| `num_mux` | int | Count of 2-to-1 multiplexers |
| `num_primary_inputs` | int | Count of top-level input bits |
| `num_primary_outputs` | int | Count of top-level output bits |
| `max_fanout` | int | Maximum fan-out degree across nodes |
| `mean_fanout` | float | Average fan-out degree across nodes |
| `max_fanin` | int | Maximum fan-in degree across nodes |
| `mean_fanin` | float | Average fan-in degree across nodes |
| `total_reg_paths` | int | Number of extracted critical timing paths |
| `comb_depth` | int | Maximum combinational hop depth |
| `critical_path_delay` | float | Maximum path delay in design |
| `mean_path_delay` | float | Average path delay across endpoint paths |
| `min_path_delay` | float | Shortest endpoint path delay |
| `path_delay_p10` | float | 10th percentile path delay |
| `path_delay_p50` | float | 50th percentile (median) path delay |
| `path_delay_p90` | float | 90th percentile path delay |
| `total_toggle_rate` | float | Sum of toggle rates across all nodes |
| `mean_toggle_rate` | float | Mean switching activity across nodes |
| `weighted_toggle_rate` | float | Sum of $(\text{fan\_out} \times \alpha)$ across nodes |
| `est_sequential_area` | float | Estimated DFF area ($N \times 4.522\,\mu m^2$) |
| `est_combinational_area` | float | Estimated combinational gate area ($\mu m^2$) |
| `est_total_area` | float | Total estimated standard cell area ($\mu m^2$) |

---

## 5. Execution Guide

### 5.1 One-Click Windows PowerShell Execution
```powershell
.\scripts\run_pipeline.ps1
```
This script automatically discovers the local OSS CAD Suite environment, executes batch synthesis, builds SOG representations, and exports summary reports.

### 5.2 Python Direct Batch Execution
```bash
python -m src.data_prep.batch_parse data/raw_rtl -o data/yosys_synth --sog-dir data/sog_graphs --report-dir data/reports
```

### 5.3 Single Design Execution
```bash
# 1. Synthesize Verilog to JSON AST
python -m src.data_prep.yosys_synthesis data/raw_rtl/adder_cla_16bit.v -o data/yosys_synth

# 2. Build SOG and extract features
python -m src.data_prep.sog_converter data/yosys_synth/adder_cla_16bit.json -o data/sog_graphs --name adder_cla_16bit
```

### 5.4 Running the Test Suite
```bash
pytest -v
```
Verifies toolchain discovery, operator normalization, STA DAG reachability, critical path extraction, and all ML models.

---

## 6. Machine Learning Models for Presynthesis PPA Estimation

MasterRTL implements the exact multistage modeling architecture specified in **Table II** of the reference paper:

### 6.1 Timing Model Architecture (Section II-B)
1. **Analytical Node Delay**: Linear RC delay model evaluated on SOG ($k_{\text{fanout}} = 0.15$).
2. **Critical Path Extraction**: Acyclic STA DAG dynamic programming extracting $N$ endpoint paths $P^R_{* \to j}$.
3. **Path-Level Delay Model ($f_{\text{path}}^t$)**:
   - **Model**: `RandomForestRegressor` with 80 trees, max depth of 20.
   - **Features**: Total nodes on path, operator counts (`AND`, `OR`, `XOR`, `NOT`, `MUX`, `DFF`), accumulated analytical delay, accumulated fan-out.
   - **Output**: Predicted gate-level netlist path delay.
4. **Path-Level Inference Engine**:
   - For all $N$ endpoint paths, predicts delay $\hat{d}_j$.
   - $\text{WNS}^R = \min_{j} (clk - \hat{d}_j)$.
   - $\text{TNS}^R = \sum_{j} \min(0, clk - \hat{d}_j)$.
   - Slack distribution percentiles: worst 1% of critical paths (worst, 10%, 50%, 90% percentiles).
5. **Design-Level Timing Calibration Model**:
   - **Model**: `XGBoostRegressor` with 45 estimators, max depth of 8.
   - **Features**: Design scale features (total nodes, edges, registers, combinational nodes, operator breakdown) + path estimates ($\text{WNS}^R, \text{TNS}^R$) + slack percentiles.
   - **Target**: Calibrated netlist $\text{WNS}$ and $\text{TNS}$.

### 6.2 Power Model Architecture (Section II-C)
1. **Toggle Rate Propagation**: Topologically propagates switching activities $\alpha(v)$ through logic gates.
2. **Module/Sub-SOG Level Power Model**:
   - **Model**: `XGBoostRegressor` with 30 estimators, max depth of 6.
   - **Features**: Sum of toggle rates, average toggle rate, fanout-weighted switching sum ($\sum \text{fan\_out} \cdot \alpha$), total nodes, operator breakdown.
   - Computes $\text{Power}^G = \sum_{i=1}^M k_i \cdot \text{Power}^{G_i}$.
3. **Design-Level Power Calibration Model**:
   - **Model**: `XGBoostRegressor` with 45 estimators, max depth of 8.
   - **Features**: Summed module power, global toggle statistics, and whole SOG design scale.
   - **Target**: Calibrated total power ($P_G$).

### 6.3 Area Model Architecture (Section II-D)
1. **Sequential Area**:
   - Exact analytical formula: $\text{Area}_{\text{seq}} = N_{\text{registers}} \times \text{Area}_{\text{DFF}}$ (NanGate 45nm standard cell DFF area $= 4.522\,\mu m^2$).
2. **Combinational Area Prediction**:
   - **Model**: `XGBoostRegressor` with 45 estimators, max depth of 12.
   - **Features**: Preliminary combinational area from operator weights, gate counts (`AND`, `OR`, `XOR`, `NOT`, `MUX`), total nodes, edges, density, and fan-in/fan-out statistics.
3. **Total Area**:
   - $\text{Area}_{\text{total}} = \text{Area}_{\text{seq}} + \widehat{\text{Area}}_{\text{comb}}$.

### 6.4 Transfer Model Architecture (Section II-E)
1. **Layout & Tech Transfer Model**:
   - **Model**: `XGBoostRegressor` with 15 estimators, max depth of 8.
   - **Features**: Initial MasterRTL source PPA predictions, library scaling factors between target and source libraries, scaled PPA metrics, and design scale features.
   - **Targets**: Post-placement PPA (layout stage) and cross-technology PPA (TSMC 22nm, 28nm, 40nm, 65nm, MIN/MAX process corners).

---

## 7. Model Training & 10-Fold Cross-Validation

To reproduce the experimental methodology in Section III-A:

```powershell
# Windows PowerShell
.\scripts\train_all.ps1
```

```bash
# Linux / macOS
./scripts/train_all.sh
```

```bash
# Python Direct
python scripts/train_all.py --k-folds 10 --output-model models/master_rtl_checkpoint.pkl
```

### Evaluation Metrics (Section III-A, Equations 4, 5, 6)
- **Correlation ($R$)**: Pearson correlation coefficient between predicted and ground-truth metrics.
- **MAPE (%)**: Mean Absolute Percentage Error.
- **MAE**: Mean Absolute Error.
- **RRSE**: Root Relative Square Error.


