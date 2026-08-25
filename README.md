# MasterRTL Project

This project prepares raw RTL designs for machine-learning-based PPA estimation by converting HDL into a technology-independent bit-level representation using Yosys.

## Required dependencies

1. Install Yosys from the official project, or via your system package manager.
   - Linux/macOS: `sudo apt install yosys` or equivalent
   - macOS (Homebrew): `brew install yosys`
   - Windows: install the Yosys binary and ensure it is on your PATH
2. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Data preparation flow

The synthesis pipeline converts `.v`/`.sv` sources into two outputs per design:

- `*.synth.v`: technology-independent bit-level Verilog
- `*.json`: Yosys AST JSON for graph extraction

## Single design synthesis

```bash
python -m src.data_prep.yosys_synthesis path/to/design.v --output-dir data/processed
```

Optional explicit top module:

```bash
python -m src.data_prep.yosys_synthesis path/to/design.v --top top_module --output-dir data/processed
```

## Batch processing

```bash
python -m src.data_prep.batch_parse data/raw_rtl --recursive -o data/yosys_synth
```

This command scans the directory for Verilog files, synthesizes each one individually, and logs success or failure without stopping the entire batch.

## Simple Operator Graph (SOG) construction

After generating a Yosys JSON AST, the SOG builder converts the gate-level netlist into a directed graph for downstream timing and switching analysis.

### What the SOG contains

- Nodes represent net indices and logic outputs
- Directed edges capture signal dependencies from sources to sinks
- Supported operators are restricted to:
  - `AND`
  - `OR`
  - `XOR`
  - `NOT`
  - `MUX`
  - `DFF`
- Each node stores:
  - `op`
  - `bit_width`
  - `fan_in`
  - `fan_out`
  - `operator_one_hot`

### Build a SOG from one design

```bash
python -m src.data_prep.sog_converter data/yosys_synth/example.json -o data/sog_graphs --name example
```

This creates:

- `data/sog_graphs/example.pkl` — the NetworkX graph object
- `data/sog_graphs/example_features.pkl` — extracted delay/path metadata

### What is extracted

The converter computes:

1. Node delay scores using a simple RC-inspired linear fan-out model.
2. Register-to-register path summaries, including delay and traversal path.
3. Toggle-rate propagation using a basic Boolean switching approximation.

These artifacts are intended to be used as lightweight graph features before model training or PPA feature extraction.

### Example script usage

```python
from pathlib import Path
from src.data_prep.sog_converter import build_sog_from_json, save_graph_artifacts

graph = build_sog_from_json(Path("data/yosys_synth/example.json"))
artifacts = save_graph_artifacts(graph, Path("data/sog_graphs"), "example")
print(artifacts)
```
