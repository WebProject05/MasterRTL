"""Batch Graph Generator for all representations mentioned in the MasterRTL paper.

Generates:
1. SOG (Simple Operator Graph) -> data/graphs/sog/
2. AST (Word-level Abstract Syntax Tree) -> data/graphs/ast/
3. AIG (And-Inverter Graph) -> data/graphs/aig/
4. Netlist (Gate-Level Netlist Graph) -> data/graphs/netlist/
5. Timing DAG (STA Acyclic Timing Graph) -> data/graphs/timing_dag/
6. CDFG (Control Data Flow Graph) -> data/graphs/cdfg/

Outputs cross-representation comparison tables to data/graphs/graph_summary.md and .csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data_prep.graph_generators import (
    build_aig_graph,
    build_ast_graph,
    build_cdfg_graph,
    build_netlist_graph,
    build_sog,
    build_timing_dag_graph,
    extract_graph_stats,
)
from src.data_prep.sog_converter import (
    compute_node_delays,
    compute_path_features,
    propagate_toggle_rates,
)
from src.data_prep.yosys_synthesis import (
    detect_top_module,
    find_yosys_binary,
    prepare_yosys_env,
)


def _normalize(p: Path | str) -> str:
    return str(Path(p).resolve()).replace("\\", "/")


def synthesize_all_stages(
    verilog_path: Path,
    top_module: str,
    stage_dir: Path,
    yosys_exe: Path,
    env: dict[str, str],
) -> dict[str, Path]:
    """Execute multi-stage Yosys passes to produce JSON ASTs for AST, SOG, AIG, and Netlist."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    stem = verilog_path.stem

    ast_json = stage_dir / f"{stem}_ast.json"
    sog_json = stage_dir / f"{stem}_sog.json"
    aig_json = stage_dir / f"{stem}_aig.json"
    net_json = stage_dir / f"{stem}_netlist.json"

    norm_v = _normalize(verilog_path)

    # 1. AST pass (Word-level elaboration before bit-blasting)
    cmd_ast = (
        f"read_verilog -sv {norm_v};\n"
        f"hierarchy -check -top {top_module};\n"
        f"proc;\n"
        f"opt_expr;\n"
        f"wreduce;\n"
        f"alumacc;\n"
        f"share;\n"
        f"opt;\n"
        f"write_json {_normalize(ast_json)};\n"
    )

    # 2. SOG pass (Canonical bit-level primitives: AND, OR, XOR, NOT, MUX, DFF)
    cmd_sog = (
        f"design -reset;\n"
        f"read_verilog -sv {norm_v};\n"
        f"hierarchy -check -top {top_module};\n"
        f"proc;\n"
        f"opt_expr;\n"
        f"wreduce;\n"
        f"alumacc;\n"
        f"share;\n"
        f"opt;\n"
        f"fsm;\n"
        f"memory -nomap;\n"
        f"memory_map;\n"
        f"techmap;\n"
        f"abc -g AND,OR,XOR,MUX;\n"
        f"clean;\n"
        f"write_json {_normalize(sog_json)};\n"
    )

    # 3. AIG pass (And-Inverter Graph: strictly 2-input AND + NOT + DFF)
    cmd_aig = (
        f"design -reset;\n"
        f"read_verilog -sv {norm_v};\n"
        f"hierarchy -check -top {top_module};\n"
        f"proc;\n"
        f"memory -nomap;\n"
        f"memory_map;\n"
        f"techmap;\n"
        f"abc -g AND;\n"
        f"clean;\n"
        f"write_json {_normalize(aig_json)};\n"
    )

    # 4. Netlist pass (Standard logic cell library primitives)
    cmd_net = (
        f"design -reset;\n"
        f"read_verilog -sv {norm_v};\n"
        f"hierarchy -check -top {top_module};\n"
        f"proc;\n"
        f"memory -nomap;\n"
        f"memory_map;\n"
        f"techmap;\n"
        f"abc -g AND,OR,XOR,MUX,NAND,NOR;\n"
        f"clean;\n"
        f"write_json {_normalize(net_json)};\n"
    )

    combined_cmd = cmd_ast + cmd_sog + cmd_aig + cmd_net

    p = subprocess.run(
        [str(yosys_exe), "-p", combined_cmd],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Yosys multi-stage synthesis failed for {verilog_path}: {p.stderr}")

    return {
        "ast": ast_json,
        "sog": sog_json,
        "aig": aig_json,
        "netlist": net_json,
    }


def process_single_design(
    verilog_path: Path,
    graphs_root: Path,
    stage_dir: Path,
    yosys_exe: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    """Generate all 6 graph representations for one Verilog benchmark design."""
    top_module = detect_top_module(verilog_path)
    stem = verilog_path.stem

    # 1. Synthesize multi-stage JSON ASTs
    json_paths = synthesize_all_stages(verilog_path, top_module, stage_dir, yosys_exe, env)

    # 2. Build SOG & Timing DAG
    sog_graph = build_sog(json_paths["sog"], top_module)
    sog_toggles = propagate_toggle_rates(sog_graph)
    timing_dag = build_timing_dag_graph(sog_graph)
    crit_paths = compute_path_features(sog_graph)

    # 3. Build AST & CDFG
    ast_graph = build_ast_graph(json_paths["ast"], top_module)
    cdfg_graph = build_cdfg_graph(json_paths["ast"], top_module)

    # 4. Build AIG & Netlist
    aig_graph = build_aig_graph(json_paths["aig"], top_module)
    netlist_graph = build_netlist_graph(json_paths["netlist"], top_module)

    graph_map = {
        "sog": sog_graph,
        "timing_dag": timing_dag,
        "ast": ast_graph,
        "cdfg": cdfg_graph,
        "aig": aig_graph,
        "netlist": netlist_graph,
    }

    results: dict[str, Any] = {}

    for gtype, g_obj in graph_map.items():
        type_dir = graphs_root / gtype
        type_dir.mkdir(parents=True, exist_ok=True)

        pkl_path = type_dir / f"{stem}.pkl"
        stats_path = type_dir / f"{stem}_stats.json"

        # Serialize NetworkX graph
        with open(pkl_path, "wb") as f:
            pickle.dump(g_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Extract stats
        stats = extract_graph_stats(g_obj, gtype, stem)

        # Extra features for SOG
        if gtype == "sog":
            feat_path = type_dir / f"{stem}_features.pkl"
            with open(feat_path, "wb") as f:
                pickle.dump({"critical_paths": crit_paths, "toggle_rates": sog_toggles}, f)
            stats["critical_paths_count"] = len(crit_paths)

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        results[gtype] = stats

    return results


def build_comparison_reports(
    all_results: dict[str, dict[str, Any]],
    graphs_root: Path,
) -> None:
    """Generate Markdown and CSV cross-representation comparison tables."""
    csv_path = graphs_root / "graph_summary.csv"
    md_path = graphs_root / "graph_summary.md"
    manifest_path = graphs_root / "graph_manifest.json"

    graphs_root.mkdir(parents=True, exist_ok=True)
    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Types ordered by abstraction level
    type_order = ["ast", "cdfg", "sog", "aig", "netlist", "timing_dag"]
    type_labels = {
        "ast": "Word AST (ICCAD'22/ISCA'22)",
        "cdfg": "Control Data Flow (CDFG)",
        "sog": "MasterRTL SOG (Bit-level)",
        "aig": "And-Inverter Graph (AIG)",
        "netlist": "Gate Netlist Graph (G)",
        "timing_dag": "STA Timing DAG (G_dag)",
    }

    # CSV Summary
    rows = []
    for design, types in sorted(all_results.items()):
        for gtype in type_order:
            if gtype in types:
                st = types[gtype]
                rows.append({
                    "design": design,
                    "representation": gtype,
                    "representation_name": type_labels.get(gtype, gtype),
                    "nodes": st.get("num_nodes", 0),
                    "edges": st.get("num_edges", 0),
                    "density": st.get("density", 0.0),
                    "avg_degree": st.get("avg_degree", 0.0),
                    "is_dag": st.get("is_dag", False),
                    "max_depth": st.get("max_depth", 0),
                    "registers": st.get("num_registers", 0),
                    "primary_inputs": st.get("num_primary_inputs", 0),
                    "primary_outputs": st.get("num_primary_outputs", 0),
                })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Compute Averages
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        gt = r["representation"]
        aggregates[gt]["nodes"].append(r["nodes"])
        aggregates[gt]["edges"].append(r["edges"])
        aggregates[gt]["density"].append(r["density"])
        aggregates[gt]["avg_degree"].append(r["avg_degree"])
        aggregates[gt]["max_depth"].append(r["max_depth"])
        aggregates[gt]["registers"].append(r["registers"])

    # Markdown Summary
    md_lines = [
        "# MasterRTL Graph Representations Summary & Comparative Analysis",
        "",
        "This document details the generation and comparative structural properties of all graph representation",
        "types analyzed in the paper: **'Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques'** (IEEE TCAD 2025).",
        "",
        "---",
        "",
        "## 1. Cross-Representation Aggregate Comparison",
        "",
        "Comparison of structural complexity across the 6 graph representations across all benchmark designs:",
        "",
        "| Representation Type | Category | Avg Nodes (|V|) | Avg Edges (|E|) | Avg Density | Avg Degree | Avg Depth | Avg Registers | Guaranteed DAG |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for gt in type_order:
        data = aggregates[gt]
        n_count = len(data["nodes"])
        avg_n = sum(data["nodes"]) / n_count if n_count else 0
        avg_e = sum(data["edges"]) / n_count if n_count else 0
        avg_dens = sum(data["density"]) / n_count if n_count else 0
        avg_deg = sum(data["avg_degree"]) / n_count if n_count else 0
        avg_depth = sum(data["max_depth"]) / n_count if n_count else 0
        avg_regs = sum(data["registers"]) / n_count if n_count else 0
        dag_str = "Yes (Acyclic)" if gt == "timing_dag" else "Cyclic (Registers)"
        cat_str = "Word-level" if gt in ["ast", "cdfg"] else ("Gate-level" if gt == "netlist" else "Bit-level")

        md_lines.append(
            f"| **{type_labels[gt]}** | {cat_str} | {avg_n:,.1f} | {avg_e:,.1f} | {avg_dens:.5f} | {avg_deg:.2f} | {avg_depth:.1f} | {avg_regs:.1f} | {dag_str} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Representation Details & Paper Grounding",
        "",
        "### A. SOG (Simple Operator Graph) — `data/graphs/sog/`",
        "- **Reference**: Paper Section II-A, Fig. 2(b).",
        "- **Description**: Bit-level representation bypassing technology mapping and logic optimization. Comprises single-bit registers and 5 primary single-bit logic operations (`AND`, `OR`, `XOR`, `NOT`, `MUX`).",
        "",
        "### B. AST (Abstract Syntax Tree) — `data/graphs/ast/`",
        "- **Reference**: Paper Section I, Section II-A, Fig. 2(a), Table III.",
        "- **Description**: Word-level AST representation used in prior state-of-the-art works (ICCAD'22 [21], ISCA'22 [22]). Operations like 16-bit additions and multi-bit multiplexers are retained as single multi-bit nodes.",
        "",
        "### C. AIG (And-Inverter Graph) — `data/graphs/aig/`",
        "- **Reference**: Paper Section I (Page 2), citing [18], [19], [20].",
        "- **Description**: Classical logic synthesis bit-level representation where combinational logic is strictly decomposed into 2-input `AND` gates and `NOT` inverters.",
        "",
        "### D. Netlist Gate Graph ($G$) — `data/graphs/netlist/`",
        "- **Reference**: Paper Section II, Fig. 2(c).",
        "- **Description**: Gate-level netlist graph where nodes represent physical logic cells (NAND, NOR, XNOR, AOI, OAI, MUX, DFF) and edges represent interconnect nets.",
        "",
        "### E. STA Timing DAG ($G_{\\text{dag}}$) — `data/graphs/timing_dag/`",
        "- **Reference**: Paper Section II-B (Timing Evaluation Flow).",
        "- **Description**: Decoupled acyclic timing graph where registers are split into launch points ($R.Q$) and capture points ($R.D$). Enables exact dynamic programming for arrival times and critical paths $P^R_{* \\to j}$.",
        "",
        "### F. CDFG (Control Data Flow Graph) — `data/graphs/cdfg/`",
        "- **Reference**: Behavioral flow graph differentiating control flow decisions (`BRANCH`, `STATE`) from data arithmetic manipulation (`DATA_ARITH`, `DATA_LOGIC`).",
        "",
        "---",
        "",
        "## 3. Per-Design Representation Matrix (Sample)",
        "",
        "| Design | Metric | AST | CDFG | SOG | AIG | Netlist | Timing DAG |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    sample_designs = list(sorted(all_results.keys()))[:15]
    for d in sample_designs:
        types = all_results[d]
        v_str = [str(types.get(gt, {}).get("num_nodes", "-")) for gt in type_order]
        e_str = [str(types.get(gt, {}).get("num_edges", "-")) for gt in type_order]
        md_lines.append(f"| `{d}` | Nodes (|V|) | {v_str[0]} | {v_str[1]} | {v_str[2]} | {v_str[3]} | {v_str[4]} | {v_str[5]} |")
        md_lines.append(f"| | Edges (|E|) | {e_str[0]} | {e_str[1]} | {e_str[2]} | {e_str[3]} | {e_str[4]} | {e_str[5]} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch generate all graph representations mentioned in the paper.")
    parser.add_argument("input_dir", nargs="?", default="data/raw_rtl", help="Directory containing input Verilog files")
    parser.add_argument("-o", "--output-dir", default="data/graphs", help="Root directory for generated graphs")
    parser.add_argument("--stage-dir", default="data/yosys_stages", help="Directory for temporary Yosys stage JSONs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of designs to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    stage_dir = Path(args.stage_dir).expanduser().resolve()

    verilog_files = sorted(input_dir.glob("*.v"))
    if not verilog_files:
        print(f"[ERROR] No Verilog files found in {input_dir}")
        return 1

    if args.limit:
        verilog_files = verilog_files[: args.limit]

    yosys_exe = find_yosys_binary()
    env = prepare_yosys_env(yosys_exe)

    print("=================================================================")
    print(" MasterRTL: Multi-Representation Graph Generation Pipeline")
    print(f" Source: {input_dir} ({len(verilog_files)} designs)")
    print(f" Target: {output_dir}")
    print(" Types:  SOG, AST, AIG, Netlist, Timing DAG, CDFG")
    print("=================================================================")

    start_time = time.perf_counter()
    all_results: dict[str, dict[str, Any]] = {}
    success_count = 0
    fail_count = 0

    for idx, v_file in enumerate(verilog_files, 1):
        t0 = time.perf_counter()
        print(f"[{idx}/{len(verilog_files)}] Processing {v_file.name}...", end="", flush=True)
        try:
            res = process_single_design(v_file, output_dir, stage_dir, yosys_exe, env)
            all_results[v_file.stem] = res
            elapsed = time.perf_counter() - t0
            sog_nodes = res["sog"]["num_nodes"]
            ast_nodes = res["ast"]["num_nodes"]
            aig_nodes = res["aig"]["num_nodes"]
            net_nodes = res["netlist"]["num_nodes"]
            print(f" SUCCESS | AST={ast_nodes} -> SOG={sog_nodes} -> AIG={aig_nodes} -> Netlist={net_nodes} | {elapsed:.2f}s")
            success_count += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f" FAILED | {exc} | {elapsed:.2f}s")
            fail_count += 1

    # Generate comparison reports
    print("\nGenerating cross-representation comparison reports...")
    build_comparison_reports(all_results, output_dir)

    total_elapsed = time.perf_counter() - start_time
    print("\nBatch Graph Generation Summary:")
    print(f"  Total Scanned: {len(verilog_files)}")
    print(f"  Succeeded:     {success_count}")
    print(f"  Failed:        {fail_count}")
    print(f"  Total Runtime: {total_elapsed:.2f}s")
    print(f"  Reports saved in: {output_dir}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
