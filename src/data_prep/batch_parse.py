"""Batch utility for synthesizing a directory of Verilog files with Yosys and generating SOG graphs.

Performs:
1. Technology-independent generic synthesis to single-bit gates and DFFs via Yosys.
2. Construction of Simple Operator Graph (SOG) representations (.pkl).
3. Critical path extraction and feature computation (_features.pkl).
4. Boolean toggle-rate propagation.
5. Aggregation of topological, timing, power, and area statistics into CSV, JSON, and Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

from src.data_prep.sog_converter import (
    build_sog_from_json,
    save_graph_artifacts,
)
from src.data_prep.yosys_synthesis import run_yosys_synthesis

logger = logging.getLogger(__name__)


def iter_verilog_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of Verilog source files under the given directory."""
    patterns = ("*.v", "*.sv")
    if recursive:
        matches = [path for pattern in patterns for path in input_dir.rglob(pattern)]
    else:
        matches = [path for pattern in patterns for path in input_dir.glob(pattern)]
    return sorted({path.resolve() for path in matches if not path.name.endswith(".synth.v")})


def export_aggregated_reports(
    records: list[dict[str, Any]],
    report_dir: Path,
) -> dict[str, Path]:
    """Export summary statistics across all designs to CSV, JSON, and Markdown formats."""
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "sog_statistics.csv"
    json_path = report_dir / "sog_statistics.json"
    md_path = report_dir / "sog_summary.md"

    if not records:
        return {"csv": csv_path, "json": json_path, "md": md_path}

    fieldnames = list(records[0].keys())

    # 1. Export CSV
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 2. Export JSON
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    # 3. Export Markdown Table
    md_lines = [
        "# MasterRTL Simple Operator Graph (SOG) Benchmark Statistics\n",
        f"Generated: {len(records)} designs processed successfully.\n",
        "| Design | Nodes | Edges | Regs (DFF) | Comb Gates | AND | OR | XOR | NOT | MUX | Crit Delay | Mean Delay | Weighted Toggle | Total Area |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        name = r.get("module_name", "unknown")
        nodes = r.get("total_nodes", 0)
        edges = r.get("total_edges", 0)
        regs = r.get("num_registers", 0)
        comb = r.get("num_comb_nodes", 0)
        n_and = r.get("num_and", 0)
        n_or = r.get("num_or", 0)
        n_xor = r.get("num_xor", 0)
        n_not = r.get("num_not", 0)
        n_mux = r.get("num_mux", 0)
        crit_d = r.get("critical_path_delay", 0.0)
        mean_d = r.get("mean_path_delay", 0.0)
        wt_tog = r.get("weighted_toggle_rate", 0.0)
        tot_area = r.get("est_total_area", 0.0)
        md_lines.append(
            f"| `{name}` | {nodes} | {edges} | {regs} | {comb} | {n_and} | {n_or} | {n_xor} | {n_not} | {n_mux} | {crit_d:.2f} | {mean_d:.2f} | {wt_tog:.2f} | {tot_area:.1f} |"
        )
    md_lines.append("\n*Area approximations based on NanGate 45nm standard cell footprints.*\n")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {"csv": csv_path, "json": json_path, "md": md_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch synthesize Verilog files with Yosys, generate Simple Operator Graphs (SOG), "
            "and record comprehensive timing, toggle, and structural statistics."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing Verilog inputs.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated synthesis artifacts. Defaults to data/yosys_synth.",
    )
    parser.add_argument(
        "--sog-dir",
        type=Path,
        default=None,
        help="Directory for generated SOG graph artifacts (.pkl, stats). Defaults to data/sog_graphs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for summary reports (CSV, JSON, Markdown). Defaults to data/reports.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively for Verilog files.",
    )
    parser.add_argument(
        "--yosys-bin",
        default="yosys",
        help="Yosys executable to invoke (default: yosys).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    synth_base = (args.output_dir or Path("data/yosys_synth")).expanduser().resolve()
    sog_base = (args.sog_dir or Path("data/sog_graphs")).expanduser().resolve()
    report_base = (args.report_dir or Path("data/reports")).expanduser().resolve()

    synth_base.mkdir(parents=True, exist_ok=True)
    sog_base.mkdir(parents=True, exist_ok=True)
    report_base.mkdir(parents=True, exist_ok=True)

    files = iter_verilog_files(input_dir, recursive=args.recursive)
    if not files:
        print(f"No Verilog files found under {input_dir}.")
        return 1

    print(f"Found {len(files)} Verilog designs to process.\n")

    total_start = time.perf_counter()
    total = 0
    succeeded = 0
    failed = 0
    all_statistics: list[dict[str, Any]] = []

    for design_path in files:
        total += 1
        started = time.perf_counter()
        rel_path = design_path.name
        print(f"[{total}/{len(files)}] Processing {rel_path}...")

        try:
            # 1. Yosys Generic Synthesis
            synth_res = run_yosys_synthesis(
                input_path=design_path,
                output_dir=synth_base,
                yosys_bin=args.yosys_bin,
            )

            # 2. SOG Graph Generation
            json_file = Path(synth_res["json_ast"])
            graph = build_sog_from_json(json_file)
            artifacts = save_graph_artifacts(graph, sog_base, design_path.stem)

            # 3. Read statistics
            with open(artifacts["stats"], "r", encoding="utf-8") as h:
                design_stats = json.load(h)
            all_statistics.append(design_stats)

            elapsed = time.perf_counter() - started
            print(
                f"  SUCCESS: top={synth_res['top_module']} | "
                f"nodes={design_stats['total_nodes']} | "
                f"regs={design_stats['num_registers']} | "
                f"comb={design_stats['num_comb_nodes']} | "
                f"crit_delay={design_stats['critical_path_delay']} | "
                f"elapsed={elapsed:.2f}s"
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            print(f"  ERROR: {design_path.name} | elapsed={elapsed:.2f}s | {exc}")
            logger.exception("Processing failed for %s", design_path)
            failed += 1

    total_elapsed = time.perf_counter() - total_start

    # 4. Export aggregated summary reports
    if all_statistics:
        reports = export_aggregated_reports(all_statistics, report_base)
        print("\nGenerated Aggregated Reports:")
        print(f"  CSV:      {reports['csv']}")
        print(f"  JSON:     {reports['json']}")
        print(f"  Markdown: {reports['md']}")

    print("\nBatch execution summary:")
    print(f"  Total scanned:   {total}")
    print(f"  Succeeded:       {succeeded}")
    print(f"  Failed:          {failed}")
    print(f"  Total runtime:   {total_elapsed:.2f}s")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
