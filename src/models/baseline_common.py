"""Common utilities, dataset loaders, and graph feature extractors for baseline models.

Supports:
- ICCAD'22 [21] (Sengupta et al.) register-tree extraction with duplicate logic counting
- ISCA'22 [22] (Xu et al. SNS) pseudo-path / sequence extraction (lacking fanout)
- Unified dataset pairing between AST (word-level) and SOG (bit-level) representations
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

from src.models.master_rtl import (
    compute_mae,
    compute_mape,
    compute_pearson_r,
    compute_rrse,
    evaluate_metrics,
    load_ppa_dataset,
)


def load_ast_dataset(
    ast_dir: str | Path = "data/graphs/ast",
    sog_dir: str | Path = "data/sog_graphs",
    labels_dir: str | Path = "data/ppa_labels",
) -> list[dict[str, Any]]:
    """Load AST word-level graph dataset, aligned with ground-truth PPA labels from SOG."""
    ast_path = Path(ast_dir).expanduser().resolve()
    sog_dataset = load_ppa_dataset(sog_dir=sog_dir, labels_dir=labels_dir)
    sog_map = {d["name"]: d for d in sog_dataset}

    ast_dataset: list[dict[str, Any]] = []
    stats_files = sorted(ast_path.glob("*_stats.json"))

    for sf in stats_files:
        stem = sf.stem.replace("_stats", "")
        if stem not in sog_map:
            continue

        with sf.open("r", encoding="utf-8") as handle:
            ast_stats = json.load(handle)

        graph_file = ast_path / f"{stem}.pkl"
        graph = None
        if graph_file.is_file():
            try:
                with graph_file.open("rb") as handle:
                    graph = pickle.load(handle)
            except Exception:
                graph = None

        sog_entry = sog_map[stem]
        labels = sog_entry["labels"]
        sog_stats = sog_entry["stats"]
        sog_paths = sog_entry.get("paths", [])

        if graph is not None:
            reg_bits = sum(d.get("width", 1) for n, d in graph.nodes(data=True) if d.get("is_sequential", False) or "DFF" in str(d.get("op", "")))
            tot_bits = sum(d.get("width", 1) for n, d in graph.nodes(data=True))
            ast_stats["total_reg_bits"] = reg_bits
            ast_stats["total_op_bits"] = tot_bits
        else:
            ast_stats["total_reg_bits"] = sog_stats.get("num_registers", 1)
            ast_stats["total_op_bits"] = sog_stats.get("total_nodes", 10)

        ast_dataset.append({
            "name": stem,
            "representation": "ast",
            "stats": ast_stats,
            "sog_stats": sog_stats,
            "graph": graph,
            "paths": sog_paths,
            "labels": labels,
        })

    return ast_dataset


# =========================================================================
# ICCAD'22 [21] Feature Extraction: Register Trees with Duplicate Counting
# =========================================================================

def extract_iccad21_tree_features(
    stats: dict[str, Any],
    graph: nx.DiGraph | None = None,
    representation: str = "ast",
) -> list[float]:
    """Extract register-tree features matching ICCAD'22 [21].

    As noted in Section I and Section II-A:
    [21] traverses backward from each register through combinational logic
    to build register trees. Logic operators driving multiple endpoints
    are duplicated in the count, representing early presynthesis tree scale.
    """
    if graph is not None and isinstance(graph, nx.DiGraph) and graph.number_of_nodes() > 0:
        # Identify registers
        registers = [
            n for n, d in graph.nodes(data=True)
            if d.get("is_sequential", False)
            or d.get("node_type") == "REGISTER"
            or "DFF" in str(d.get("op", ""))
        ]
        num_regs = max(1, len(registers))

        # Backward traversal from each register to accumulate tree nodes with duplication
        tree_node_counts = []
        tree_depths = []
        op_counts: dict[str, int] = {}

        for reg in registers:
            visited = set()
            queue = [(reg, 0)]
            depth = 0
            count = 0
            while queue:
                curr, d = queue.pop(0)
                depth = max(depth, d)
                count += 1
                op = str(graph.nodes[curr].get("op", "UNKNOWN")).upper()
                op_counts[op] = op_counts.get(op, 0) + 1

                for pred in graph.predecessors(curr):
                    p_data = graph.nodes[pred]
                    is_p_reg = (
                        p_data.get("is_sequential", False)
                        or p_data.get("node_type") == "REGISTER"
                        or "DFF" in str(p_data.get("op", ""))
                    )
                    # Stop at boundary registers (combinational logic tree)
                    if not is_p_reg and pred not in visited:
                        visited.add(pred)
                        queue.append((pred, d + 1))

            tree_node_counts.append(count)
            tree_depths.append(depth)

        total_tree_nodes = float(sum(tree_node_counts)) if tree_node_counts else float(graph.number_of_nodes())
        avg_tree_depth = float(np.mean(tree_depths)) if tree_depths else 1.0
        max_tree_depth = float(max(tree_depths)) if tree_depths else 1.0

        num_and = float(op_counts.get("AND", 0))
        num_or = float(op_counts.get("OR", 0))
        num_xor = float(op_counts.get("XOR", 0))
        num_not = float(op_counts.get("NOT", 0))
        num_mux = float(op_counts.get("MUX", 0))
        num_add = float(op_counts.get("ADD", 0) + op_counts.get("SUB", 0))
        num_mul = float(op_counts.get("MUL", 0))
    else:
        # Fallback to stats dictionary with empirical duplication multiplier
        num_regs = float(stats.get("num_registers", 1))
        tot_nodes = float(stats.get("total_nodes", stats.get("num_nodes", 10)))
        # Duplication factor in register trees: logic sharing multiplier ~ 1.8x
        total_tree_nodes = tot_nodes * (1.8 if representation == "ast" else 1.4)
        avg_tree_depth = float(stats.get("max_depth", stats.get("comb_depth", 10))) * 0.7
        max_tree_depth = float(stats.get("max_depth", stats.get("comb_depth", 10)))
        num_and = float(stats.get("num_and", stats.get("top_operators", {}).get("AND", 0))) * 1.5
        num_or = float(stats.get("num_or", stats.get("top_operators", {}).get("OR", 0))) * 1.5
        num_xor = float(stats.get("num_xor", stats.get("top_operators", {}).get("XOR", 0))) * 1.5
        num_not = float(stats.get("num_not", stats.get("top_operators", {}).get("NOT", 0))) * 1.5
        num_mux = float(stats.get("num_mux", stats.get("top_operators", {}).get("MUX", 0))) * 1.5
        num_add = float(stats.get("top_operators", {}).get("ADD", 0)) * 1.5
        num_mul = float(stats.get("top_operators", {}).get("MUL", 0)) * 1.5

    density = float(stats.get("density", 0.01))
    pi_count = float(stats.get("num_primary_inputs", 10))
    po_count = float(stats.get("num_primary_outputs", 10))
    tot_reg_bits = float(stats.get("total_reg_bits", float(num_regs)))
    tot_op_bits = float(stats.get("total_op_bits", total_tree_nodes))

    return [
        float(num_regs),
        tot_reg_bits,
        total_tree_nodes,
        tot_op_bits,
        total_tree_nodes / max(1.0, float(num_regs)),  # avg nodes per tree
        avg_tree_depth,
        max_tree_depth,
        num_and,
        num_or,
        num_xor,
        num_not,
        num_mux,
        num_add,
        num_mul,
        density,
        pi_count,
        po_count,
    ]


# =========================================================================
# ISCA'22 [22] Feature Extraction: Path Sequence (Without Fanout)
# =========================================================================

def extract_isca22_path_features(path_record: dict[str, Any]) -> list[float]:
    """Extract path-level feature vector matching ISCA'22 [22] (SNS).

    As emphasized in Paper Section II-B:
    ISCA'22 [22] treats each path as an operator sequence and
    CRUCIALLY LACKS fan-out information in its features.
    """
    length = float(path_record.get("length", len(path_record.get("path", []))))
    op_counts = path_record.get("op_counts", {})
    count_and = float(op_counts.get("AND", 0))
    count_or = float(op_counts.get("OR", 0))
    count_xor = float(op_counts.get("XOR", 0))
    count_not = float(op_counts.get("NOT", 0))
    count_mux = float(op_counts.get("MUX", 0))
    count_dff = float(op_counts.get("DFF", 0))
    # ISCA'22 omits accumulated fanout and liberty cell RC delay
    return [
        length,
        count_and,
        count_or,
        count_xor,
        count_not,
        count_mux,
        count_dff,
    ]


def extract_isca22_design_features(stats: dict[str, Any], representation: str = "ast") -> list[float]:
    """Extract direct design-level features for ISCA'22 power and area models.

    ISCA'22 uses direct single-stage regression without toggle rate propagation
    and without separating sequential DFF area from combinational area.
    """
    tot_nodes = float(stats.get("total_nodes", stats.get("num_nodes", 10)))
    tot_edges = float(stats.get("total_edges", stats.get("num_edges", 10)))
    regs = float(stats.get("num_registers", 1))
    comb_depth = float(stats.get("comb_depth", stats.get("max_depth", 10)))
    density = float(stats.get("density", 0.01))

    num_and = float(stats.get("num_and", stats.get("top_operators", {}).get("AND", 0)))
    num_or = float(stats.get("num_or", stats.get("top_operators", {}).get("OR", 0)))
    num_xor = float(stats.get("num_xor", stats.get("top_operators", {}).get("XOR", 0)))
    num_not = float(stats.get("num_not", stats.get("top_operators", {}).get("NOT", 0)))
    num_mux = float(stats.get("num_mux", stats.get("top_operators", {}).get("MUX", 0)))

    tot_reg_bits = float(stats.get("total_reg_bits", stats.get("num_registers", 1)))
    tot_op_bits = float(stats.get("total_op_bits", stats.get("total_nodes", stats.get("num_nodes", 10))))

    return [
        tot_nodes,
        tot_edges,
        regs,
        tot_reg_bits,
        tot_op_bits,
        comb_depth,
        density,
        num_and,
        num_or,
        num_xor,
        num_not,
        num_mux,
    ]
