"""MasterRTL Design Optimization Advisor & What-If Analyzer.

Provides actionable, beginner-friendly RTL design advice and quantitative
what-if PPA projections to help engineers achieve timing closure, reduce power,
and optimize silicon area before committing to expensive commercial logic synthesis.
"""

from __future__ import annotations

import copy
from typing import Any

from src.models.master_rtl import MasterRTL


class PPAOptimizationAdvisor:
    """Intelligent PPA Optimization Advisor and What-If Projection Engine."""

    @staticmethod
    def diagnose_bottlenecks(
        stats: dict[str, Any],
        pred: dict[str, Any],
        clk_period: float = 1.0,
    ) -> list[dict[str, str]]:
        """Identify circuit structural bottlenecks causing timing violations, high power, or area bloat."""
        findings = []

        summary = pred.get("summary", {})
        wns = summary.get("predicted_wns", pred.get("wns", 0.0))
        tns = summary.get("predicted_tns", pred.get("tns", 0.0))
        power = summary.get("predicted_power_mW", pred.get("power_mW", 0.0))
        area = summary.get("predicted_area_um2", pred.get("area_um2", 0.0))

        crit_depth = stats.get("critical_path_delay", 0.0)
        num_dff = stats.get("num_registers", 0)
        num_comb = stats.get("num_comb_nodes", 0)
        toggle_rate = stats.get("mean_toggle_rate", 0.0)

        # 1. Timing Bottleneck Diagnosis
        if wns < 0:
            findings.append({
                "category": "TIMING",
                "severity": "CRITICAL",
                "issue": f"Setup Timing Violation: WNS is {wns:.3f} ns (TNS: {tns:.3f} ns)",
                "cause": (
                    f"Combinational critical path depth is {crit_depth:.1f} logic units, "
                    f"which exceeds the {clk_period:.3f} ns clock period timing budget."
                ),
            })
        else:
            findings.append({
                "category": "TIMING",
                "severity": "OPTIMAL",
                "issue": f"Timing Met: WNS has +{wns:.3f} ns positive slack margin",
                "cause": "Critical path propagation delay is within the target clock cycle budget.",
            })

        # 2. Power Consumption Diagnosis
        if toggle_rate >= 0.05 or power > 0.8:
            findings.append({
                "category": "POWER",
                "severity": "WARNING",
                "issue": f"Elevated Switching Power ({power:.3f} mW)",
                "cause": (
                    f"Mean toggle rate is {toggle_rate:.3f} transitions/cycle across {num_dff} registers. "
                    "Un-gated flip-flops toggle continuously during non-active cycles."
                ),
            })
        else:
            findings.append({
                "category": "POWER",
                "severity": "OPTIMAL",
                "issue": f"Low Power Dissipation ({power:.3f} mW)",
                "cause": "Circuit exhibits low switching activity and modest leakage current.",
            })

        # 3. Area / Structural Ratio Diagnosis
        if num_dff > 0:
            seq_ratio = (num_dff * 4.522) / area if area > 0 else 0
            if seq_ratio > 0.60:
                findings.append({
                    "category": "AREA",
                    "severity": "INFO",
                    "issue": f"Sequential-Dominated Floorplan ({seq_ratio*100:.1f}% DFF area)",
                    "cause": f"{num_dff} DFF registers contribute the majority of standard cell silicon area.",
                })
            else:
                findings.append({
                    "category": "AREA",
                    "severity": "INFO",
                    "issue": f"Combinational-Dominated Floorplan ({(1-seq_ratio)*100:.1f}% logic area)",
                    "cause": f"{num_comb} combinational gates form the main area and routing footprint.",
                })

        return findings

    @staticmethod
    def generate_recommendations(
        stats: dict[str, Any],
        pred: dict[str, Any],
        clk_period: float = 1.0,
    ) -> list[dict[str, str]]:
        """Generate concrete, beginner-friendly RTL optimization techniques."""
        recs = []

        summary = pred.get("summary", {})
        wns = summary.get("predicted_wns", pred.get("wns", 0.0))
        crit_depth = stats.get("critical_path_delay", 0.0)
        toggle_rate = stats.get("mean_toggle_rate", 0.0)

        # Rec 1: Pipelining
        if wns < 0 or crit_depth > 22.0:
            recs.append({
                "id": "REC_PIPELINE",
                "title": "Insert 1-Stage RTL Pipelining (Register Retiming)",
                "description": (
                    "Partition deep combinational logic chains into 2 balanced pipeline stages. "
                    "This cuts topological depth by ~45-50%, resolving setup timing violations."
                ),
                "expected_gain": "WNS improves by +0.30 to +0.45 ns; closes timing with positive slack margin.",
                "tradeoff": "Adds ~10-15% flip-flop area and 1 cycle of functional latency.",
            })

        # Rec 2: Clock Gating
        if toggle_rate >= 0.04:
            recs.append({
                "id": "REC_CLOCK_GATING",
                "title": "Apply Fine-Grained Clock Gating (ICG)",
                "description": (
                    "Insert integrated clock gating cells (ICGs) for registers with condition enable signals "
                    "to shut down clock distribution when register values do not change."
                ),
                "expected_gain": "Reduces dynamic switching power by 20% to 30%.",
                "tradeoff": "Negligible area overhead (< 1% for gating latch logic).",
            })

        # Rec 3: Tree Balancing
        if crit_depth > 25.0:
            recs.append({
                "id": "REC_TREE_BALANCING",
                "title": "Restructure Cascaded Logic to Balanced Trees (Kogge-Stone / CLA)",
                "description": (
                    "Replace ripple-carry or serial cascaded multiplexer chains with balanced tree structures. "
                    "Reduces delay complexity from linear O(N) to logarithmic O(log N)."
                ),
                "expected_gain": "Reduces combinational path delay by 30-40% without increasing latency.",
                "tradeoff": "Requires minor RTL datapath refactoring.",
            })

        # If timing is already met and power is low
        if not recs:
            recs.append({
                "id": "REC_SIZING",
                "title": "Downsize Non-Critical Path Gates for Area Minimization",
                "description": (
                    "Current design comfortably meets all timing deadlines. High-drive cells on non-critical "
                    "paths can be swapped for minimal-area standard cells."
                ),
                "expected_gain": "Reduces total cell area by 5-10% without violating timing.",
                "tradeoff": "None; timing slack is maintained.",
            })

        return recs

    @staticmethod
    def simulate_what_if(
        model: MasterRTL,
        stats: dict[str, Any],
        paths: list[dict[str, Any]],
        clk_period: float = 1.0,
        scenario: str = "combined_opt",
    ) -> dict[str, Any]:
        """Simulate the quantitative PPA impact of applying optimization recommendations.

        Supported scenarios:
        - 'pipelining': 1-stage pipeline insertion (halves depth, adds DFFs)
        - 'clock_gating': fine-grained clock gating (reduces toggle rate by 28%)
        - 'combined_opt': both pipelining and clock gating applied simultaneously
        """
        mod_stats = copy.deepcopy(stats)
        mod_paths = copy.deepcopy(paths)

        crit_depth = stats.get("critical_path_delay", 20.0)
        mean_depth = stats.get("mean_path_delay", 10.0)
        num_dff = stats.get("num_registers", 10)
        num_comb = stats.get("num_comb_nodes", 50)
        toggle = stats.get("mean_toggle_rate", 0.08)

        if scenario in {"pipelining", "combined_opt"}:
            # Pipelining: cuts critical depth by ~48%, adds pipeline registers
            mod_stats["critical_path_delay"] = max(4.0, crit_depth * 0.52)
            mod_stats["mean_path_delay"] = max(3.0, mean_depth * 0.60)
            added_regs = max(4, int(num_dff * 0.16))
            mod_stats["num_registers"] = num_dff + added_regs
            mod_stats["total_nodes"] = stats.get("total_nodes", 100) + added_regs
            mod_stats["num_comb_nodes"] = num_comb + max(2, int(added_regs * 0.25))

            # Scale path delays in path list
            for p in mod_paths:
                if "delay" in p:
                    p["delay"] = max(3.0, p["delay"] * 0.52)

        if scenario in {"clock_gating", "combined_opt"}:
            # Clock gating: reduces effective toggle activity by ~28%
            mod_stats["mean_toggle_rate"] = max(0.015, toggle * 0.72)
            mod_stats["weighted_toggle_rate"] = max(0.012, stats.get("weighted_toggle_rate", 0.06) * 0.70)

        # Run prediction through MasterRTL's ML models
        opt_pred = model.predict(mod_stats, mod_paths, clk_period=clk_period)

        # Baseline prediction
        base_pred = model.predict(stats, paths, clk_period=clk_period)

        base_wns = base_pred.get("wns", 0.0)
        base_tns = base_pred.get("tns", 0.0)
        base_pwr = base_pred.get("power_mW", 0.0)
        base_area = base_pred.get("area_um2", 0.0)

        opt_wns = opt_pred.get("wns", 0.0)
        opt_tns = opt_pred.get("tns", 0.0)
        opt_pwr = opt_pred.get("power_mW", 0.0)
        opt_area = opt_pred.get("area_um2", 0.0)

        delta_wns = opt_wns - base_wns
        delta_pwr_pct = ((opt_pwr - base_pwr) / base_pwr * 100.0) if base_pwr > 0 else 0.0
        delta_area_pct = ((opt_area - base_area) / base_area * 100.0) if base_area > 0 else 0.0

        return {
            "scenario": scenario,
            "baseline": {
                "wns": base_wns,
                "tns": base_tns,
                "power_mW": base_pwr,
                "area_um2": base_area,
            },
            "projected": {
                "wns": opt_wns,
                "tns": opt_tns,
                "power_mW": opt_pwr,
                "area_um2": opt_area,
            },
            "improvement": {
                "delta_wns": round(delta_wns, 3),
                "wns_status_before": "VIOLATED" if base_wns < 0 else "MET",
                "wns_status_after": "VIOLATED" if opt_wns < 0 else "MET (CLOSURE ACHIEVED)",
                "power_savings_pct": round(-delta_pwr_pct, 1),
                "area_overhead_pct": round(delta_area_pct, 1),
            },
            "optimized_stats": mod_stats,
        }

    @classmethod
    def evaluate_design(
        cls,
        model: MasterRTL,
        stats: dict[str, Any],
        paths: list[dict[str, Any]],
        pred: dict[str, Any],
        clk_period: float = 1.0,
    ) -> dict[str, Any]:
        """Comprehensive evaluation generating bottlenecks, recommendations, and what-if simulation."""
        findings = cls.diagnose_bottlenecks(stats, pred, clk_period=clk_period)
        recs = cls.generate_recommendations(stats, pred, clk_period=clk_period)
        whatif_combined = cls.simulate_what_if(model, stats, paths, clk_period=clk_period, scenario="combined_opt")
        whatif_pipe = cls.simulate_what_if(model, stats, paths, clk_period=clk_period, scenario="pipelining")
        whatif_cg = cls.simulate_what_if(model, stats, paths, clk_period=clk_period, scenario="clock_gating")

        return {
            "findings": findings,
            "recommendations": recs,
            "what_if": {
                "combined": whatif_combined,
                "pipelining": whatif_pipe,
                "clock_gating": whatif_cg,
            },
        }

    @staticmethod
    def format_advisory_table(advisory_res: dict[str, Any], top_module: str = "design") -> str:
        """Format the optimization advisory report into a clean, easy-to-read ASCII table."""
        lines = []
        w = 76
        sep = "=" * w
        sub_sep = "-" * w

        lines.append("\n" + sep)
        lines.append(f"  MasterRTL Smart PPA Optimization Advisor: {top_module}")
        lines.append(sep)

        # 1. Bottleneck Findings
        lines.append("  1. BOTTLENECK DIAGNOSTICS & ROOT CAUSES")
        lines.append(sub_sep)
        for f in advisory_res.get("findings", []):
            sev = f"[{f['severity']}]".ljust(11)
            lines.append(f"  * {sev} {f['issue']}")
            lines.append(f"               --> {f['cause']}")
        lines.append(sub_sep)

        # 2. Recommendations
        lines.append("  2. ACTIONABLE RTL OPTIMIZATION STRATEGIES")
        lines.append(sub_sep)
        for i, r in enumerate(advisory_res.get("recommendations", []), 1):
            lines.append(f"  ({i}) {r['title']}")
            lines.append(f"      Action : {r['description']}")
            lines.append(f"      Gain   : {r['expected_gain']}")
            lines.append(f"      Impact : {r['tradeoff']}")
        lines.append(sub_sep)

        # 3. What-If Comparison
        comb = advisory_res.get("what_if", {}).get("combined", {})
        if comb:
            base = comb.get("baseline", {})
            proj = comb.get("projected", {})
            imp = comb.get("improvement", {})

            lines.append("  3. WHAT-IF PROJECTED PPA (WITH PIPELINING + CLOCK GATING)")
            lines.append(sub_sep)
            lines.append(f"  {'METRIC':<18} {'BASELINE':<15} {'PROJECTED':<15} {'IMPROVEMENT / DELTA'}")
            lines.append(f"  {'-'*16:<18} {'-'*13:<15} {'-'*13:<15} {'-'*24}")

            b_wns = base.get("wns", 0.0)
            p_wns = proj.get("wns", 0.0)
            d_wns = imp.get("delta_wns", 0.0)
            wns_note = "CLOSURE MET" if p_wns >= 0 else "Slack Improved"
            lines.append(f"  {'Timing (WNS)':<18} {b_wns:>7.3f} ns     {p_wns:>7.3f} ns     {d_wns:>+7.3f} ns ({wns_note})")

            b_tns = base.get("tns", 0.0)
            p_tns = proj.get("tns", 0.0)
            tns_note = "All Violations Fixed" if p_tns >= 0 else f"{abs(p_tns):.3f} ns left"
            lines.append(f"  {'Timing (TNS)':<18} {b_tns:>7.3f} ns     {p_tns:>7.3f} ns     {tns_note}")

            b_pwr = base.get("power_mW", 0.0)
            p_pwr = proj.get("power_mW", 0.0)
            pwr_sav = imp.get("power_savings_pct", 0.0)
            lines.append(f"  {'Total Power':<18} {b_pwr:>7.3f} mW     {p_pwr:>7.3f} mW     {pwr_sav:>+5.1f}% power saved")

            b_area = base.get("area_um2", 0.0)
            p_area = proj.get("area_um2", 0.0)
            area_oh = imp.get("area_overhead_pct", 0.0)
            lines.append(f"  {'Silicon Area':<18} {b_area:>7.1f} um^2   {p_area:>7.1f} um^2   {area_oh:>+5.1f}% (pipeline DFFs)")

        lines.append(sep + "\n")
        return "\n".join(lines)
