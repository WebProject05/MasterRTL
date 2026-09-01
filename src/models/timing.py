"""RTL-Stage Timing Modeling Framework for MasterRTL.

Corresponds to Section II-B and Table II in:
"Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques"
(IEEE TCAD 2025).

Architecture:
1. Analytical linear RC node delay model on SOG (Stage 1)
2. Critical path identification and mapping between SOG R and netlist G (Stage 2)
3. Path-level delay model f_path_t: Random Forest Regressor (80 trees, max_depth=20) (Stage 3)
4. Path-level delay inference for TNS^R and WNS^R across endpoint registers (Stage 4)
5. Design-level calibration: XGBoost Regressor (45 trees, max_depth=8) (Stage 5)
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestRegressor


def extract_path_features(path_record: dict[str, Any]) -> list[float]:
    """Extract path-level feature vector from an SOG critical path dictionary.

    Features:
    1. Total node count (path length)
    2. Count of AND operators
    3. Count of OR operators
    4. Count of XOR operators
    5. Count of NOT operators
    6. Count of MUX operators
    7. Count of DFF operators
    8. Accumulated analytical node delay
    9. Accumulated node fan-out sum
    """
    length = float(path_record.get("length", len(path_record.get("path", []))))
    op_counts = path_record.get("op_counts", {})
    count_and = float(op_counts.get("AND", 0))
    count_or = float(op_counts.get("OR", 0))
    count_xor = float(op_counts.get("XOR", 0))
    count_not = float(op_counts.get("NOT", 0))
    count_mux = float(op_counts.get("MUX", 0))
    count_dff = float(op_counts.get("DFF", 0))
    delay = float(path_record.get("delay", 0.0))
    accum_fanout = float(path_record.get("accumulated_fanout", 0.0))

    return [
        length,
        count_and,
        count_or,
        count_xor,
        count_not,
        count_mux,
        count_dff,
        delay,
        accum_fanout,
    ]


class PathLevelTimingModel:
    """Path-level delay model f_path_t predicting netlist path delay from SOG path features.

    Table II: Random Forest Regressor, 80 estimators, maximum depth of 20.
    """

    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray | list[list[float]], y: np.ndarray | list[float]) -> PathLevelTimingModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        if not self.is_fitted:
            # Fallback heuristic: analytical accumulated delay
            X_arr = np.asarray(X, dtype=np.float32)
            if X_arr.ndim == 1:
                return np.array([X_arr[7]], dtype=np.float32)
            return X_arr[:, 7]
        return self.model.predict(np.asarray(X, dtype=np.float32))

    def predict_path(self, path_record: dict[str, Any]) -> float:
        feats = extract_path_features(path_record)
        pred = self.predict([feats])
        return float(pred[0])


def infer_design_timing_paths(
    path_model: PathLevelTimingModel,
    critical_paths: list[dict[str, Any]],
    clk_period: float = 1.0,
) -> dict[str, Any]:
    """Inference on the N endpoint critical paths of a design (Stage 4 in Paper).

    Computes:
    - Path delays d_j = f_path_t(P^R_{*->j})
    - Path slacks = clk_period - d_j
    - WNS^R = min(clk_period - d_j)
    - TNS^R = sum(min(0, clk_period - d_j))
    - Slack distribution percentiles (worst 1% of critical paths): worst, p10, p50, p90
    """
    if not critical_paths:
        return {
            "wns_r": 0.0,
            "tns_r": 0.0,
            "slack_worst": 0.0,
            "slack_p10": 0.0,
            "slack_p50": 0.0,
            "slack_p90": 0.0,
            "pred_path_delays": [],
            "pred_path_slacks": [],
        }

    feats_list = [extract_path_features(p) for p in critical_paths]
    pred_delays = path_model.predict(feats_list)
    slacks = [clk_period - float(d) for d in pred_delays]

    wns_r = min(slacks)
    tns_r = sum(min(0.0, s) for s in slacks)

    # Focus on the worst paths (up to top 1% or worst 20)
    sorted_slacks = sorted(slacks)
    worst_count = max(1, int(math.ceil(0.01 * len(sorted_slacks))))
    worst_subset = sorted_slacks[:max(worst_count, min(10, len(sorted_slacks)))]

    def percentile(vals: list[float], pct: float) -> float:
        if not vals:
            return 0.0
        idx = int(math.ceil(pct / 100.0 * len(vals))) - 1
        return vals[max(0, min(idx, len(vals) - 1))]

    return {
        "wns_r": round(float(wns_r), 4),
        "tns_r": round(float(tns_r), 4),
        "slack_worst": round(float(worst_subset[0]), 4),
        "slack_p10": round(float(percentile(worst_subset, 10.0)), 4),
        "slack_p50": round(float(percentile(worst_subset, 50.0)), 4),
        "slack_p90": round(float(percentile(worst_subset, 90.0)), 4),
        "pred_path_delays": [round(float(d), 4) for d in pred_delays],
        "pred_path_slacks": [round(float(s), 4) for s in slacks],
    }


def extract_design_timing_calibration_features(
    stats: dict[str, Any],
    timing_inference: dict[str, Any],
) -> list[float]:
    """Build design-level feature vector for Stage 5 WNS/TNS calibration.

    Features:
    1. SOG graph features reflecting design scale:
       - total_nodes, total_edges, num_registers, num_comb_nodes
       - num_and, num_or, num_xor, num_not, num_mux
       - mean_fanout, max_fanout, density
    2. Estimated TNS^R and WNS^R from path-level model
    3. Slack distribution percentiles of top critical paths (worst, p10, p50, p90)
    """
    return [
        float(stats.get("total_nodes", 0)),
        float(stats.get("total_edges", 0)),
        float(stats.get("num_registers", 0)),
        float(stats.get("num_comb_nodes", 0)),
        float(stats.get("num_and", 0)),
        float(stats.get("num_or", 0)),
        float(stats.get("num_xor", 0)),
        float(stats.get("num_not", 0)),
        float(stats.get("num_mux", 0)),
        float(stats.get("mean_fanout", 1.0)),
        float(stats.get("max_fanout", 1.0)),
        float(stats.get("density", 0.0)),
        float(timing_inference.get("wns_r", 0.0)),
        float(timing_inference.get("tns_r", 0.0)),
        float(timing_inference.get("slack_worst", 0.0)),
        float(timing_inference.get("slack_p10", 0.0)),
        float(timing_inference.get("slack_p50", 0.0)),
        float(timing_inference.get("slack_p90", 0.0)),
    ]


class DesignLevelTimingCalibrationModel:
    """Stage 5 Calibration Model refining WNS and TNS predictions to gate-level netlist labels.

    Table II: XGBoost Regressor, 45 estimators, maximum depth of 8.
    """

    def __init__(
        self,
        n_estimators: int = 45,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        if HAS_XGBOOST:
            self.model_wns = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
            )
            self.model_tns = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            self.model_wns = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            self.model_tns = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray | list[list[float]],
        y_wns: np.ndarray | list[float],
        y_tns: np.ndarray | list[float],
    ) -> DesignLevelTimingCalibrationModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_wns_arr = np.asarray(y_wns, dtype=np.float32)
        y_tns_arr = np.asarray(y_tns, dtype=np.float32)

        self.model_wns.fit(X_arr, y_wns_arr)
        self.model_tns.fit(X_arr, y_tns_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float32)
        if not self.is_fitted:
            # Return uncalibrated path estimates (WNS^R, TNS^R at columns 12, 13)
            return X_arr[:, 12], X_arr[:, 13]
        pred_wns = self.model_wns.predict(X_arr)
        pred_tns = self.model_tns.predict(X_arr)
        return pred_wns, pred_tns


class TimingPPAEstimator:
    """Unified multistage timing estimator coordinating path modeling, inference, and calibration."""

    def __init__(self) -> None:
        self.path_model = PathLevelTimingModel()
        self.calib_model = DesignLevelTimingCalibrationModel()

    def fit_path_model(self, path_records: list[dict[str, Any]], netlist_delays: list[float]) -> None:
        feats = [extract_path_features(p) for p in path_records]
        self.path_model.fit(feats, netlist_delays)

    def fit_calibration_model(
        self,
        design_features: list[list[float]],
        target_wns: list[float],
        target_tns: list[float],
    ) -> None:
        self.calib_model.fit(design_features, target_wns, target_tns)

    def predict_design(
        self,
        stats: dict[str, Any],
        critical_paths: list[dict[str, Any]],
        clk_period: float = 1.0,
    ) -> dict[str, Any]:
        path_inference = infer_design_timing_paths(self.path_model, critical_paths, clk_period=clk_period)
        calib_feats = extract_design_timing_calibration_features(stats, path_inference)
        pred_wns, pred_tns = self.calib_model.predict([calib_feats])

        return {
            "wns_r": path_inference["wns_r"],
            "tns_r": path_inference["tns_r"],
            "predicted_wns": round(float(pred_wns[0]), 4),
            "predicted_tns": round(float(pred_tns[0]), 4),
            "slack_distribution": {
                "worst": path_inference["slack_worst"],
                "p10": path_inference["slack_p10"],
                "p50": path_inference["slack_p50"],
                "p90": path_inference["slack_p90"],
            },
        }
