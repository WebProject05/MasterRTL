"""RTL-Stage Area Modeling Framework for MasterRTL.

Corresponds to Section II-D and Table II in:
"Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques"
(IEEE TCAD 2025).

Architecture:
1. Sequential area calculation:
   Area_seq = N_registers * Area_DFF (NanGate 45nm standard cell DFF area = 4.522 um^2)
2. Combinational area prediction:
   XGBoost Regressor (45 estimators, maximum depth of 12)
3. Total gate area:
   Area_total = Area_seq + Area_comb
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    HAS_XGBOOST = False

DFF_STANDARD_CELL_AREA = 4.522  # NanGate 45nm DFF_X1 area in um^2


def compute_sequential_area(num_registers: int, dff_cell_area: float = DFF_STANDARD_CELL_AREA) -> float:
    """Calculate sequential gate area directly from register count.

    Paper Section II-D: Product of total register count in SOG and area of standard DFF cell.
    """
    return float(num_registers) * float(dff_cell_area)


def extract_combinational_area_features(stats: dict[str, Any]) -> list[float]:
    """Extract feature vector for combinational area prediction.

    Features:
    1. Preliminary combinational area from SOG operator weights
    2. Count of combinational gates (num_comb_nodes)
    3. Count of AND operators
    4. Count of OR operators
    5. Count of XOR operators
    6. Count of NOT operators
    7. Count of MUX operators
    8. Total nodes
    9. Total edges
    10. Graph density
    11. Mean fan-out
    12. Max fan-out
    13. Mean fan-in
    14. Max fan-in
    """
    return [
        float(stats.get("est_combinational_area", 0.0)),
        float(stats.get("num_comb_nodes", 0)),
        float(stats.get("num_and", 0)),
        float(stats.get("num_or", 0)),
        float(stats.get("num_xor", 0)),
        float(stats.get("num_not", 0)),
        float(stats.get("num_mux", 0)),
        float(stats.get("total_nodes", 0)),
        float(stats.get("total_edges", 0)),
        float(stats.get("density", 0.0)),
        float(stats.get("mean_fanout", 1.0)),
        float(stats.get("max_fanout", 1.0)),
        float(stats.get("mean_fanin", 1.0)),
        float(stats.get("max_fanin", 1.0)),
    ]


class CombinationalAreaModel:
    """Combinational area prediction model refining SOG preliminary estimates to gate-level area.

    Table II: XGBoost Regressor, 45 estimators, maximum depth of 12.
    """

    def __init__(
        self,
        n_estimators: int = 45,
        max_depth: int = 12,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        if HAS_XGBOOST:
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        self.is_fitted = False

    def fit(self, X: np.ndarray | list[list[float]], y_comb_area: np.ndarray | list[float]) -> CombinationalAreaModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y_comb_area, dtype=np.float32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if not self.is_fitted:
            # Fallback to preliminary analytical combinational area (column 0)
            return X_arr[:, 0]
        return self.model.predict(X_arr)


class AreaPPAEstimator:
    """Unified Area Estimator coordinating sequential exact calculation and combinational ML model."""

    def __init__(self, dff_cell_area: float = DFF_STANDARD_CELL_AREA) -> None:
        self.dff_cell_area = dff_cell_area
        self.comb_model = CombinationalAreaModel()

    def fit(
        self,
        design_stats_list: list[dict[str, Any]],
        netlist_combinational_areas: list[float],
    ) -> None:
        feats = [extract_combinational_area_features(s) for s in design_stats_list]
        self.comb_model.fit(feats, netlist_combinational_areas)

    def predict_design(self, stats: dict[str, Any]) -> dict[str, Any]:
        num_regs = int(stats.get("num_registers", 0))
        seq_area = compute_sequential_area(num_regs, self.dff_cell_area)

        feats = extract_combinational_area_features(stats)
        comb_pred = float(self.comb_model.predict([feats])[0])
        total_area = seq_area + max(0.0, comb_pred)

        return {
            "sequential_area": round(seq_area, 2),
            "predicted_combinational_area": round(max(0.0, comb_pred), 2),
            "predicted_total_area": round(total_area, 2),
        }

