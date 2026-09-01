"""RTL-Stage Transfer Modeling Framework for MasterRTL.

Corresponds to Section II-E and Table II in:
"Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques"
(IEEE TCAD 2025).

Architecture:
1. Layout stage transfer (post-synthesis -> post-placement PPA)
2. Technology node transfer (NanGate 45nm -> TSMC 22/28/40/65nm)
3. Process corner transfer (TYP -> MIN, MAX)
4. Model: XGBoost Regressor (15 estimators, maximum depth of 8)
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

# Default technology scaling coefficients relative to NanGate 45nm TYP
# Values derived from standard foundry characterization ratios in Table IV / Fig. 12
DEFAULT_TECH_SCALING = {
    "NanGate_45nm_TYP": {"delay": 1.00, "power": 1.00, "area": 1.00},
    "NanGate_45nm_MIN": {"delay": 0.65, "power": 1.35, "area": 1.00},
    "NanGate_45nm_MAX": {"delay": 1.45, "power": 0.75, "area": 1.00},
    "TSMC_65nm_TYP":    {"delay": 1.25, "power": 1.40, "area": 1.85},
    "TSMC_40nm_TYP":    {"delay": 0.88, "power": 0.82, "area": 0.78},
    "TSMC_28nm_TYP":    {"delay": 0.62, "power": 0.54, "area": 0.40},
    "TSMC_22nm_TYP":    {"delay": 0.51, "power": 0.42, "area": 0.30},
}


def extract_transfer_features(
    source_ppa: dict[str, float],
    stats: dict[str, Any],
    target_scale: dict[str, float] | None = None,
) -> list[float]:
    """Extract feature vector for layout and technology transfer.

    Features (Section II-E):
    1. Initial PPA predictions for source technology (WNS, TNS, Power, Area)
    2. Library scale factors (delay_scale, power_scale, area_scale)
    3. Scaled PPA metrics (source_ppa * scale)
    4. Design scale-related features (total_nodes, num_registers, num_comb_nodes, total_edges)
    """
    scale = target_scale or {"delay": 1.0, "power": 1.0, "area": 1.0}
    delay_s = float(scale.get("delay", 1.0))
    power_s = float(scale.get("power", 1.0))
    area_s = float(scale.get("area", 1.0))

    src_wns = float(source_ppa.get("wns", 0.0))
    src_tns = float(source_ppa.get("tns", 0.0))
    src_power = float(source_ppa.get("power", 0.0))
    src_area = float(source_ppa.get("area", 0.0))

    scaled_wns = src_wns * (1.0 / delay_s if delay_s > 0 else 1.0)
    scaled_tns = src_tns * (1.0 / delay_s if delay_s > 0 else 1.0)
    scaled_power = src_power * power_s
    scaled_area = src_area * area_s

    total_nodes = float(stats.get("total_nodes", 0))
    total_edges = float(stats.get("total_edges", 0))
    num_regs = float(stats.get("num_registers", 0))
    num_comb = float(stats.get("num_comb_nodes", 0))

    return [
        src_wns,
        src_tns,
        src_power,
        src_area,
        delay_s,
        power_s,
        area_s,
        scaled_wns,
        scaled_tns,
        scaled_power,
        scaled_area,
        total_nodes,
        total_edges,
        num_regs,
        num_comb,
    ]


class LayoutAndTechTransferModel:
    """Layout and Technology Transfer Model predicting cross-stage and cross-technology PPA.

    Table II: XGBoost Regressor, 15 estimators, maximum depth of 8.
    """

    def __init__(
        self,
        n_estimators: int = 15,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        def make_reg():
            if HAS_XGBOOST:
                return XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    n_jobs=-1,
                )
            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )

        self.model_wns = make_reg()
        self.model_tns = make_reg()
        self.model_power = make_reg()
        self.model_area = make_reg()
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray | list[list[float]],
        y_wns: np.ndarray | list[float],
        y_tns: np.ndarray | list[float],
        y_power: np.ndarray | list[float],
        y_area: np.ndarray | list[float],
    ) -> LayoutAndTechTransferModel:
        X_arr = np.asarray(X, dtype=np.float32)
        self.model_wns.fit(X_arr, np.asarray(y_wns, dtype=np.float32))
        self.model_tns.fit(X_arr, np.asarray(y_tns, dtype=np.float32))
        self.model_power.fit(X_arr, np.asarray(y_power, dtype=np.float32))
        self.model_area.fit(X_arr, np.asarray(y_area, dtype=np.float32))
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray | list[list[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float32)
        if not self.is_fitted:
            # Return scaled PPA metrics (columns 7, 8, 9, 10)
            return X_arr[:, 7], X_arr[:, 8], X_arr[:, 9], X_arr[:, 10]

        pred_wns = self.model_wns.predict(X_arr)
        pred_tns = self.model_tns.predict(X_arr)
        pred_power = self.model_power.predict(X_arr)
        pred_area = self.model_area.predict(X_arr)
        return pred_wns, pred_tns, pred_power, pred_area


class TransferPPAEstimator:
    """Unified Transfer Estimator coordinating layout-stage and technology-node scaling."""

    def __init__(self) -> None:
        self.transfer_model = LayoutAndTechTransferModel()

    def fit(
        self,
        X: list[list[float]],
        y_wns: list[float],
        y_tns: list[float],
        y_power: list[float],
        y_area: list[float],
    ) -> None:
        self.transfer_model.fit(X, y_wns, y_tns, y_power, y_area)

    def transfer_design(
        self,
        source_ppa: dict[str, float],
        stats: dict[str, Any],
        target_technology: str = "NanGate_45nm_TYP",
        target_scale: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        scale = target_scale or DEFAULT_TECH_SCALING.get(target_technology, {"delay": 1.0, "power": 1.0, "area": 1.0})
        feats = extract_transfer_features(source_ppa, stats, target_scale=scale)

        p_wns, p_tns, p_power, p_area = self.transfer_model.predict([feats])

        return {
            "target_technology": target_technology,
            "transferred_wns": round(float(p_wns[0]), 4),
            "transferred_tns": round(float(p_tns[0]), 4),
            "transferred_power": round(max(0.001, float(p_power[0])), 4),
            "transferred_area": round(max(0.1, float(p_area[0])), 2),
            "scale_factors": scale,
        }

