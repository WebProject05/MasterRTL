"""RTL-Stage Power Modeling Framework for MasterRTL.

Corresponds to Section II-C and Table II in:
"Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques"
(IEEE TCAD 2025).

Architecture:
1. Node toggle rate propagation on SOG (Stage 1)
2. Module/sub-SOG level power estimation: XGBoost Regressor (30 trees, max_depth=6) (Stage 2)
3. Module power aggregation: Power^G = sum(k_i * Power^{G_i})
4. Design-level power calibration: XGBoost Regressor (45 trees, max_depth=8) (Stage 3)
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


def extract_module_power_features(
    module_stats: dict[str, Any],
) -> list[float]:
    """Extract feature vector for module-level power estimation.

    Features from Section II-C:
    1. Sum of toggle rates
    2. Average toggle rate
    3. Sum of (fan_out * toggle_rate) on each node (weighted switching)
    4. Total number of nodes
    5. Count of AND operators
    6. Count of OR operators
    7. Count of XOR operators
    8. Count of NOT operators
    9. Count of MUX operators
    10. Total register count (DFFs, governing static leakage power)
    """
    return [
        float(module_stats.get("total_toggle_rate", 0.0)),
        float(module_stats.get("mean_toggle_rate", 0.05)),
        float(module_stats.get("weighted_toggle_rate", 0.0)),
        float(module_stats.get("total_nodes", 0)),
        float(module_stats.get("num_and", 0)),
        float(module_stats.get("num_or", 0)),
        float(module_stats.get("num_xor", 0)),
        float(module_stats.get("num_not", 0)),
        float(module_stats.get("num_mux", 0)),
        float(module_stats.get("num_registers", 0)),
    ]


class ModuleLevelPowerModel:
    """Module-level power model predicting power consumption of sub-SOG modules.

    Table II: XGBoost Regressor, 30 estimators, maximum depth of 6.
    """

    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int = 6,
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

    def fit(self, X: np.ndarray | list[list[float]], y: np.ndarray | list[float]) -> ModuleLevelPowerModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if not self.is_fitted:
            # Baseline dynamic power formula approximation: P ~ C * V^2 * f * alpha
            weighted_toggle = X_arr[:, 2]
            num_nodes = X_arr[:, 3]
            num_regs = X_arr[:, 9]
            approx_dynamic = 0.05 * weighted_toggle
            approx_static = 0.002 * (num_nodes + num_regs)
            return approx_dynamic + approx_static
        return self.model.predict(X_arr)


def extract_design_power_calibration_features(
    stats: dict[str, Any],
    sum_module_power: float,
) -> list[float]:
    """Build design-level feature vector for Stage 3 power calibration.

    Features:
    1. Sum of module power predictions
    2. SOG design-scale features (total nodes, total edges, registers, combinational)
    3. Global toggle rate statistics
    4. Operator breakdown
    """
    return [
        float(sum_module_power),
        float(stats.get("total_nodes", 0)),
        float(stats.get("total_edges", 0)),
        float(stats.get("num_registers", 0)),
        float(stats.get("num_comb_nodes", 0)),
        float(stats.get("total_toggle_rate", 0.0)),
        float(stats.get("mean_toggle_rate", 0.05)),
        float(stats.get("weighted_toggle_rate", 0.0)),
        float(stats.get("num_and", 0)),
        float(stats.get("num_or", 0)),
        float(stats.get("num_xor", 0)),
        float(stats.get("num_not", 0)),
        float(stats.get("num_mux", 0)),
        float(stats.get("mean_fanout", 1.0)),
        float(stats.get("max_fanout", 1.0)),
    ]


class DesignLevelPowerCalibrationModel:
    """Stage 3 Power Calibration Model refining summed module power to gate-level netlist power.

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

    def fit(
        self,
        X: np.ndarray | list[list[float]],
        y_power: np.ndarray | list[float],
    ) -> DesignLevelPowerCalibrationModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y_power, dtype=np.float32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if not self.is_fitted:
            # Return uncalibrated module sum (column 0)
            return X_arr[:, 0]
        return self.model.predict(X_arr)


class PowerPPAEstimator:
    """Unified multistage power estimator coordinating module estimation and calibration."""

    def __init__(self) -> None:
        self.module_model = ModuleLevelPowerModel()
        self.calib_model = DesignLevelPowerCalibrationModel()

    def fit(
        self,
        module_features_list: list[list[float]],
        module_power_labels: list[float],
        design_calibration_features: list[list[float]],
        design_power_labels: list[float],
    ) -> None:
        if module_features_list and module_power_labels:
            self.module_model.fit(module_features_list, module_power_labels)
        if design_calibration_features and design_power_labels:
            self.calib_model.fit(design_calibration_features, design_power_labels)

    def predict_design(
        self,
        stats: dict[str, Any],
        modules_stats: list[tuple[dict[str, Any], int]] | None = None,
    ) -> dict[str, Any]:
        """Predict total power for a design.

        If modules_stats is provided (list of (sub_module_stats, instance_count_ki)),
        module-level power is evaluated and summed. Otherwise, whole design is used.
        """
        if modules_stats:
            sum_power = 0.0
            for mod_stat, ki in modules_stats:
                feats = extract_module_power_features(mod_stat)
                pred_mod = float(self.module_model.predict([feats])[0])
                sum_power += ki * pred_mod
        else:
            feats = extract_module_power_features(stats)
            sum_power = float(self.module_model.predict([feats])[0])

        calib_feats = extract_design_power_calibration_features(stats, sum_power)
        calibrated_power = float(self.calib_model.predict([calib_feats])[0])

        return {
            "sum_module_power": round(sum_power, 4),
            "predicted_power": round(max(0.001, calibrated_power), 4),
        }

