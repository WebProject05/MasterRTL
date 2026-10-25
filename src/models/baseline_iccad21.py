"""ICCAD'22 [21] Presynthesis PPA Baseline Model.

Implements the register-tree logic counting and machine learning evaluation method from:
P. Sengupta, A. Tyagi, Y. Chen, and J. Hu,
"How good is your Verilog RTL code? A quick answer from machine learning,"
IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2022.

Key Characteristics (Paper Section I, Section II-A, Table III):
- Representation: Evaluated on both AST-alike (word-level) and SOG (bit-level).
- Logic Extraction: Register trees rooted at register nodes with backward combinational traversal.
  Exhibits logic duplication across shared logic trees.
- Primary Targets: TNS and vector-less Power.
  Also supports WNS and Area for comprehensive baseline comparisons.
- Absence of: Path-level critical delay mapping and toggle-rate propagation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    HAS_XGBOOST = False

from src.models.baseline_common import extract_iccad21_tree_features
from src.models.master_rtl import evaluate_metrics


class ICCAD21Baseline:
    """ICCAD'22 [21] Register-Tree Presynthesis PPA Estimation Model."""

    def __init__(
        self,
        representation: str = "ast",
        n_estimators: int = 50,
        max_depth: int = 8,
        random_state: int = 42,
    ) -> None:
        self.representation = representation.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Models for TNS and Power (primary targets of [21]) + WNS and Area
        if HAS_XGBOOST:
            self.model_tns = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.08,
                random_state=random_state,
                n_jobs=1,
            )
            self.model_power = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.08,
                random_state=random_state,
                n_jobs=1,
            )
            self.model_wns = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.08,
                random_state=random_state,
                n_jobs=1,
            )
            self.model_area = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.08,
                random_state=random_state,
                n_jobs=1,
            )
        else:
            self.model_tns = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1
            )
            self.model_power = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1
            )
            self.model_wns = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1
            )
            self.model_area = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1
            )

        self.is_fitted = False

    def extract_features(self, sample: dict[str, Any]) -> list[float]:
        """Extract ICCAD'22 register-tree features from sample."""
        stats = sample.get("stats", {})
        graph = sample.get("graph", None)
        return extract_iccad21_tree_features(stats, graph, representation=self.representation)

    def train(self, dataset: list[dict[str, Any]]) -> ICCAD21Baseline:
        """Fit ICCAD'22 models across training dataset."""
        X = []
        y_tns = []
        y_power = []
        y_wns = []
        y_area = []

        for d in dataset:
            feats = self.extract_features(d)
            X.append(feats)
            lbl = d["labels"]
            y_tns.append(lbl["tns"])
            y_power.append(lbl["power"])
            y_wns.append(lbl.get("wns", 0.0))
            y_area.append(lbl.get("area", 100.0))

        X_arr = np.asarray(X, dtype=np.float32)
        self.model_tns.fit(X_arr, np.asarray(y_tns, dtype=np.float32))
        self.model_power.fit(X_arr, np.asarray(y_power, dtype=np.float32))
        self.model_wns.fit(X_arr, np.asarray(y_wns, dtype=np.float32))
        self.model_area.fit(X_arr, np.asarray(y_area, dtype=np.float32))

        self.is_fitted = True
        return self

    def predict(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Predict presynthesis PPA for a design using ICCAD'22 register tree model."""
        feats = self.extract_features(sample)
        X_arr = np.asarray([feats], dtype=np.float32)

        pred_tns = float(self.model_tns.predict(X_arr)[0])
        pred_power = max(0.01, float(self.model_power.predict(X_arr)[0]))
        pred_wns = min(0.0, float(self.model_wns.predict(X_arr)[0]))
        pred_area = max(1.0, float(self.model_area.predict(X_arr)[0]))

        return {
            "predicted_wns": round(pred_wns, 4),
            "predicted_tns": round(pred_tns, 4),
            "predicted_power_mW": round(pred_power, 4),
            "predicted_area_um2": round(pred_area, 2),
            "features": feats,
        }

    def evaluate_dataset(self, dataset: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Evaluate predictions across dataset against ground truth."""
        y_true_wns, y_pred_wns = [], []
        y_true_tns, y_pred_tns = [], []
        y_true_pwr, y_pred_pwr = [], []
        y_true_area, y_pred_area = [], []

        for d in dataset:
            pred = self.predict(d)
            lbl = d["labels"]

            y_true_wns.append(lbl.get("wns", 0.0))
            y_pred_wns.append(pred["predicted_wns"])

            y_true_tns.append(lbl["tns"])
            y_pred_tns.append(pred["predicted_tns"])

            y_true_pwr.append(lbl["power"])
            y_pred_pwr.append(pred["predicted_power_mW"])

            y_true_area.append(lbl.get("area", 100.0))
            y_pred_area.append(pred["predicted_area_um2"])

        return {
            "WNS": evaluate_metrics(y_true_wns, y_pred_wns),
            "TNS": evaluate_metrics(y_true_tns, y_pred_tns),
            "Power": evaluate_metrics(y_true_pwr, y_pred_pwr),
            "Area": evaluate_metrics(y_true_area, y_pred_area),
        }
