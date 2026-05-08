# SHAP explainability module.
# For every prediction, returns the top-5 features pushing the score toward
# fraud or normal, with their SHAP values. Uses TreeExplainer for RF/XGB/LGB,
# skips AdaBoost (unsupported), and re-normalizes weights accordingly.
#
# Interview talking point: this is not just calling shap.summary_plot().
# It does weighted multi-model SHAP aggregation and directional analysis
# (positive SHAP = toward fraud, negative = toward normal).

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class FraudExplainer:

    def __init__(self, ensemble, feature_names=None):
        self.ensemble = ensemble
        self.explainers = {}
        self.feature_names = feature_names
        self._background = None
        self._is_fitted = False
        self._shap_weights = {}

    def fit_background(self, X_background, sample_size=200):
        # TreeExplainer only needs ~100-200 background samples for stable SHAP values.
        import shap

        bg = X_background
        if sample_size and len(bg) > sample_size:
            bg = bg.sample(n=sample_size, random_state=42)

        self._background = bg
        self.feature_names = list(bg.columns)

        for name, detector in self.ensemble.models.items():
            model = detector.model.named_steps["clf"]
            preprocessor = detector.model.named_steps["scaler"]
            try:
                if name == "logistic_regression":
                    exp = shap.LinearExplainer(model, preprocessor.transform(bg),
                                               feature_dependence="independent")
                elif name == "adaboost":
                    # sklearn's AdaBoostClassifier is not supported by TreeExplainer.
                    # The other three models carry enough weight.
                    continue
                else:
                    exp = shap.TreeExplainer(model, data=preprocessor.transform(bg),
                                             feature_perturbation="interventional")
                self.explainers[name] = exp
            except Exception as e:
                print(f"  Skipping {name}: {e}")

        if not self.explainers:
            raise RuntimeError("No SHAP explainers could be fitted")

        # Re-normalize ensemble weights to only include explained models
        total = sum(self.ensemble._model_weights.get(n, 1.0) for n in self.explainers)
        self._shap_weights = {
            n: self.ensemble._model_weights.get(n, 1.0) / max(total, 1e-8)
            for n in self.explainers
        }
        self._is_fitted = True
        print(f"SHAP explainers fitted: {list(self.explainers.keys())}")

    def _preprocess(self, X):
        # Apply each sub-model's scaler to get the features the classifier actually sees.
        result = {}
        for name, detector in self.ensemble.models.items():
            scaler = detector.model.named_steps["scaler"]
            result[name] = scaler.transform(X)
        return result

    @staticmethod
    def _extract_shap_1d(explainer, X_scaled_row):
        # SHAP output format varies across versions: list of arrays, 3D tensor, etc.
        # Normalize to a flat 1D array of per-feature SHAP values.
        raw = explainer.shap_values(X_scaled_row)
        arr = np.array(raw, dtype=float)
        if isinstance(raw, list):
            return np.array(raw[1], dtype=float).flatten()
        if arr.ndim == 3:
            return arr[0, :, 1] if arr.shape[2] > 1 else arr[0, :, 0]
        return arr.flatten()[-X_scaled_row.shape[1]:]

    def explain_prediction(self, X):
        if not self._is_fitted:
            raise RuntimeError("Call fit_background() first.")
        X_scaled = self._preprocess(X)

        global_importance = self._compute_global_importance(X_scaled)
        per_prediction = []

        for i in range(len(X)):
            model_explanations = {}
            for name, explainer in self.explainers.items():
                sv = self._extract_shap_1d(explainer, X_scaled[name][i:i+1])
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)) and not np.isscalar(ev):
                    base = float(np.mean(np.array(ev, dtype=float).flatten()))
                else:
                    base = float(ev)
                model_explanations[name] = {"base_value": base, "shap_values": {
                    f: float(v) for f, v in zip(self.feature_names, sv)
                }}

            # Weighted-average SHAP across all explained models
            weighted_shap = np.zeros(len(self.feature_names))
            for name, exp in model_explanations.items():
                w = self._shap_weights.get(name, 1.0 / len(self.explainers))
                sv_list = [exp["shap_values"][f] for f in self.feature_names]
                weighted_shap += np.array(sv_list) * w

            top_factors = self._top_risk_factors(weighted_shap, top_n=5)
            per_prediction.append({
                "model_explanations": model_explanations,
                "weighted_shap": {f: float(v) for f, v in zip(self.feature_names, weighted_shap)},
                "top_risk_factors": top_factors,
            })

        return {"global_importance": global_importance, "per_prediction": per_prediction}

    def _compute_global_importance(self, X_scaled):
        # Mean absolute SHAP across all models, weighted by ensemble importance.
        importances = {}
        for name, explainer in self.explainers.items():
            raw = explainer.shap_values(X_scaled[name])
            if isinstance(raw, list):
                sv = np.array(raw[1], dtype=float)
            else:
                sv = np.array(raw, dtype=float)
                if sv.ndim == 3 and sv.shape[2] > 1:
                    sv = sv[:, :, 1]
                elif sv.ndim == 3:
                    sv = sv[:, :, 0]
            w = self._shap_weights.get(name, 1.0 / len(self.explainers))
            mean_abs = np.abs(sv).mean(axis=0)
            for feat, val in zip(self.feature_names, mean_abs):
                importances[feat] = importances.get(feat, 0.0) + val * w
        return dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True))

    def _top_risk_factors(self, shap_array, top_n=5):
        # Sort by SHAP value descending — positive values push toward fraud.
        pairs = list(zip(self.feature_names, shap_array))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [
            {"feature": f, "impact": round(float(v), 6),
             "direction": "fraud" if v > 0 else "normal"}
            for f, v in pairs[:top_n]
        ]

    def get_top_features(self, top_n=10):
        if not self._is_fitted or self._background is None:
            return []
        X_scaled = self._preprocess(self._background)
        importance = self._compute_global_importance(X_scaled)
        return list(importance.keys())[:top_n]
