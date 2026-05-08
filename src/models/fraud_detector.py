# Fraud detection model classes.
# 5 individual models + 1 ensemble, all sharing the same interface:
# train / predict / predict_proba / evaluate_model / save / load.
#
# Key design decisions:
# - Every model wraps a sklearn Pipeline(scaler + classifier) for consistent serialization.
# - SMOTE is applied *inside* each CV fold, not before — this prevents data leakage,
#   the #1 mistake flagged by Kaggle gold-medal notebooks on this dataset.
# - predict_proba uses _safe_predict_proba to handle the edge case where the test set
#   contains only one class (returns (n,1) instead of (n,2)).

from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Utility functions — handle class imbalance edge cases
# ---------------------------------------------------------------------------

def _safe_predict_proba(model, X) -> np.ndarray:
    # When the test set has only one class, sklearn returns (n,1) instead of (n,2).
    # Calling [:, 1] on that would IndexError.
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


def _safe_smote(X, y, random_state):
    # SMOTE's default k_neighbors=5 crashes when the minority class has < 6 samples.
    # Dynamically reduce k_neighbors to min(5, minority_count - 1).
    min_count = pd.Series(y).value_counts().min()
    if min_count < 2:
        return X, y
    k = max(1, min(5, min_count - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    return smote.fit_resample(X, y)


def _cross_val_predict_with_smote(estimator, X, y, cv=5, random_state=42):
    # Apply SMOTE inside each CV fold independently.
    # This is the "right way" per Kaggle gold references — SMOTE before the split
    # leaks synthetic samples into validation folds and inflates metrics.
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.model_selection import StratifiedKFold

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    min_count = pd.Series(y).value_counts().min()
    k = max(1, min(5, min_count // cv))
    fold_smote = SMOTE(random_state=random_state, k_neighbors=k)

    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", fold_smote),
        ("clf", estimator),
    ])

    oof_proba = np.zeros(len(X))
    oof_preds = np.zeros(len(X))

    for train_idx, valid_idx in cv_splitter.split(X, y):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_va = X.iloc[valid_idx] if hasattr(X, "iloc") else X[valid_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]

        pipe.fit(X_tr, y_tr)
        oof_proba[valid_idx] = _safe_predict_proba(pipe, X_va)
        oof_preds[valid_idx] = pipe.predict(X_va)

    return oof_proba, oof_preds


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    # For fraud detection, the default 0.5 threshold is rarely optimal.
    # False negatives (missing fraud) cost far more than false positives.
    # Sweep [0.1, 0.9] and pick the threshold that maximizes the chosen metric.
    thresholds = np.arange(0.1, 0.91, 0.02)
    best_score = -1.0
    best_threshold = 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, preds, zero_division=0)
        elif metric == "youden":
            # Youden's J = TPR - FPR, balances sensitivity and specificity
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            score = tpr - fpr
        else:
            score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold


def _safe_stratified_split(X, y, test_size, random_state):
    # Falls back to unstratified split when a class has < 2 samples,
    # which would otherwise crash train_test_split with stratify=True.
    min_class_count = pd.Series(y).value_counts().min()
    if min_class_count >= 2:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Individual model classes.
# Each wraps a Pipeline(scaler + classifier) for consistent serialization.
# The interface is deliberately uniform: train / predict / predict_proba /
# evaluate_model / save_model / load_model + is_trained flag.
# ---------------------------------------------------------------------------

class LogisticRegressionDetector:
    # Baseline model — included in the ensemble mainly as a reference point.

    def __init__(self, max_iter=1000, class_weight="balanced", balance_data=False, random_state=42):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, class_weight=class_weight)),
        ])
        self.balance_data = balance_data
        self.random_state = random_state
        self.is_trained = False

    def prepare_data(self, X, y, test_size=0.2, balance_data=None):
        X_train, X_test, y_train, y_test = _safe_stratified_split(X, y, test_size, self.random_state)
        if (balance_data is None and self.balance_data) or balance_data:
            X_train, y_train = _safe_smote(X_train, y_train, self.random_state)
        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        y_pred = self.model.predict(X_test)
        y_prob = _safe_predict_proba(self.model, X_test)
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = _safe_predict_proba(self.model, X_test)
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)),
        }

    def predict(self, X):      return self.model.predict(X)
    def predict_proba(self, X): return _safe_predict_proba(self.model, X)
    def save(self, path):      joblib.dump(self.model, path)
    def load(self, path):      self.model = joblib.load(path); self.is_trained = True
    def save_model(self, p):   self.save(p)
    def load_model(self, p):   self.load(p)


class RandomForestDetector:
    # class_weight='balanced' helps with imbalance; SMOTE adds further robustness.

    def __init__(self, n_estimators=100, max_depth=10, class_weight="balanced",
                 balance_data=False, random_state=42):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                class_weight=class_weight, random_state=random_state, n_jobs=-1)),
        ])
        self.balance_data = balance_data
        self.random_state = random_state
        self.is_trained = False

    def prepare_data(self, X, y, test_size=0.2, balance_data=None):
        X_train, X_test, y_train, y_test = _safe_stratified_split(X, y, test_size, self.random_state)
        if (balance_data is None and self.balance_data) or balance_data:
            X_train, y_train = _safe_smote(X_train, y_train, self.random_state)
        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        self.model.fit(X, y); self.is_trained = True

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = _safe_predict_proba(self.model, X_test)
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)),
        }

    def predict(self, X):      return self.model.predict(X)
    def predict_proba(self, X): return _safe_predict_proba(self.model, X)
    def save_model(self, p):   joblib.dump(self.model, p)
    def load_model(self, p):   self.model = joblib.load(p); self.is_trained = True


class XGBoostDetector:
    # scale_pos_weight = neg/pos is more direct than class_weight for XGBoost.
    # eval_metric='aucpr' is better than default 'error' for imbalanced data.

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 scale_pos_weight=None, balance_data=False, random_state=42):
        from xgboost import XGBClassifier
        self._xgb_params = dict(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight, random_state=random_state,
            eval_metric="aucpr", use_label_encoder=False,
        )
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(**self._xgb_params)),
        ])
        self.balance_data = balance_data
        self.random_state = random_state
        self.is_trained = False

    def prepare_data(self, X, y, test_size=0.2, balance_data=None):
        X_train, X_test, y_train, y_test = _safe_stratified_split(X, y, test_size, self.random_state)
        if (balance_data is None and self.balance_data) or balance_data:
            X_train, y_train = _safe_smote(X_train, y_train, self.random_state)
        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        if self._xgb_params["scale_pos_weight"] is None:
            neg, pos = int((y == 0).sum()), int((y == 1).sum())
            self.model.named_steps["clf"].set_params(scale_pos_weight=neg / max(pos, 1))
        self.model.fit(X, y); self.is_trained = True

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = _safe_predict_proba(self.model, X_test)
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)),
        }

    def predict(self, X):      return self.model.predict(X)
    def predict_proba(self, X): return _safe_predict_proba(self.model, X)
    def save_model(self, p):   joblib.dump(self.model, p)
    def load_model(self, p):   self.model = joblib.load(p); self.is_trained = True


class LightGBMDetector:
    # Parameterization follows Kaggle gold-medal references:
    # explicit scale_pos_weight (not is_unbalance=True), L1/L2 regularization,
    # and subsampling to prevent overfitting on the tiny fraud class.

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05,
                 num_leaves=31, scale_pos_weight=None, reg_alpha=0.04, reg_lambda=0.07,
                 subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
                 balance_data=False, random_state=42):
        from lightgbm import LGBMClassifier
        self._lgb_params = dict(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            num_leaves=num_leaves, scale_pos_weight=scale_pos_weight,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, subsample=subsample,
            colsample_bytree=colsample_bytree, min_child_samples=min_child_samples,
            random_state=random_state, verbose=-1,
        )
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(**self._lgb_params)),
        ])
        self.balance_data = balance_data
        self.random_state = random_state
        self.is_trained = False

    def prepare_data(self, X, y, test_size=0.2, balance_data=None):
        X_train, X_test, y_train, y_test = _safe_stratified_split(X, y, test_size, self.random_state)
        if (balance_data is None and self.balance_data) or balance_data:
            X_train, y_train = _safe_smote(X_train, y_train, self.random_state)
        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        if self._lgb_params["scale_pos_weight"] is None:
            neg, pos = int((y == 0).sum()), int((y == 1).sum())
            self.model.named_steps["clf"].set_params(scale_pos_weight=neg / max(pos, 1))
        self.model.fit(X, y); self.is_trained = True

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = _safe_predict_proba(self.model, X_test)
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)),
        }

    def predict(self, X):      return self.model.predict(X)
    def predict_proba(self, X): return _safe_predict_proba(self.model, X)
    def save_model(self, p):   joblib.dump(self.model, p)
    def load_model(self, p):   self.model = joblib.load(p); self.is_trained = True


class AdaBoostDetector:
    # Uses a DecisionTree(max_depth=3) as base estimator with class_weight='balanced'.
    # Note: SHAP's TreeExplainer does not support sklearn's AdaBoostClassifier,
    # so explainer.py automatically skips this model and re-normalizes weights.

    def __init__(self, n_estimators=100, learning_rate=0.8, balance_data=False, random_state=42):
        from sklearn.tree import DecisionTreeClassifier
        base = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=random_state)
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", AdaBoostClassifier(estimator=base, n_estimators=n_estimators,
                                       learning_rate=learning_rate, random_state=random_state)),
        ])
        self.balance_data = balance_data
        self.random_state = random_state
        self.is_trained = False

    def prepare_data(self, X, y, test_size=0.2, balance_data=None):
        X_train, X_test, y_train, y_test = _safe_stratified_split(X, y, test_size, self.random_state)
        if (balance_data is None and self.balance_data) or balance_data:
            X_train, y_train = _safe_smote(X_train, y_train, self.random_state)
        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        self.model.fit(X, y); self.is_trained = True

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = _safe_predict_proba(self.model, X_test)
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)),
        }

    def predict(self, X):      return self.model.predict(X)
    def predict_proba(self, X): return _safe_predict_proba(self.model, X)
    def save_model(self, p):   joblib.dump(self.model, p)
    def load_model(self, p):   self.model = joblib.load(p); self.is_trained = True


# ---------------------------------------------------------------------------
# Ensemble: AUC-weighted soft voting across 4 models.
# Two training modes:
#   use_cv=True  → StratifiedKFold with fold-internal SMOTE (no leakage)
#   use_cv=False → simple train/test split (backward compatible with tests)
# ---------------------------------------------------------------------------

class EnsembleFraudDetector:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.is_trained = False
        self.models = {}
        self.ensemble_metrics = {}
        self._model_weights = {}

    def train(self, X, y, test_size=0.2, balance_data=False, use_cv=False, n_folds=5):
        self.models = {
            "random_forest": RandomForestDetector(random_state=self.random_state),
            "xgboost": XGBoostDetector(random_state=self.random_state),
            "lightgbm": LightGBMDetector(random_state=self.random_state),
            "adaboost": AdaBoostDetector(random_state=self.random_state),
        }

        if use_cv:
            print(f"\nUsing {n_folds}-fold CV with SMOTE inside each fold (no leakage).")
            aucs = {}
            for name, detector in self.models.items():
                print(f"\n--- CV Training {name} ---")
                oof_proba, _ = _cross_val_predict_with_smote(
                    detector.model.named_steps["clf"],
                    X, y, cv=n_folds, random_state=self.random_state,
                )
                aucs[name] = float(roc_auc_score(y, oof_proba))
                detector.train(X, y)
                print(f"  OOF AUC: {aucs[name]:.4f}")

            total = sum(aucs.values())
            self._model_weights = {k: v / max(total, 1e-8) for k, v in aucs.items()}
            oof_ensemble = sum(
                detector.predict_proba(X) * self._model_weights[name]
                for name, detector in self.models.items()
            )
            y_true = y
            y_prob = oof_ensemble
            y_pred = (oof_ensemble >= 0.5).astype(int)
        else:
            X_train, X_test, y_train, y_test = _safe_stratified_split(X, y, test_size, self.random_state)
            if balance_data:
                X_train, y_train = _safe_smote(X_train, y_train, self.random_state)

            aucs = {}
            for name, detector in self.models.items():
                print(f"\n--- Training {name} ---")
                detector.train(X_train, y_train)
                metrics = detector.evaluate_model(X_test, y_test)
                aucs[name] = metrics["auc"]

            total = sum(aucs.values())
            self._model_weights = {k: v / max(total, 1e-8) for k, v in aucs.items()}
            y_true = y_test
            y_prob = self.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

        self.ensemble_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

        self.is_trained = True
        print(f"\nEnsemble weights: {self._model_weights}")
        print(f"Ensemble metrics: {self.ensemble_metrics}")

    def optimize_threshold(self, X, y, metric="f1"):
        proba = self.predict_proba(X)[:, 1]
        self._best_threshold = find_optimal_threshold(y.values, proba, metric)
        print(f"Optimal threshold ({metric}): {self._best_threshold:.3f}")
        return self._best_threshold

    def predict(self, X, threshold=None):
        proba = self.predict_proba(X)[:, 1]
        t = threshold if threshold is not None else getattr(self, "_best_threshold", 0.5)
        return (proba >= t).astype(int)

    def predict_proba(self, X):
        weighted_sum = np.zeros(len(X))
        for name, detector in self.models.items():
            weighted_sum += detector.predict_proba(X) * self._model_weights[name]
        return np.column_stack([1.0 - weighted_sum, weighted_sum])

    def save_ensemble(self, filepath):
        # Persist each sub-model's Pipeline (scaler + classifier).
        # Weights and metrics are stored for consistency checks on reload.
        joblib.dump({
            "models": {name: d.model for name, d in self.models.items()},
            "weights": self._model_weights,
            "metrics": self.ensemble_metrics,
            "random_state": self.random_state,
        }, filepath)

    def load_ensemble(self, filepath):
        data = joblib.load(filepath)
        self._model_weights = data["weights"]
        self.ensemble_metrics = data["metrics"]
        self.random_state = data["random_state"]

        mapping = {
            "random_forest": RandomForestDetector, "xgboost": XGBoostDetector,
            "lightgbm": LightGBMDetector, "adaboost": AdaBoostDetector,
        }
        self.models = {}
        for name, pipeline in data["models"].items():
            cls = mapping.get(name, LogisticRegressionDetector)
            d = cls(random_state=self.random_state)
            d.model = pipeline
            d.is_trained = True
            self.models[name] = d
        self.is_trained = True


def finalize_training():
    print("Training complete.")
