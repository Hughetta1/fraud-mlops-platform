"""
End-to-end model training script with MLflow experiment tracking.

Usage:
    python src/models/train.py                          # Full training on real data
    python src/models/train.py --n-samples 50000        # Train on 50k sample
    python src/models/train.py --no-mlflow              # Skip MLflow tracking
    python src/models/train.py --compare-models         # Compare all individual models
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.generate_data import create_fraud_dataset
from src.data_processing.feature_engineering import AdvancedFeatureEngineering, FeatureConfig
from src.data_processing.outlier_handler import remove_fraud_outliers
from src.models.version_manager import register_version
from src.models.fraud_detector import (
    AdaBoostDetector,
    EnsembleFraudDetector,
    LightGBMDetector,
    LogisticRegressionDetector,
    RandomForestDetector,
    XGBoostDetector,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to use (default: full dataset)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    parser.add_argument("--compare-models", action="store_true",
                        help="Train and compare all individual models")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Apply SMOTE balancing (default: True)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set fraction (default: 0.2)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5, set 0 for single split)")
    parser.add_argument("--optimize-threshold", action="store_true", default=True,
                        help="Find optimal decision threshold (default: True)")
    return parser.parse_args()


def setup_mlflow():
    """Initialize MLflow tracking."""
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:///{PROJECT_ROOT / 'mlruns'}")
        mlflow.set_experiment("fraud-detection")
        return mlflow
    except ImportError:
        print("MLflow not installed. Skipping experiment tracking.")
        return None


def train_individual_models(X_train, y_train, X_test, y_test):
    """Train all individual models and return their metrics."""
    models = {
        "logistic_regression": LogisticRegressionDetector(),
        "random_forest": RandomForestDetector(),
        "xgboost": XGBoostDetector(),
        "lightgbm": LightGBMDetector(),
        "adaboost": AdaBoostDetector(),
    }

    results = {}
    for name, detector in models.items():
        print(f"\n{'=' * 50}")
        print(f"Training {name}...")
        print(f"{'=' * 50}")
        t0 = time.time()
        detector.train(X_train, y_train)
        train_time = time.time() - t0

        metrics = detector.evaluate_model(X_test, y_test)
        metrics["train_time_seconds"] = train_time
        results[name] = metrics

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Train time: {train_time:.1f}s")

    return results


def print_comparison_table(results):
    """Print a formatted comparison table of all models."""
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print(f"{'=' * 80}")
    header = f"{'Model':<25} {'Accuracy':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Time':>8}"
    print(header)
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['auc']:>8.4f} "
              f"{m.get('train_time_seconds', 0):>7.1f}s")
    print("-" * 80)


def main():
    args = parse_args()

    # --- MLflow setup ---
    mlflow = None if args.no_mlflow else setup_mlflow()
    if mlflow:
        mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "n_samples": args.n_samples or "full",
            "balance": args.balance,
            "test_size": args.test_size,
        })

    # --- Load data ---
    print("Loading data...")
    df = create_fraud_dataset(n_samples=args.n_samples)
    print(f"Dataset: {len(df):,} rows, fraud rate: {df['Class'].mean():.4%}")

    # --- Outlier removal ---
    print("\nRemoving extreme IQR outliers (V14, V12, V10)...")
    df = remove_fraud_outliers(df, multiplier=1.5)

    # --- Feature engineering ---
    print("\nRunning feature engineering...")
    fe_config = FeatureConfig(
        enable_velocity_features=True,
        enable_anomaly_features=True,
        enable_clustering_features=True,
        scaling_method="robust",
        feature_selection_method="mutual_info",
        n_features_to_select=50,
        enable_pca=True,
        pca_components=0.95,
    )
    fe_pipeline = AdvancedFeatureEngineering(config=fe_config, target_column="Class")
    df_processed = fe_pipeline.fit_transform(df)

    X = df_processed.drop("Class", axis=1)
    y = df_processed["Class"]

    # --- Save reference data for model monitoring ---
    ref_path = PROJECT_ROOT / "models" / "reference_data.csv"
    # Save a sample (up to 5000 rows) of engineered features for drift detection baseline
    sample_size = min(5000, len(X))
    df_sample = X.sample(n=sample_size, random_state=42)
    df_sample.to_csv(ref_path, index=False)
    print(f"Reference data saved: {ref_path} ({sample_size} rows)")

    if mlflow:
        mlflow.log_metrics({
            "original_features": df.shape[1],
            "engineered_features": df_processed.shape[1],
            "selected_features": len(fe_pipeline.selected_features),
        })

    # --- Train individual models (if requested) ---
    if args.compare_models:
        from sklearn.model_selection import train_test_split as tts
        X_tr, X_te, y_tr, y_te = tts(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
        results = train_individual_models(X_tr, y_tr, X_te, y_te)
        print_comparison_table(results)

        if mlflow:
            for name, metrics in results.items():
                for k, v in metrics.items():
                    mlflow.log_metric(f"{name}/{k}", v)

    # --- Train ensemble ---
    print(f"\n{'=' * 50}")
    print("Training Ensemble Model")
    print(f"{'=' * 50}")
    t0 = time.time()
    ensemble = EnsembleFraudDetector(random_state=42)

    use_cv = args.cv_folds > 0
    ensemble.train(
        X, y,
        test_size=args.test_size,
        balance_data=args.balance,
        use_cv=use_cv,
        n_folds=args.cv_folds,
    )
    train_time = time.time() - t0

    print(f"\nEnsemble training complete in {train_time:.1f}s")
    for metric, value in ensemble.ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # --- Threshold optimization ---
    if args.optimize_threshold:
        print(f"\n{'=' * 50}")
        print("Optimizing Decision Threshold")
        print(f"{'=' * 50}")
        best_t = ensemble.optimize_threshold(X, y, metric="f1")
        ensemble._best_threshold = best_t

    # --- Save models with versioning ---
    n_total = len(df)
    version = register_version(
        fe_pipeline=fe_pipeline,
        ensemble=ensemble,
        metrics=ensemble.ensemble_metrics,
        training_samples=n_total,
        set_as_production=True,
    )

    models_dir = PROJECT_ROOT / "models"
    print(f"\nModels saved to: {models_dir}")
    print(f"  Version: {version}")
    print(f"  Archive: {models_dir / 'archive'}")

    # --- MLflow logging ---
    if mlflow:
        mlflow.log_param("model_version", version)
        for metric, value in ensemble.ensemble_metrics.items():
            mlflow.log_metric(f"ensemble/{metric}", value)
        mlflow.log_metric("ensemble/train_time_seconds", train_time)
        mlflow.log_artifact(str(models_dir / "archive" / f"feature_pipeline_{version}.pkl"))
        mlflow.log_artifact(str(models_dir / "archive" / f"ensemble_{version}.pkl"))
        mlflow.log_artifact(str(models_dir / "version_manifest.json"))
        mlflow.end_run()
        print(f"\nMLflow run logged. View with: mlflow ui --backend-store-uri file:///{PROJECT_ROOT / 'mlruns'}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
