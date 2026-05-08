"""
Real-Time Fraud Detection API

Production-ready FastAPI application for real-time fraud detection with:
- Real-time transaction processing
- ML model inference with ensemble predictions
- SHAP explainability for decision transparency
- Comprehensive request validation
- Detailed response formatting
- Performance monitoring and logging
- Health checks and status endpoints

Author: Sunny Nguyen
"""

import logging
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

# ML and data processing
import joblib
import pandas as pd

# FastAPI imports
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system using ensemble machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Global variables for models and performance metrics
# ---------------------------------------------------------------------
feature_pipeline = None
ensemble_model = None
shadow_pipeline = None   # Shadow model feature pipeline (Phase 5 prep)
shadow_model = None      # Shadow model for A/B comparison (Phase 5 prep)
model_monitor = None     # ModelMonitor instance for drift detection
fraud_explainer = None   # SHAP FraudExplainer for decision transparency
model_loaded = False
model_load_time: Optional[str] = None
production_version: Optional[str] = None
shadow_version: Optional[str] = None

request_count = 0
total_processing_time = 0.0
start_time = time.time()

# Prediction log database (SQLite) for future label joining
import sqlite3
import threading

_pred_log_lock = threading.Lock()
_pred_log_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "predictions.db",
)


def _init_prediction_log():
    """Create the prediction log table if it doesn't exist."""
    os.makedirs(os.path.dirname(_pred_log_path), exist_ok=True)
    with _pred_log_lock:
        conn = sqlite3.connect(_pred_log_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                fraud_probability REAL,
                risk_level TEXT,
                model_version TEXT,
                shadow_probability REAL,
                shadow_version TEXT,
                processing_time_ms REAL,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()


def _log_prediction(
    transaction_id: Optional[str],
    fraud_probability: float,
    risk_level: str,
    model_version: str,
    processing_time_ms: float,
    shadow_probability: Optional[float] = None,
    shadow_version: Optional[str] = None,
):
    """Log a prediction to SQLite (fire-and-forget, non-blocking)."""
    try:
        with _pred_log_lock:
            conn = sqlite3.connect(_pred_log_path)
            conn.execute(
                "INSERT INTO predictions "
                "(transaction_id, fraud_probability, risk_level, model_version, "
                "shadow_probability, shadow_version, processing_time_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    transaction_id,
                    fraud_probability,
                    risk_level,
                    model_version,
                    shadow_probability,
                    shadow_version,
                    processing_time_ms,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass  # Never let logging break the prediction path


# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """Schema for validating a single transaction"""

    Time: float = Field(..., ge=0, description="Time in seconds from reference point")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")

    # PCA-anonymized features
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: Optional[float] = None
    V22: Optional[float] = None
    V23: Optional[float] = None
    V24: Optional[float] = None
    V25: Optional[float] = None
    V26: Optional[float] = None
    V27: Optional[float] = None
    V28: Optional[float] = None

    transaction_id: Optional[str] = Field(None, description="Unique transaction ID")

    @validator("Amount")
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError("Amount must be non-negative")
        if v > 100000:
            logger.warning(f"Unusually large transaction: ${v:,.2f}")
        return v

    @validator("Time")
    def validate_time(cls, v):
        if v < 0:
            raise ValueError("Time must be non-negative")
        return v


class BatchTransactionRequest(BaseModel):
    """Schema for validating a batch of transactions"""

    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=1000)


class FraudPredictionResponse(BaseModel):
    """Response model for a single fraud prediction"""

    transaction_id: Optional[str]
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    confidence_score: float = Field(..., ge=0, le=1)
    explanation: Optional[Dict[str, Any]] = None
    top_risk_factors: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: List[FraudPredictionResponse]
    batch_summary: Dict[str, Any]
    total_processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    model_loaded: bool
    model_load_time: Optional[str]
    uptime_seconds: float
    version: str


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def load_models() -> bool:
    """Load production model (and optional shadow model) using version manager."""
    global feature_pipeline, ensemble_model, model_loaded, model_load_time
    global shadow_pipeline, shadow_model, production_version, shadow_version
    global model_monitor

    try:
        logger.info("Loading fraud detection models via version manager...")

        from src.models.version_manager import (
            load_production,
            load_shadow,
            get_production_version,
            get_shadow_version,
        )

        # Load production model
        fe, ens = load_production()
        if fe is None or ens is None:
            logger.error("Production model not found. Run train.py first.")
            return False

        feature_pipeline = fe
        ensemble_model = ens
        production_version = get_production_version()
        model_loaded = True
        model_load_time = datetime.now().isoformat()
        logger.info(f"Production model loaded: {production_version}")

        # Load shadow model (if available)
        try:
            s_fe, s_ens = load_shadow()
            if s_fe is not None and s_ens is not None:
                shadow_pipeline = s_fe
                shadow_model = s_ens
                shadow_version = get_shadow_version()
                logger.info(f"Shadow model loaded: {shadow_version}")
            else:
                logger.info("No shadow model configured")
        except Exception as e:
            logger.warning(f"Shadow model load skipped: {e}")

        # Initialize model monitor with reference data
        try:
            from src.monitoring.model_monitor import ModelMonitor
            import pandas as pd
            ref_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models", "reference_data.csv",
            )
            if os.path.exists(ref_path):
                reference_data = pd.read_csv(ref_path)
                logger.info(f"Monitor reference loaded: {len(reference_data)} rows")
            else:
                reference_data = pd.DataFrame()
                logger.warning("No reference data found, monitor using empty baseline")
            model_monitor = ModelMonitor(reference_data, drift_threshold=0.05)
            logger.info("Model monitor initialized")
        except Exception as e:
            logger.warning(f"Model monitor init skipped: {e}")

        # Initialize SHAP explainer
        try:
            from src.models.explainer import FraudExplainer
            global fraud_explainer
            fraud_explainer = FraudExplainer(ensemble_model)
            if not reference_data.empty:
                bg_sample = reference_data.head(200) if len(reference_data) >= 200 else reference_data
                fraud_explainer.fit_background(bg_sample)
                logger.info("SHAP explainer initialized")
            else:
                logger.warning("No reference data for SHAP background")
        except Exception as e:
            logger.warning(f"SHAP explainer init skipped: {e}")

        return True
    except Exception as e:
        logger.exception(f"Error loading models: {e}")
        model_loaded = False
        return False


def preprocess_transaction(transaction: TransactionRequest) -> pd.DataFrame:
    """Convert a TransactionRequest into a preprocessed DataFrame"""
    try:
        data = transaction.model_dump(exclude_none=True)
        data.pop("transaction_id", None)
        df = pd.DataFrame([data])

        if feature_pipeline is not None and hasattr(feature_pipeline, "transform"):
            return feature_pipeline.transform(df)
        logger.warning("Feature pipeline unavailable, using raw features.")
        return df
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")


def get_risk_level(probability: float) -> str:
    if probability >= 0.9:
        return "CRITICAL"
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.3:
        return "MEDIUM"
    return "LOW"


def get_confidence_score(probability: float) -> float:
    """Confidence increases as probability approaches 0 or 1"""
    return round(abs(probability - 0.5) * 2, 4)


async def check_model_dependency():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please reload or restart the server.",
        )


# ---------------------------------------------------------------------
# FastAPI event hooks
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Fraud Detection API...")
    _init_prediction_log()
    if not load_models():
        logger.warning("Models failed to load during startup.")


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        model_load_time=model_load_time,
        uptime_seconds=time.time() - start_time,
        version=production_version or "unknown",
    )


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_model_dependency),
):
    global request_count, total_processing_time
    t_total_start = time.perf_counter()

    # Step 1: Feature transform
    t_fe_start = time.perf_counter()
    try:
        df = preprocess_transaction(transaction)
    except Exception as e:
        logger.exception(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")
    t_fe_ms = (time.perf_counter() - t_fe_start) * 1000

    # Step 2: Model inference
    t_inf_start = time.perf_counter()
    fraud_prob = ensemble_model.predict_proba(df)[0, 1]
    t_inf_ms = (time.perf_counter() - t_inf_start) * 1000

    # Step 3: Post-processing
    risk_level = get_risk_level(fraud_prob)
    confidence = get_confidence_score(fraud_prob)
    t_total_ms = (time.perf_counter() - t_total_start) * 1000

    request_count += 1
    total_processing_time += t_total_ms

    # Log latency breakdown for production monitoring
    logger.info(
        f"Predict | total={t_total_ms:.1f}ms | "
        f"fe_transform={t_fe_ms:.1f}ms | "
        f"inference={t_inf_ms:.1f}ms | "
        f"risk={risk_level} | prob={fraud_prob:.4f}"
    )

    # Feed model monitor for drift detection (fire-and-forget)
    if model_monitor is not None:
        try:
            model_monitor.analyze_prediction_distribution(
                np.array([fraud_prob])
            )
        except Exception:
            pass

    # Log prediction for future label joining (fire-and-forget)
    _log_prediction(
        transaction_id=transaction.transaction_id,
        fraud_probability=round(fraud_prob, 4),
        risk_level=risk_level,
        model_version=production_version or "unknown",
        processing_time_ms=round(t_total_ms, 2),
    )

    # Shadow model comparison (Phase 5 prep)
    shadow_result = None
    if shadow_model is not None and shadow_pipeline is not None:
        try:
            t_shadow_start = time.perf_counter()
            df_shadow = shadow_pipeline.transform(
                pd.DataFrame([transaction.model_dump(exclude_none=True)]).drop(
                    columns=["transaction_id"], errors="ignore"
                )
            )
            shadow_prob = shadow_model.predict_proba(df_shadow)[0, 1]
            shadow_ms = (time.perf_counter() - t_shadow_start) * 1000
            shadow_result = {
                "fraud_probability": round(shadow_prob, 4),
                "risk_level": get_risk_level(shadow_prob),
                "processing_time_ms": round(shadow_ms, 2),
            }
            # Log shadow prediction for A/B comparison over time
            _log_prediction(
                transaction_id=transaction.transaction_id,
                fraud_probability=round(shadow_prob, 4),
                risk_level=get_risk_level(shadow_prob),
                model_version=shadow_version or "unknown",
                processing_time_ms=round(shadow_ms, 2),
            )
        except Exception as e:
            logger.warning(f"Shadow prediction failed: {e}")

    # SHAP explanation: extract top risk factors for decision transparency
    top_risk_factors = None
    explanation_data = None
    if fraud_explainer is not None and fraud_explainer._is_fitted:
        try:
            t_shap_start = time.perf_counter()
            shap_result = fraud_explainer.explain_prediction(df)
            top_risk_factors = shap_result["per_prediction"][0]["top_risk_factors"]
            shap_ms = (time.perf_counter() - t_shap_start) * 1000
            logger.info(f"SHAP explanation in {shap_ms:.1f}ms")
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

    return FraudPredictionResponse(
        transaction_id=transaction.transaction_id,
        fraud_probability=round(fraud_prob, 4),
        is_fraud=fraud_prob > 0.5,
        risk_level=risk_level,
        confidence_score=confidence,
        explanation=shadow_result,
        top_risk_factors=top_risk_factors,
        processing_time_ms=round(t_total_ms, 2),
        model_version=production_version or "ensemble-v1.0",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_request: BatchTransactionRequest,
    _: None = Depends(check_model_dependency),
):
    start_ms = time.time() * 1000
    preds, frauds, highs = [], 0, 0
    try:
        for tx in batch_request.transactions:
            df = preprocess_transaction(tx)
            prob = ensemble_model.predict_proba(df)[0, 1]
            risk = get_risk_level(prob)
            if prob > 0.5:
                frauds += 1
            if risk in ["HIGH", "CRITICAL"]:
                highs += 1
            preds.append(
                FraudPredictionResponse(
                    transaction_id=tx.transaction_id,
                    fraud_probability=round(prob, 4),
                    is_fraud=prob > 0.5,
                    risk_level=risk,
                    confidence_score=get_confidence_score(prob),
                    processing_time_ms=0.0,
                    model_version="ensemble-v1.0",
                    timestamp=datetime.now().isoformat(),
                )
            )

        total_ms = time.time() * 1000 - start_ms
        avg_ms = total_ms / len(preds)
        for p in preds:
            p.processing_time_ms = round(avg_ms, 2)

        summary = {
            "total_transactions": len(preds),
            "fraud_detected": frauds,
            "high_risk_transactions": highs,
            "fraud_rate": round(frauds / len(preds), 4),
            "avg_processing_time_ms": round(avg_ms, 2),
        }

        return BatchPredictionResponse(
            predictions=preds,
            batch_summary=summary,
            total_processing_time_ms=round(total_ms, 2),
        )
    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")


@app.get("/monitoring/report")
async def monitoring_report():
    """Return the full model monitoring report."""
    if model_monitor is None:
        raise HTTPException(status_code=503, detail="Model monitor not initialized")
    try:
        return model_monitor.generate_monitoring_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/drift")
async def monitoring_drift():
    """Return drift status for all tracked features."""
    if model_monitor is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Monitor or model not ready")
    try:
        # Use a small reference sample for drift comparison
        import pandas as pd
        ref_data = getattr(model_monitor, "reference_data", pd.DataFrame())
        if ref_data.empty:
            return {"message": "No reference data available for drift comparison", "drift_reports": []}
        # Compare against accumulated prediction distribution
        pred_history = model_monitor.prediction_history
        return {
            "prediction_distribution": pred_history[-20:] if pred_history else [],
            "drift_threshold": model_monitor.drift_threshold,
            "total_predictions_tracked": len(pred_history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/versions")
async def monitoring_versions():
    """List all registered model versions from the manifest."""
    from src.models.version_manager import list_versions, get_production_version, get_shadow_version
    return {
        "production": get_production_version(),
        "shadow": get_shadow_version(),
        "versions": list_versions(),
    }


@app.get("/metrics")
async def metrics():
    uptime = time.time() - start_time
    avg_time = total_processing_time / request_count if request_count else 0
    return {
        "requests_processed": request_count,
        "average_processing_time_ms": round(avg_time, 2),
        "total_processing_time_ms": round(total_processing_time, 2),
        "uptime_seconds": round(uptime, 2),
        "requests_per_second": round(request_count / uptime, 2)
        if uptime
        else 0,
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/reload-models")
async def reload_models(
    version: Optional[str] = None,
    role: Optional[str] = None,
):
    """
    Reload models with optional version and role selection.

    - No params: reload production from disk
    - ?version=X: load version X (must exist in version_manifest.json)
    - ?version=X&role=production: set version X as production
    - ?version=X&role=shadow: set version X as shadow
    - ?version=none&role=shadow: clear shadow model
    """
    from src.models.version_manager import (
        set_production,
        set_shadow,
        list_versions,
    )

    if role and version:
        if role == "production":
            ok = set_production(version)
            if not ok:
                available = list(list_versions().keys())
                raise HTTPException(
                    status_code=404,
                    detail=f"Version {version} not found. Available: {available}",
                )
        elif role == "shadow":
            if version.lower() == "none":
                set_shadow(None)
            else:
                ok = set_shadow(version)
                if not ok:
                    available = list(list_versions().keys())
                    raise HTTPException(
                        status_code=404,
                        detail=f"Version {version} not found. Available: {available}",
                    )
        else:
            raise HTTPException(
                status_code=400,
                detail="role must be 'production' or 'shadow'",
            )
        # Reload to apply changes
        load_models()

    elif version:
        # Just load a specific version to inspect (doesn't change slots)
        from src.models.version_manager import load_version, get_version_info
        try:
            load_version(version)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(
            status_code=400,
            detail="Specify role=production or role=shadow to apply this version",
        )

    elif role and not version:
        raise HTTPException(
            status_code=400,
            detail="Specify version= when using role=",
        )

    else:
        # Default: reload production
        load_models()

    return {
        "message": "Models reloaded",
        "production_version": production_version,
        "shadow_version": shadow_version,
        "available_versions": list(list_versions().keys()),
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    # Preserve detail from HTTPException (e.g., version not found)
    if hasattr(exc, "detail") and exc.detail:
        return JSONResponse(status_code=404, content={"detail": str(exc.detail)})
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/predict", "/docs"],
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Fraud Detection API server...")
    uvicorn.run(
        "fraud_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
