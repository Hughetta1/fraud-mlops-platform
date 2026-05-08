# Model version management.
# Every training run auto-generates a version tag → archived to models/archive/.
# version_manifest.json tracks the production and shadow slots.
#
# This is the foundation for Phase 5 shadow deployment:
# set_shadow(v2) → API loads both v1 (production) and v2 (shadow)
# → /predict compares both, logs differences → promote shadow to production when ready.

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_ARCHIVE_DIR = _MODELS_DIR / "archive"
_MANIFEST_PATH = _MODELS_DIR / "version_manifest.json"

# Active production slots — always these two fixed paths.
# Versioned copies live in archive/ with timestamped filenames.
_PRODUCTION_FE = _MODELS_DIR / "feature_engineering_pipeline.pkl"
_PRODUCTION_ENSEMBLE = _MODELS_DIR / "fraud_detection_ensemble.pkl"


def _ensure_dirs():
    _MODELS_DIR.mkdir(exist_ok=True); _ARCHIVE_DIR.mkdir(exist_ok=True)


def _generate_version():
    return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _load_manifest():
    _ensure_dirs()
    if _MANIFEST_PATH.exists():
        with open(_MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {"production": None, "shadow": None, "versions": {}}


def _save_manifest(m):
    _ensure_dirs()
    with open(_MANIFEST_PATH, "w") as f:
        json.dump(m, f, indent=2)


def register_version(fe_pipeline, ensemble, metrics, training_samples,
                     version=None, set_as_production=True):
    # Archive a new version and optionally promote it to production.
    version = version or _generate_version()
    _ensure_dirs()

    fe_archive = _ARCHIVE_DIR / f"feature_pipeline_{version}.pkl"
    ens_archive = _ARCHIVE_DIR / f"ensemble_{version}.pkl"
    joblib.dump(fe_pipeline, fe_archive)
    ensemble.save_ensemble(str(ens_archive))

    manifest = _load_manifest()
    manifest["versions"][version] = {
        "fe_pipeline": str(fe_archive.name),
        "ensemble": str(ens_archive.name),
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "training_samples": training_samples,
        "created_at": datetime.now().isoformat(),
    }

    if set_as_production:
        manifest["production"] = version
        joblib.dump(fe_pipeline, _PRODUCTION_FE)
        ensemble.save_ensemble(str(_PRODUCTION_ENSEMBLE))

    _save_manifest(manifest)
    print(f"Registered version: {version}")
    print(f"  Production: {manifest['production']}")
    print(f"  Total versions: {len(manifest['versions'])}")
    return version


def get_production_version():  return _load_manifest().get("production")
def get_shadow_version():      return _load_manifest().get("shadow")
def get_version_info(v):       return _load_manifest()["versions"].get(v)
def list_versions():           return _load_manifest().get("versions", {})


def load_version(version):
    manifest = _load_manifest()
    vinfo = manifest["versions"].get(version)
    if vinfo is None:
        raise FileNotFoundError(f"Version {version} not found")
    fe_path = _ARCHIVE_DIR / vinfo["fe_pipeline"]
    ens_path = _ARCHIVE_DIR / vinfo["ensemble"]
    if not fe_path.exists() or not ens_path.exists():
        raise FileNotFoundError(f"Files missing for {version}")
    fe = joblib.load(fe_path)
    from src.models.fraud_detector import EnsembleFraudDetector
    ensemble = EnsembleFraudDetector()
    ensemble.load_ensemble(str(ens_path))
    return fe, ensemble


def set_production(version):
    # Copy versioned files to the active production slots.
    manifest = _load_manifest()
    if version not in manifest["versions"]:
        print(f"Version {version} not found. Available: {list(manifest['versions'].keys())}")
        return False
    fe, ensemble = load_version(version)
    joblib.dump(fe, _PRODUCTION_FE)
    ensemble.save_ensemble(str(_PRODUCTION_ENSEMBLE))
    manifest["production"] = version
    _save_manifest(manifest)
    print(f"Production set to {version}")
    return True


def set_shadow(version):
    # Set or clear the shadow slot. version=None disables shadow mode.
    manifest = _load_manifest()
    if version is not None and version not in manifest["versions"]:
        print(f"Version {version} not found.")
        return False
    manifest["shadow"] = version
    _save_manifest(manifest)
    if version:
        print(f"Shadow set to {version}")
    else:
        print("Shadow cleared")
    return True


def load_production():
    if not _PRODUCTION_FE.exists() or not _PRODUCTION_ENSEMBLE.exists():
        return None, None
    fe = joblib.load(_PRODUCTION_FE)
    from src.models.fraud_detector import EnsembleFraudDetector
    ensemble = EnsembleFraudDetector()
    ensemble.load_ensemble(str(_PRODUCTION_ENSEMBLE))
    return fe, ensemble


def load_shadow():
    version = get_shadow_version()
    if version is None:
        return None, None
    try:
        return load_version(version)
    except Exception as e:
        print(f"Failed to load shadow {version}: {e}")
        return None, None


_ensure_dirs()
