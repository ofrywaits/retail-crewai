"""
בדיקות — מודל ML
"""
import pytest
import json
import joblib
import numpy as np
from pathlib import Path

ROOT          = Path(__file__).parent.parent
META_PATH     = ROOT / "models/model_meta.json"
MODEL_PATH    = ROOT / "models/model.pkl"
ENCODERS_PATH = ROOT / "models/encoders.json"


@pytest.fixture
def meta():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def model():
    return joblib.load(MODEL_PATH)

@pytest.fixture
def encoders():
    with open(ENCODERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_model_file_exists():
    assert MODEL_PATH.exists(), "model.pkl לא נמצא — הרץ run_flow.py"


def test_meta_file_exists():
    assert META_PATH.exists(), "model_meta.json לא נמצא"


def test_meta_has_required_keys(meta):
    for key in ["best_model", "random_forest", "logistic_regression", "feature_importances"]:
        assert key in meta, f"מפתח חסר ב-meta: {key}"


def test_rf_accuracy_above_threshold(meta):
    assert meta["random_forest"]["accuracy"] > 0.7, "Random Forest accuracy נמוך מ-70%"


def test_rf_f1_above_threshold(meta):
    assert meta["random_forest"]["f1"] > 0.7, "Random Forest F1 נמוך מ-70%"


def test_model_predicts(model):
    sample = np.array([[3, 120.0, 5, 6, 2, 0, 0, 1, 2, 3, 1]], dtype=float)
    pred = model.predict(sample)
    assert pred[0] in [0, 1], "ניבוי לא תקין — צריך להיות 0 או 1"


def test_model_predict_proba(model):
    sample = np.array([[3, 120.0, 5, 6, 2, 0, 0, 1, 2, 3, 1]], dtype=float)
    proba = model.predict_proba(sample)[0]
    assert len(proba) == 2, "predict_proba צריך להחזיר 2 הסתברויות"
    assert abs(sum(proba) - 1.0) < 0.001, "הסתברויות לא מסתכמות ל-1"


def test_encoders_has_required_keys(encoders):
    for key in ["encoders", "median_profit_threshold", "feature_cols"]:
        assert key in encoders, f"מפתח חסר ב-encoders: {key}"


def test_feature_cols_count(encoders):
    assert len(encoders["feature_cols"]) == 11, "מספר פיצ'רים שגוי"
