"""고정한 4개 예측을 조건별 로지스틱 모델로 결합합니다.

confirm: outer train 안에서 base/meta를 분리합니다. outer valid 레이블은 평가에만 사용합니다.
refit: seed42 검증 예측으로 메타 모델을 학습하고 전체 학습 모델의 test 예측에 적용합니다.
기존 검증 분할을 여러 번 사용했으므로 독립 테스트 성능으로 해석하지 않습니다.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from threadpoolctl import threadpool_limits

import challenge_models as challenge
import improve_963 as previous


ROOT = Path(__file__).resolve().parent
SOURCE = previous.OUT
OUT = challenge.OUT / "stacking"
NAMES = ["lgb_original_deep", "cat_original_stats", "keras_original_prior", "previous_960"]


def write(path, value):
    previous.write(path, value)


def context_values(rows):
    return rows["Class"].astype(str) + "|" + rows["Type of Travel"].astype(str) + "|" + rows["Customer Type"].astype(str)


def features(predictions, rows, levels):
    p = np.clip(predictions, 1e-6, 1-1e-6)
    logits = np.log(p / (1-p))
    codes = pd.Index(levels).get_indexer(context_values(rows))
    onehot = np.zeros((len(rows), len(levels)), dtype="float64")
    known = np.flatnonzero(codes >= 0)
    onehot[known, codes[known]] = 1
    return np.column_stack([logits, p, p*(1-p), onehot,
        (logits[:, :, None] * onehot[:, None, :]).reshape(len(p), -1)])


def fit_meta(predictions, rows, target):
    levels = sorted(context_values(rows).unique().tolist())
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, solver="lbfgs"))
    with threadpool_limits(limits=8):
        model.fit(features(predictions, rows, levels), target)
    return {"model": model, "levels": levels, "names": NAMES, "C": 0.1}


def predict_meta(bundle, predictions, rows):
    with threadpool_limits(limits=8):
        return bundle["model"].predict_proba(features(predictions, rows, bundle["levels"]))[:, 1]


def keras(a, b, y, folder, epochs):
    folder.mkdir(parents=True, exist_ok=True)
    a, b = pd.get_dummies(a, dtype="float32"), pd.get_dummies(b, dtype="float32")
    b = b.reindex(columns=a.columns, fill_value=0)
    median = a.median().fillna(0)
    scaler = QuantileTransformer(n_quantiles=256, subsample=100000, output_distribution="normal", random_state=42)
    np.save(folder / "X_train.npy", scaler.fit_transform(a.fillna(median)).astype("float32"))
    np.save(folder / "X_other.npy", scaler.transform(b.fillna(median)).astype("float32"))
    np.save(folder / "y_train.npy", np.asarray(y))
    joblib.dump({"columns": a.columns.tolist(), "median": median, "scaler": scaler}, folder / "dense_preprocessing.joblib")
    subprocess.run([r"C:\Users\Admin\miniconda3\envs\tf29x-gpu\python.exe", "-u", str(ROOT / "keras_candidate.py"),
        "--folder", str(folder), "--epochs", str(epochs), "--fixed"], check=True)
    return np.load(folder / "probability.npy")


def component_predictions(data, y, tr, other, original, other_original, phase, other_indices=None):
    folder = OUT / phase
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "train_indices.npy", tr)
    if other_indices is not None:
        np.save(folder / "other_indices.npy", other_indices)
    predictions = []
    for name in NAMES:
        path = folder / f"{name}.npy"
        if path.exists():
            prediction = np.load(path)
        elif phase in ["outer", "full"] and name in ["lgb_original_deep", "previous_960"]:
            # 동일 seed2026 분할/전체 학습으로 계산한 기존 예측만 재사용합니다.
            part = "confirm" if phase == "outer" else "test"
            prediction = np.load(SOURCE / f"{name}_{part}.npy")
            np.save(path, prediction)
        elif name == "previous_960":
            prediction = previous.previous_predict(data, y, tr, other_indices)
            np.save(path, prediction)
        else:
            record = json.loads((SOURCE / f"{name}_candidate.json").read_text(encoding="utf-8"))
            config = record["config"]
            a, b, cats, preprocessing = previous.inputs(data.iloc[tr], other, y.iloc[tr], original.iloc[tr], other_original, config)
            print("STACK COMPONENT", phase, name, len(a), len(b), flush=True)
            if config["kind"] == "keras":
                prediction = keras(a, b, y.iloc[tr], folder / name, record["iterations"])
            else:
                model, _ = previous.fit(config, a, y.iloc[tr], cats, iterations=record["iterations"])
                prediction = model.predict_proba(b)[:, 1]
                if config["kind"] == "lgb":
                    model_path = folder / f"{name}.txt"
                    model.booster_.save_model(str(model_path))
                    reloaded = previous.lgb.Booster(model_file=str(model_path))
                    check = reloaded.predict(b.iloc[:1000])
                else:
                    model_path = folder / f"{name}.cbm"
                    model.save_model(str(model_path))
                    reloaded = previous.CatBoostClassifier()
                    reloaded.load_model(str(model_path))
                    check = reloaded.predict_proba(b.iloc[:1000])[:, 1]
                error = float(np.abs(check - prediction[:1000]).max())
                assert error < 1e-6
                write(folder / f"{name}_reload.json", {"rows": len(check), "max_error": error})
            joblib.dump(preprocessing, folder / f"{name}_preprocessing.joblib")
            np.save(path, prediction)
        assert len(prediction) == len(other) and np.isfinite(prediction).all()
        predictions.append(prediction)
    return np.column_stack(predictions)


def confirm(data, y, original):
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=2026)
    base_tr, meta_tr = train_test_split(tr, test_size=0.2, stratify=y.iloc[tr], random_state=314)
    assert not np.intersect1d(meta_tr, va).size and not np.intersect1d(base_tr, meta_tr).size
    inner = component_predictions(data, y, base_tr, data.iloc[meta_tr], original, original.iloc[meta_tr], "inner", meta_tr)
    bundle = fit_meta(inner, data.iloc[meta_tr], y.iloc[meta_tr])
    joblib.dump(bundle, OUT / "confirmation_meta.joblib")
    outer = component_predictions(data, y, tr, data.iloc[va], original, original.iloc[va], "outer", va)
    prediction = predict_meta(bundle, outer, data.iloc[va])
    np.save(OUT / "confirmation_prediction.npy", prediction)
    score = float(roc_auc_score(y.iloc[va], prediction))
    incumbent = float(roc_auc_score(y.iloc[va], challenge.incumbent("confirm")))
    result = {"seed": 2026, "base_train_rows": len(base_tr), "meta_train_rows": len(meta_tr), "validation_rows": len(va),
        "combined_auc": score, "incumbent_auc": incumbent, "difference": score-incumbent,
        "goal_reached": score >= 0.963, "C": 0.1, "contextual": True,
        "note": "Outer validation labels are excluded from base/meta training. Split reused for exploratory checks; not an independent test."}
    write(OUT / "confirmation.json", result)
    print("STACK CONFIRMATION", result, flush=True)


def refit(data, y, original):
    _, meta_rows = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    meta_predictions = np.column_stack([np.load(SOURCE / f"{n}_validation.npy") for n in NAMES])
    bundle = fit_meta(meta_predictions, data.iloc[meta_rows], y.iloc[meta_rows])
    joblib.dump(bundle, OUT / "full_meta.joblib")
    test = pd.read_csv(ROOT / "test.csv")
    full = component_predictions(data, y, np.arange(len(y)), test, original, previous.priors("test"), "full")
    prediction = predict_meta(bundle, full, test)
    restored = joblib.load(OUT / "full_meta.joblib")
    error = float(np.abs(predict_meta(restored, full[:1000], test.iloc[:1000]) - prediction[:1000]).max())
    assert error < 1e-10
    assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    submission = pd.read_csv(ROOT / "sample_submission.csv")
    assert submission.id.equals(test.id) and submission.id.is_unique
    submission["satisfaction"] = prediction
    path = OUT / "submission_stacking.csv"
    submission.to_csv(path, index=False)
    saved = pd.read_csv(path)
    assert saved.columns.tolist() == ["id", "satisfaction"] and saved.id.equals(test.id)
    export_error = float(np.abs(saved.satisfaction - prediction).max())
    assert export_error < 1e-10
    write(OUT / "verification.json", {"rows": len(saved), "ids_match": True, "missing": int(saved.satisfaction.isna().sum()),
        "min_probability": float(prediction.min()), "max_probability": float(prediction.max()),
        "meta_reload_max_error": error, "export_max_error": export_error})
    print("STACK SUBMISSION", path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["confirm", "refit"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # 모델 설정 또는 데이터가 달라지면 캐시를 재사용하지 않습니다.
    source_manifest = json.loads((SOURCE / "data_manifest.json").read_text(encoding="utf-8"))
    actual = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_manifest}
    assert actual == source_manifest, "Source data changed"
    records = {n: json.loads((SOURCE / f"{n}_candidate.json").read_text(encoding="utf-8")) for n in NAMES}
    manifest = {"source": actual, "records": records, "names": NAMES, "C": 0.1,
        "outer_seed": 2026, "inner_seed": 314, "meta_fraction": 0.2}
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest, "Configuration changed"
    write(manifest_path, manifest)
    data, y = previous.base.load_data()
    original = previous.priors("train")
    if args.phase == "confirm":
        confirm(data, y, original)
    else:
        refit(data, y, original)


if __name__ == "__main__":
    main()
