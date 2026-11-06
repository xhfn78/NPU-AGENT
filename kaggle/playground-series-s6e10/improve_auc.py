"""고정 검증 분할에서 후보를 비교하고 선택한 모델로 제출 파일을 만듭니다.

python improve_auc.py --phase tune
python improve_auc.py --phase refit
모델/가중치 선택에 사용한 검증 AUC이므로 최종 일반화 성능과 구분해야 합니다.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
SEED = 42
RATING_COLUMNS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]
CANDIDATES = {
    "xgb_fine_bins": {
        "kind": "xgb", "features": "raw", "settings": {
            "n_estimators": 3500, "learning_rate": 0.04, "max_depth": 5,
            "min_child_weight": 20, "subsample": 0.9, "colsample_bytree": 1.0,
            "reg_lambda": 10.0, "max_bin": 2048,
        },
    },
    "xgb_engineered": {
        "kind": "xgb", "features": "engineered", "settings": {
            "n_estimators": 3500, "learning_rate": 0.04, "max_depth": 5,
            "min_child_weight": 20, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_lambda": 10.0, "max_bin": 1024,
        },
    },
    "cat_engineered": {
        "kind": "cat", "features": "engineered", "settings": {
            "iterations": 3000, "learning_rate": 0.06, "depth": 7,
            "l2_leaf_reg": 5.0, "one_hot_max_size": 10,
        },
    },
    "xgb_targetencoded": {
        "kind": "xgb", "features": "targetencoded", "settings": {
            "n_estimators": 3000, "learning_rate": 0.04, "max_depth": 5,
            "min_child_weight": 20, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_lambda": 10.0, "max_bin": 512,
        },
    },
    "cat_targetencoded": {
        "kind": "cat", "features": "targetencoded", "settings": {
            "iterations": 3000, "learning_rate": 0.06, "depth": 7,
            "l2_leaf_reg": 5.0, "one_hot_max_size": 10,
        },
    },
    "xgb_context_encoded": {
        "kind": "xgb", "features": "context_targetencoded", "settings": {
            "n_estimators": 3000, "learning_rate": 0.04, "max_depth": 5,
            "min_child_weight": 30, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_lambda": 15.0, "max_bin": 512,
        },
    },
    "xgb_route_profiles": {
        "kind": "xgb", "features": "targetencoded", "route_profiles": True, "settings": {
            "n_estimators": 3000, "learning_rate": 0.04, "max_depth": 5,
            "min_child_weight": 30, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_lambda": 15.0, "max_bin": 512,
        },
    },
    "xgb_context_gpu": {
        "kind": "xgb", "features": "context_targetencoded", "device": "cuda", "settings": {
            "n_estimators": 3000, "learning_rate": 0.04, "max_depth": 5,
            "min_child_weight": 30, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_lambda": 15.0, "max_bin": 512,
        },
    },
    "xgb_profiles_gpu": {
        "kind": "xgb", "features": "targetencoded", "device": "cuda", "route_profiles": True,
        "settings": {
            "n_estimators": 3000, "learning_rate": 0.04, "max_depth": 5,
            "min_child_weight": 30, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_lambda": 15.0, "max_bin": 512,
        },
    },
    "cat_te_gpu": {
        "kind": "cat", "features": "targetencoded", "device": "cuda", "settings": {
            "iterations": 3000, "learning_rate": 0.07, "depth": 7,
            "l2_leaf_reg": 5.0, "one_hot_max_size": 10, "border_count": 128,
        },
    },
    "cat_context_gpu": {
        "kind": "cat", "features": "context_targetencoded", "device": "cuda", "settings": {
            "iterations": 3000, "learning_rate": 0.07, "depth": 8,
            "l2_leaf_reg": 10.0, "one_hot_max_size": 10, "border_count": 128,
        },
    },
}

for depth in [4, 6, 7]:
    CANDIDATES[f"xgb_context_d{depth}_gpu"] = {
        **CANDIDATES["xgb_context_gpu"],
        "settings": {
            **CANDIDATES["xgb_context_gpu"]["settings"],
            "max_depth": depth,
            "min_child_weight": 40 if depth >= 6 else 20,
        },
    }

CANDIDATES["xgb_service_gpu"] = {
    **CANDIDATES["xgb_context_gpu"],
    "features": "service_context_targetencoded",
    "settings": {**CANDIDATES["xgb_context_gpu"]["settings"], "max_depth": 4},
}
CANDIDATES["cat_service_gpu"] = {
    **CANDIDATES["cat_context_gpu"], "features": "service_context_targetencoded",
}
CANDIDATES["xgb_service_d6_gpu"] = {
    **CANDIDATES["xgb_service_gpu"],
    "settings": {**CANDIDATES["xgb_service_gpu"]["settings"], "max_depth": 6},
}
CANDIDATES["xgb_service_routes_gpu"] = {
    **CANDIDATES["xgb_service_d6_gpu"], "route_profiles": "context",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def combine(predictions, weights, method="probability"):
    value = np.zeros_like(next(iter(predictions.values())), dtype="float64")
    for name, weight in weights.items():
        p = predictions[name]
        if method == "logit":
            p = np.clip(p, 1e-6, 1 - 1e-6)
            p = np.log(p / (1 - p))
        value += weight * p
    return 1 / (1 + np.exp(-value)) if method == "logit" else value


def make_features(data, mode):
    """정답을 참조하지 않는 파생 변수만 만듭니다."""
    X = data.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    if mode in ["context_targetencoded", "service_context_targetencoded"]:
        X["travel_context"] = (
            X["Class"].astype("string") + "|" + X["Type of Travel"].astype("string")
            + "|" + X["Customer Type"].astype("string")
        )
        for numeric in ["Flight Distance", "Age"]:
            for category in ["Class", "Type of Travel", "Customer Type"]:
                X[f"{numeric}_by_{category}"] = (
                    X[numeric].astype("string") + "|" + X[category].astype("string")
                )
        if mode == "service_context_targetencoded":
            for rating in RATING_COLUMNS:
                X[f"{rating}_by_travel_context"] = (
                    X[rating].astype("string") + "|" + X["travel_context"]
                )
            for a, b in [
                ("Inflight wifi service", "Online boarding"),
                ("Seat comfort", "Inflight entertainment"),
                ("Seat comfort", "Cleanliness"),
                ("Inflight entertainment", "On-board service"),
            ]:
                X[f"{a}_by_{b}"] = X[a].astype("string") + "|" + X[b].astype("string")
    if mode == "engineered":
        service = X[RATING_COLUMNS]
        X["service_mean"] = service.mean(axis=1)
        X["service_std"] = service.std(axis=1)
        X["service_min"] = service.min(axis=1)
        X["service_max"] = service.max(axis=1)
        X["service_high_count"] = (service >= 4).sum(axis=1)
        X["service_low_count"] = (service <= 2).sum(axis=1)
        X["service_zero_count"] = (service == 0).sum(axis=1)
        X["delay_difference"] = X["Arrival Delay in Minutes"] - X["Departure Delay in Minutes"]
        for col in ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
            X[f"{col}_log1p"] = np.log1p(X[col])
            X[f"{col}_is_missing"] = X[col].isna().astype("int8")
        # 반복 등장하는 정확한 수치값의 효과와 숫자 자리별 패턴도 실험합니다.
        for col in ["Age", "Flight Distance"]:
            X[f"{col}_category"] = X[col].astype("string")
        for divisor in [10, 100]:
            X[f"distance_mod_{divisor}"] = X["Flight Distance"] % divisor
            X[f"distance_floor_{divisor}"] = X["Flight Distance"] // divisor
        X["travel_context"] = (
            X["Class"].astype("string") + "|" + X["Type of Travel"].astype("string")
            + "|" + X["Customer Type"].astype("string")
        )
    return X


def prepare(train_X, other_X, kind):
    """범주 매핑은 해당 학습 부분에서만 만들고 다른 데이터에 적용합니다."""
    train_X, other_X = train_X.copy(), other_X.copy()
    cats = train_X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    schema = {}
    for col in cats:
        a = train_X[col].astype("string").fillna("__MISSING__")
        b = other_X[col].astype("string").fillna("__MISSING__")
        if kind == "xgb":
            levels = sorted(set(a.tolist()) | {"__MISSING__"})
            schema[col] = levels
            dtype = pd.CategoricalDtype(levels)
            train_X[col] = a.astype(dtype)
            other_X[col] = b.where(b.isin(levels), pd.NA).astype(dtype)
        else:
            train_X[col], other_X[col] = a.astype(str), b.astype(str)
    return train_X, other_X, cats, schema


def add_target_encoding(train_X, other_X, y):
    """검증 라벨을 쓰지 않고, 학습 행도 내부 5개 fold로 교차 인코딩합니다."""
    train_X, other_X = train_X.copy(), other_X.copy()
    columns = train_X.columns.tolist()
    # 긴 문자열 정렬 비용을 줄이려고 먼저 범주 ID로 바꿉니다.
    # ID는 단지 범주 식별용이며, 순서나 정답 정보를 포함하지 않습니다.
    train_codes, other_codes, levels_by_column = [], [], {}
    for col in columns:
        codes, levels = pd.factorize(train_X[col], sort=False, use_na_sentinel=False)
        train_codes.append(codes)
        other_codes.append(pd.Index(levels).get_indexer(other_X[col]))
        levels_by_column[col] = levels
    a = np.column_stack(train_codes).astype("int32")
    b = np.column_stack(other_codes).astype("int32")
    encoder = TargetEncoder(
        smooth="auto", target_type="binary",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    )
    # fit().transform()는 자기 정답이 인코딩에 들어가므로 사용하지 않습니다.
    train_values = encoder.fit_transform(a, y)
    other_values = encoder.transform(b)
    for i, col in enumerate(columns):
        train_X[f"TE_{col}"] = train_values[:, i]
        other_X[f"TE_{col}"] = other_values[:, i]
    conditional = [c for c in columns if "_by_" in c or c == "travel_context"]
    train_X = train_X.drop(columns=conditional)
    other_X = other_X.drop(columns=conditional)
    return train_X, other_X, {"encoder": encoder, "source_levels": levels_by_column}


def add_route_profiles(train_X, other_X, conditional=False):
    """같은 거리의 서비스 평가 분포를 학습 부분에서만 계산합니다. 정답은 사용하지 않습니다."""
    train_X, other_X = train_X.copy(), other_X.copy()
    if conditional:
        columns = ["Online boarding", "Inflight wifi service", "Inflight entertainment", "Cleanliness"]
        keys = ["Flight Distance", "Class"]
        profile = train_X.groupby(keys)[columns].mean()
        counts = train_X.groupby(keys).size()
        for frame in [train_X, other_X]:
            index = pd.MultiIndex.from_frame(frame[keys])
            for col in columns:
                frame[f"route_class_mean_{col}"] = profile[col].reindex(index).to_numpy()
            frame["route_class_log_count"] = np.log1p(counts.reindex(index).fillna(0).to_numpy())
        return train_X, other_X, {"means": profile, "counts": counts, "keys": keys}
    profile = train_X.groupby("Flight Distance")[RATING_COLUMNS].mean()
    counts = train_X["Flight Distance"].value_counts()
    for col in RATING_COLUMNS:
        name = f"route_mean_{col}"
        train_X[name] = train_X["Flight Distance"].map(profile[col])
        other_X[name] = other_X["Flight Distance"].map(profile[col])
    train_X["route_log_count"] = np.log1p(train_X["Flight Distance"].map(counts).fillna(0))
    other_X["route_log_count"] = np.log1p(other_X["Flight Distance"].map(counts).fillna(0))
    return train_X, other_X, {"means": profile, "counts": counts}


def build(config, cats, iterations=None):
    settings = dict(config["settings"])
    if config["kind"] == "xgb":
        if iterations is not None:
            settings["n_estimators"] = iterations
        return XGBClassifier(
            **settings, objective="binary:logistic", eval_metric="auc",
            tree_method="hist", enable_categorical=True, device=config.get("device", "cpu"),
            random_state=SEED, n_jobs=8,
            early_stopping_rounds=120 if iterations is None else None,
        )
    if iterations is not None:
        settings["iterations"] = iterations
    device_settings = {"task_type": "GPU", "devices": "0"} if config.get("device") == "cuda" else {"task_type": "CPU"}
    return CatBoostClassifier(
        **settings, loss_function="Logloss", eval_metric="AUC", cat_features=cats,
        random_seed=SEED, thread_count=8, allow_writing_files=False, verbose=200,
        **device_settings,
    )


def load_data():
    data = pd.read_csv(ROOT / "train.csv")
    y = data.satisfaction.astype(str).str.lower().map({"true": 1, "false": 0})
    if y.isna().any():
        raise ValueError("Unknown target value")
    return data, y.astype("int8")


def tune(out, names):
    data, y = load_data()
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    np.save(out / "validation_indices.npy", va)
    for name in names:
        config = CANDIDATES[name]
        metrics_path = out / f"{name}_metrics.json"
        if metrics_path.exists():
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            if saved["config"] != config:
                raise ValueError(f"Cached config changed: {name}; use a new --output directory")
            print("Cached:", name, saved["validation_roc_auc"], flush=True)
            continue
        X = make_features(data, config["features"])
        a, b = X.iloc[tr].copy(), X.iloc[va].copy()
        if config["features"].endswith("targetencoded"):
            a, b, encoder = add_target_encoding(a, b, y.iloc[tr])
            joblib.dump(encoder, out / f"{name}_validation_encoder.joblib")
        if config.get("route_profiles"):
            a, b, profile = add_route_profiles(a, b, config["route_profiles"] == "context")
            joblib.dump(profile, out / f"{name}_validation_profiles.joblib")
        a, b, cats, _ = prepare(a, b, config["kind"])
        model = build(config, cats)
        print("Starting", name, "features", len(a.columns), flush=True)
        if config["kind"] == "xgb":
            model.fit(a, y.iloc[tr], eval_set=[(b, y.iloc[va])], verbose=200)
            best = model.best_iteration + 1
        else:
            model.fit(a, y.iloc[tr], eval_set=(b, y.iloc[va]), early_stopping_rounds=120, use_best_model=True)
            best = model.get_best_iteration() + 1
        prediction = model.predict_proba(b)[:, 1]
        score = float(roc_auc_score(y.iloc[va], prediction))
        np.save(out / f"{name}_validation.npy", prediction)
        suffix = "json" if config["kind"] == "xgb" else "cbm"
        model.save_model(str(out / f"{name}_validation_model.{suffix}"))
        write_json(metrics_path, {"name": name, "validation_roc_auc": score, "best_iterations": int(best), "config": config})
        print("RESULT", name, f"AUC={score:.8f}", "iterations", best, flush=True)
    select(out, y.iloc[va].to_numpy())


def select(out, y):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*_metrics.json"))]
    predictions = {r["name"]: np.load(out / f"{r['name']}_validation.npy") for r in records}
    choices = [{"weights": {r["name"]: 1.0}, "method": "probability", "validation_roc_auc": r["validation_roc_auc"]} for r in records]
    names = list(predictions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for weight in [0.25, 0.5, 0.75]:
                for method in ["probability", "logit"]:
                    weights = {a: weight, b: 1 - weight}
                    score = float(roc_auc_score(y, combine(predictions, weights, method)))
                    choices.append({"weights": weights, "method": method, "validation_roc_auc": score})
    choices.sort(key=lambda r: r["validation_roc_auc"], reverse=True)
    write_json(out / "selection.json", {"seed": SEED, "validation_size": 0.2,
        "score_is_used_for_selection": True, "selected": choices[0], "all_candidates": records, "blend_scores": choices})
    print("SELECTED", json.dumps(choices[0]), flush=True)


def refit(out):
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    records = {r["name"]: r for r in selection["all_candidates"]}
    data, y = load_data()
    test = pd.read_csv(ROOT / "test.csv")
    submission = pd.read_csv(ROOT / "sample_submission.csv")
    if not submission.id.equals(test.id):
        raise ValueError("Submission IDs do not match")
    predictions = {}
    for name, weight in selection["selected"]["weights"].items():
        record = records[name]
        config = record["config"]
        a, b = make_features(data, config["features"]), make_features(test, config["features"])
        if config["features"].endswith("targetencoded"):
            a, b, encoder = add_target_encoding(a, b, y)
            joblib.dump(encoder, out / f"{name}_full_encoder.joblib")
        if config.get("route_profiles"):
            a, b, profile = add_route_profiles(a, b, config["route_profiles"] == "context")
            joblib.dump(profile, out / f"{name}_full_profiles.joblib")
        a, b, cats, schema = prepare(a, b, config["kind"])
        print("Refitting", name, "iterations", record["best_iterations"], flush=True)
        model = build(config, cats, record["best_iterations"])
        model.fit(a, y)
        prediction = model.predict_proba(b)[:, 1]
        predictions[name] = prediction
        np.save(out / f"{name}_test.npy", prediction)
        suffix = "json" if config["kind"] == "xgb" else "cbm"
        model.save_model(str(out / f"{name}_full_model.{suffix}"))
        write_json(out / f"{name}_schema.json", {"columns": a.columns.tolist(), "categories": schema})
    combined = combine(predictions, selection["selected"]["weights"], selection["selected"].get("method", "probability"))
    if not np.isfinite(combined).all() or not ((combined >= 0) & (combined <= 1)).all():
        raise ValueError("Invalid probabilities")
    submission["satisfaction"] = combined
    submission.to_csv(out / "submission_improved.csv", index=False)
    print("Saved", out / "submission_improved.csv", flush=True)


def check_split(out):
    """선택한 설정과 트리 수를 고정한 뒤 다른 seed 분할에서 한 번 확인합니다."""
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    records = {r["name"]: r for r in selection["all_candidates"]}
    data, y = load_data()
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=2026)
    predictions = {}
    scores = {}
    for name, weight in selection["selected"]["weights"].items():
        record, config = records[name], records[name]["config"]
        X = make_features(data, config["features"])
        a, b = X.iloc[tr].copy(), X.iloc[va].copy()
        if config["features"].endswith("targetencoded"):
            a, b, _ = add_target_encoding(a, b, y.iloc[tr])
        if config.get("route_profiles"):
            a, b, _ = add_route_profiles(a, b, config["route_profiles"] == "context")
        a, b, cats, _ = prepare(a, b, config["kind"])
        model = build(config, cats, record["best_iterations"])
        print("Checking second split", name, flush=True)
        model.fit(a, y.iloc[tr])
        probability = model.predict_proba(b)[:, 1]
        scores[name] = float(roc_auc_score(y.iloc[va], probability))
        predictions[name] = probability
        print("CHECK RESULT", name, scores[name], flush=True)
    combined = combine(predictions, selection["selected"]["weights"], selection["selected"].get("method", "probability"))
    result = {"seed": 2026, "validation_size": 0.2, "weights_frozen_before_check": True,
        "single_models": scores, "combined_roc_auc": float(roc_auc_score(y.iloc[va], combined)),
        "note": "Additional shuffled holdout; not an independent test set or Kaggle score."}
    write_json(out / "second_split_check.json", result)
    print("SECOND SPLIT", json.dumps(result), flush=True)


def verify(out):
    """저장 모델과 전처리를 다시 불러와 실제 제출 파일과 예측이 일치하는지 확인합니다."""
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    records = {r["name"]: r for r in selection["all_candidates"]}
    test = pd.read_csv(ROOT / "test.csv")
    submission = pd.read_csv(out / "submission_improved.csv")
    assert submission.columns.tolist() == ["id", "satisfaction"]
    assert submission.id.equals(test.id)
    assert submission.id.is_unique
    assert submission.satisfaction.between(0, 1).all()
    assert np.isfinite(submission.satisfaction).all()
    sample = test.iloc[:1000]
    predictions = {}
    for name in selection["selected"]["weights"]:
        config = records[name]["config"]
        X = make_features(sample, config["features"])
        if config["features"].endswith("targetencoded"):
            bundle = joblib.load(out / f"{name}_full_encoder.joblib")
            levels = bundle["source_levels"]
            codes = np.column_stack([pd.Index(v).get_indexer(X[c]) for c, v in levels.items()]).astype("int32")
            values = bundle["encoder"].transform(codes)
            for i, col in enumerate(levels):
                X[f"TE_{col}"] = values[:, i]
            conditional = [c for c in levels if "_by_" in c or c == "travel_context"]
            X = X.drop(columns=conditional)
        if config.get("route_profiles"):
            bundle = joblib.load(out / f"{name}_full_profiles.joblib")
            if config["route_profiles"] == "context":
                index = pd.MultiIndex.from_frame(X[bundle["keys"]])
                for col in bundle["means"].columns:
                    X[f"route_class_mean_{col}"] = bundle["means"][col].reindex(index).to_numpy()
                X["route_class_log_count"] = np.log1p(bundle["counts"].reindex(index).fillna(0).to_numpy())
            else:
                for col in bundle["means"].columns:
                    X[f"route_mean_{col}"] = X["Flight Distance"].map(bundle["means"][col])
                X["route_log_count"] = np.log1p(X["Flight Distance"].map(bundle["counts"]).fillna(0))
        schema = json.loads((out / f"{name}_schema.json").read_text(encoding="utf-8"))
        if config["kind"] == "xgb":
            for col, levels in schema["categories"].items():
                values = X[col].astype("string").fillna("__MISSING__")
                X[col] = values.where(values.isin(levels), pd.NA).astype(pd.CategoricalDtype(levels))
            model = XGBClassifier(enable_categorical=True)
            model.load_model(out / f"{name}_full_model.json")
            model.set_params(device="cpu")
        else:
            for col in X.select_dtypes(include=["object", "string", "category"]).columns:
                X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)
            model = CatBoostClassifier()
            model.load_model(str(out / f"{name}_full_model.cbm"))
        predictions[name] = model.predict_proba(X.loc[:, schema["columns"]])[:, 1]
    reproduced = combine(predictions, selection["selected"]["weights"], selection["selected"].get("method", "probability"))
    error = float(np.abs(reproduced - submission.satisfaction.iloc[:len(sample)].to_numpy()).max())
    assert error < 1e-6, f"Saved-model prediction mismatch: {error}"
    result = {"rows": len(submission), "id_order_matches_test": True, "missing_predictions": 0,
        "min_probability": float(submission.satisfaction.min()), "max_probability": float(submission.satisfaction.max()),
        "reload_check_rows": len(sample), "reload_check_max_error": error}
    write_json(out / "verification.json", result)
    print("VERIFIED", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["tune", "check", "refit", "verify"], default="tune")
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES),
        default=["cat_service_gpu", "xgb_service_routes_gpu"])
    parser.add_argument("--output", default="outputs/improve_960_gpu")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    # 다른 원본 데이터에서 저장된 실험을 재사용하지 않도록 해시를 확인합니다.
    digest = hashlib.sha256((ROOT / "train.csv").read_bytes()).hexdigest()
    manifest = out / "data_manifest.json"
    info = {"train_sha256": digest, "seed": SEED, "validation_size": 0.2}
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != info:
        raise ValueError("Dataset/split changed; use a new --output directory")
    write_json(manifest, info)
    if args.phase == "tune":
        tune(out, args.candidates)
    elif args.phase == "check":
        check_split(out)
    elif args.phase == "verify":
        verify(out)
    else:
        refit(out)


if __name__ == "__main__":
    main()
