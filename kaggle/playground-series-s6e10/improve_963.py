"""0.963 목표의 후속 실험. 기존 improve_auc.py 및 결과는 보존합니다.

순서: --phase prepare, tune, confirm, refit, verify
외부 원본 데이터는 competition train/test와 다른 공개된 2020년 데이터입니다.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".local_packages"))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

import improve_auc as base


OUT = ROOT / "outputs" / "improve_963_gpu"
OLD = ROOT / "outputs" / "improve_960_gpu"
ORIGINAL = ROOT / "data" / "original_airline"
SOURCE_URL = "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction"
PRIOR_CONFIGS = {
    "xgb": {"kind": "xgb", "device": "cuda", "settings": {
        "n_estimators": 2000, "learning_rate": 0.05, "max_depth": 7,
        "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_lambda": 3.0, "max_bin": 256}},
    "cat": {"kind": "cat", "device": "cuda", "settings": {
        "iterations": 2000, "learning_rate": 0.07, "depth": 8,
        "l2_leaf_reg": 5.0, "one_hot_max_size": 10, "border_count": 128}},
}
CONFIGS = {
    "xgb_original_prior": {
        **base.CANDIDATES["xgb_service_routes_gpu"], "original_stats": False,
    },
    "cat_original_prior": {
        **base.CANDIDATES["cat_service_gpu"], "original_stats": False,
    },
    "lgb_original_prior": {
        "kind": "lgb", "features": "service_context_targetencoded", "route_profiles": "context",
        "original_stats": True, "settings": {
            "n_estimators": 3000, "learning_rate": 0.04, "num_leaves": 31,
            "min_child_samples": 100, "reg_lambda": 10.0,
            "subsample": 0.9, "subsample_freq": 1, "colsample_bytree": 0.9,
            "max_bin": 255,
        },
    },
    "keras_original_prior": {
        "kind": "keras", "features": "service_context_targetencoded", "route_profiles": "context",
        "original_stats": True, "settings": {"max_epochs": 60},
    },
}
CONFIGS["xgb_original_stats"] = {**CONFIGS["xgb_original_prior"], "original_stats": True}
CONFIGS["cat_original_stats"] = {**CONFIGS["cat_original_prior"], "original_stats": True}
CONFIGS["lgb_original_deep"] = {
    **CONFIGS["lgb_original_prior"],
    "settings": {**CONFIGS["lgb_original_prior"]["settings"], "num_leaves": 63,
        "min_child_samples": 150, "reg_lambda": 15.0, "learning_rate": 0.035},
}


def run_keras(a, b, y, name, epochs=None, y_valid=None):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    a, b = pd.get_dummies(a, dtype="float32"), pd.get_dummies(b, dtype="float32")
    b = b.reindex(columns=a.columns, fill_value=0)
    median = a.median().fillna(0)
    a, b = a.fillna(median), b.fillna(median)
    scaler = QuantileTransformer(n_quantiles=256, subsample=100000, output_distribution="normal", random_state=42)
    train_values = scaler.fit_transform(a).astype("float32")
    other_values = scaler.transform(b).astype("float32")
    joblib.dump({"median": median, "scaler": scaler, "columns": a.columns.tolist()}, folder / "dense_preprocessing.joblib")
    np.save(folder / "X_train.npy", train_values)
    np.save(folder / "X_other.npy", other_values)
    np.save(folder / "y_train.npy", np.asarray(y))
    command = [r"C:\Users\Admin\miniconda3\envs\tf29x-gpu\python.exe", "-u", str(ROOT / "keras_candidate.py"),
        "--folder", str(folder), "--epochs", str(epochs or 60)]
    if epochs:
        command += ["--fixed"]
    else:
        np.save(folder / "y_valid.npy", np.asarray(y_valid))
    subprocess.run(command, check=True)
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    return np.load(folder / "probability.npy"), metrics["epochs"]


def write(path, value):
    base.write_json(path, value)


def original_priors():
    source = pd.concat([pd.read_csv(ORIGINAL / "raw" / name) for name in ["train.csv", "test.csv"]], ignore_index=True)
    target = source.satisfaction.map({"satisfied": 1, "neutral or dissatisfied": 0})
    if target.isna().any():
        raise ValueError("Unexpected original target")
    target = target.astype("int8")
    train, y = base.load_data()
    test = pd.read_csv(ROOT / "test.csv")
    columns = base.make_features(train, "raw").columns.tolist()
    # 원본에만 있는 Inflight service, id, Unnamed: 0은 사용하지 않습니다.
    X = source.loc[:, columns].copy()
    queries = pd.concat([train[columns], test[columns]], ignore_index=True)
    tr, va = train_test_split(np.arange(len(source)), test_size=0.2, stratify=target, random_state=17)
    arrays, names, teacher_results = [], [], {}
    for name, config in PRIOR_CONFIGS.items():
        metrics_path = OUT / f"original_{name}_metrics.json"
        prediction_path = OUT / f"original_{name}_all.npy"
        if metrics_path.exists() and prediction_path.exists():
            probability = np.load(prediction_path)
            teacher_results[name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            a, b, cats, _ = base.prepare(X.iloc[tr], X.iloc[va], config["kind"])
            model = base.build(config, cats)
            print("ORIGINAL TEACHER", name, flush=True)
            if config["kind"] == "xgb":
                model.fit(a, target.iloc[tr], eval_set=[(b, target.iloc[va])], verbose=200)
                best = model.best_iteration + 1
            else:
                model.fit(a, target.iloc[tr], eval_set=(b, target.iloc[va]), early_stopping_rounds=100, use_best_model=True)
                best = model.get_best_iteration() + 1
            score = float(roc_auc_score(target.iloc[va], model.predict_proba(b)[:, 1]))
            print("ORIGINAL ONLY AUC", name, score, "iterations", best, flush=True)
            a, b, cats, schema = base.prepare(X, queries, config["kind"])
            model = base.build(config, cats, best)
            model.fit(a, target)
            if config["kind"] == "xgb":
                model.set_params(device="cpu")
            probability = model.predict_proba(b)[:, 1]
            np.save(prediction_path, probability)
            suffix = "json" if name == "xgb" else "cbm"
            model.save_model(str(OUT / f"original_{name}_model.{suffix}"))
            write(OUT / f"original_{name}_schema.json", {"columns": columns, "categories": schema})
            teacher_results[name] = {"original_validation_auc": score, "iterations": int(best), "config": config}
            write(metrics_path, teacher_results[name])
        arrays.extend([probability, np.log(np.clip(probability, 1e-6, 1-1e-6) / (1-np.clip(probability, 1e-6, 1-1e-6)))])
        names.extend([f"ORIG_{name}_probability", f"ORIG_{name}_logit"])

    # 외부 원본의 레이블만 사용하는 통계: 대회 검증/테스트 레이블은 참조하지 않습니다.
    groups = [["Flight Distance"], ["Age"], ["Online boarding"], ["Inflight wifi service"],
              ["Inflight entertainment"], ["Flight Distance", "Class"],
              ["Flight Distance", "Type of Travel"], ["Flight Distance", "Customer Type"], ["Age", "Class"]]
    stats = {}
    temp = X.assign(_target=target.to_numpy())
    for i, keys in enumerate(groups):
        table = temp.groupby(keys)._target.agg(["sum", "count"])
        mean = (table["sum"] + 20 * target.mean()) / (table["count"] + 20)
        index = pd.Index(queries[keys[0]]) if len(keys) == 1 else pd.MultiIndex.from_frame(queries[keys])
        arrays.extend([mean.reindex(index).fillna(target.mean()).to_numpy(), np.log1p(table["count"].reindex(index).fillna(0).to_numpy())])
        names.extend([f"ORIG_stats_{i}_target", f"ORIG_stats_{i}_logcount"])
        stats[str(i)] = {"keys": keys, "mean": mean, "count": table["count"]}
    values = np.column_stack(arrays).astype("float32")
    np.save(OUT / "original_features_train.npy", values[:len(train)])
    np.save(OUT / "original_features_test.npy", values[len(train):])
    joblib.dump(stats, OUT / "original_group_stats.joblib")
    write(OUT / "original_features.json", {"columns": names, "source_url": SOURCE_URL,
        "original_rows": len(source), "original_target_mean": float(target.mean()),
        "teacher_results": teacher_results})
    print("PREPARED ORIGINAL FEATURES", values.shape, flush=True)


def priors(part):
    names = json.loads((OUT / "original_features.json").read_text(encoding="utf-8"))["columns"]
    return pd.DataFrame(np.load(OUT / f"original_features_{part}.npy"), columns=names)


def inputs(train, other, y, original_train, original_other, config):
    a = base.make_features(train, config["features"])
    b = base.make_features(other, config["features"])
    a, b, encoder = base.add_target_encoding(a, b, y)
    profile = None
    if config.get("route_profiles"):
        a, b, profile = base.add_route_profiles(a, b, config["route_profiles"] == "context")
    original_columns = [c for c in original_train if config.get("original_stats") or not c.startswith("ORIG_stats_")]
    for col in original_columns:
        a[col] = original_train[col].to_numpy()
        b[col] = original_other[col].to_numpy()
    a, b, cats, schema = base.prepare(a, b, "cat" if config["kind"] == "cat" else "xgb")
    bundle = {"encoder": encoder, "profiles": profile, "categories": schema,
        "columns": a.columns.tolist(), "original_columns": original_columns, "config": config}
    return a, b, cats, bundle


def fit(config, a, y, cats, b=None, y_valid=None, iterations=None):
    if config["kind"] == "lgb":
        settings = {**config["settings"]}
        if iterations:
            settings["n_estimators"] = iterations
        model = lgb.LGBMClassifier(**settings, objective="binary", metric="auc", random_state=42,
            n_jobs=8, verbosity=-1, force_col_wise=True)
        if b is None:
            model.fit(a, y)
            best = iterations
        else:
            model.fit(a, y, eval_set=[(b, y_valid)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
            best = model.best_iteration_
    else:
        model = base.build(config, cats, iterations)
        if b is None:
            model.fit(a, y)
            best = iterations
        elif config["kind"] == "xgb":
            model.fit(a, y, eval_set=[(b, y_valid)], verbose=200)
            best = model.best_iteration + 1
        else:
            model.fit(a, y, eval_set=(b, y_valid), early_stopping_rounds=100, use_best_model=True)
            best = model.get_best_iteration() + 1
    return model, int(best)


def save_model(model, kind, name):
    if kind == "lgb":
        model.booster_.save_model(str(OUT / f"{name}.txt"))
    else:
        model.save_model(str(OUT / f"{name}.{'json' if kind == 'xgb' else 'cbm'}"))


def tune(names):
    data, y = base.load_data()
    original = priors("train")
    tr, va = train_test_split(np.arange(len(data)), test_size=0.2, stratify=y, random_state=42)
    assert np.array_equal(va, np.load(OLD / "validation_indices.npy"))
    previous = json.loads((OLD / "selection.json").read_text(encoding="utf-8"))["selected"]
    previous_predictions = {n: np.load(OLD / f"{n}_validation.npy") for n in previous["weights"]}
    old_pred = base.combine(previous_predictions, previous["weights"], previous["method"])
    np.save(OUT / "previous_960_validation.npy", old_pred)
    write(OUT / "previous_960_candidate.json", {"name": "previous_960", "validation_roc_auc": float(roc_auc_score(y.iloc[va], old_pred)), "previous_selection": previous})
    for name in names:
        config = CONFIGS[name]
        path = OUT / f"{name}_candidate.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["config"] != config:
                raise ValueError("Config changed; choose a new experiment directory/name")
            print("CACHED", name, cached["validation_roc_auc"], flush=True)
            continue
        a, b, cats, bundle = inputs(data.iloc[tr], data.iloc[va], y.iloc[tr], original.iloc[tr], original.iloc[va], config)
        print("TUNING", name, a.shape, flush=True)
        if config["kind"] == "keras":
            prediction, best = run_keras(a, b, y.iloc[tr], f"{name}_tune", y_valid=y.iloc[va])
        else:
            model, best = fit(config, a, y.iloc[tr], cats, b, y.iloc[va])
            prediction = model.predict_proba(b)[:, 1]
            save_model(model, config["kind"], f"{name}_validation_model")
        score = float(roc_auc_score(y.iloc[va], prediction))
        np.save(OUT / f"{name}_validation.npy", prediction)
        joblib.dump(bundle, OUT / f"{name}_validation_preprocessing.joblib")
        write(path, {"name": name, "config": config, "iterations": best, "validation_roc_auc": score})
        print("RESULT", name, score, "iterations", best, flush=True)
    select(y.iloc[va])


def select(y):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*_candidate.json"))]
    predictions = {r["name"]: np.load(OUT / f"{r['name']}_validation.npy") for r in records}
    choices = [{"weights": {r["name"]: 1.0}, "method": "probability", "auc": r["validation_roc_auc"]} for r in records]
    names = list(predictions)
    for count in [2, 3]:
        for members in itertools.combinations(names, count):
            weight_sets = [dict.fromkeys(members, 1 / count)] if count == 3 else [dict(zip(members, [w, 1-w])) for w in [0.25, 0.5, 0.75]]
            for weights in weight_sets:
                for method in ["probability", "logit"]:
                    score = float(roc_auc_score(y, base.combine(predictions, weights, method)))
                    choices.append({"weights": weights, "method": method, "auc": score})
    choices.sort(key=lambda c: c["auc"], reverse=True)
    # 수십만분의 몇 수준의 차이로 신경망/의존성/모델 수를 늘리지 않습니다.
    # 최고 관측값과 0.00005 이내에서는 신경망 수, 결합 모델 수가 적은 쪽을 우선합니다.
    record_map = {r["name"]: r for r in records}
    near_best = [c for c in choices if c["auc"] >= choices[0]["auc"] - 0.00005]
    chosen = min(near_best, key=lambda c: (
        sum(record_map[n].get("config", {}).get("kind") == "keras" for n in c["weights"]),
        len(c["weights"]), -c["auc"]))
    write(OUT / "selection.json", {"seed": 42, "validation_size": 0.2, "score_used_for_selection": True,
        "goal": 0.963, "goal_reached_on_selection_split": chosen["auc"] >= 0.963,
        "best_observed": choices[0], "simplicity_tolerance": 0.00005,
        "selected": chosen, "records": records, "blend_scores": choices})
    print("SELECTED", chosen, "BEST OBSERVED", choices[0], flush=True)


def previous_predict(data, y, tr, va):
    meta = json.loads((OLD / "selection.json").read_text(encoding="utf-8"))
    records = {r["name"]: r for r in meta["all_candidates"]}
    predictions = {}
    for name in meta["selected"]["weights"]:
        record, config = records[name], records[name]["config"]
        X = base.make_features(data, config["features"])
        a, b, _ = base.add_target_encoding(X.iloc[tr], X.iloc[va], y.iloc[tr])
        if config.get("route_profiles"):
            a, b, _ = base.add_route_profiles(a, b, config["route_profiles"] == "context")
        a, b, cats, _ = base.prepare(a, b, config["kind"])
        model = base.build(config, cats, record["best_iterations"])
        model.fit(a, y.iloc[tr])
        predictions[name] = model.predict_proba(b)[:, 1]
    return base.combine(predictions, meta["selected"]["weights"], meta["selected"]["method"])


def confirm_or_refit(confirm):
    meta = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    records = {r["name"]: r for r in meta["records"]}
    data, y = base.load_data()
    original = priors("train")
    if confirm:
        tr, va = train_test_split(np.arange(len(data)), test_size=0.2, stratify=y, random_state=2026)
        other, original_other = data.iloc[va], original.iloc[va]
    else:
        tr = np.arange(len(data))
        other, original_other = pd.read_csv(ROOT / "test.csv"), priors("test")
    predictions, scores = {}, {}
    for name in meta["selected"]["weights"]:
        if name == "previous_960":
            if confirm:
                predictions[name] = previous_predict(data, y, tr, va)
            else:
                old_selection = records[name]["previous_selection"]
                old = {n: np.load(OLD / f"{n}_test.npy") for n in old_selection["weights"]}
                predictions[name] = base.combine(old, old_selection["weights"], old_selection["method"])
        else:
            record = records[name]
            config = record["config"]
            a, b, cats, bundle = inputs(data.iloc[tr], other, y.iloc[tr], original.iloc[tr], original_other, config)
            print("CONFIRM" if confirm else "REFIT", name, "iterations", record["iterations"], flush=True)
            if config["kind"] == "keras":
                predictions[name], _ = run_keras(a, b, y.iloc[tr], f"{name}_{'confirm' if confirm else 'full'}", epochs=record["iterations"])
            else:
                model, _ = fit(config, a, y.iloc[tr], cats, iterations=record["iterations"])
                predictions[name] = model.predict_proba(b)[:, 1]
            if not confirm:
                joblib.dump(bundle, OUT / f"{name}_full_preprocessing.joblib")
                if config["kind"] != "keras":
                    save_model(model, config["kind"], f"{name}_full_model")
                # 저장 모델을 다시 불러와 출력이 일치하는지 확인합니다.
                if config["kind"] == "keras":
                    check = predictions[name][:1000]  # helper에서도 H5 재로딩을 검증합니다.
                elif config["kind"] == "lgb":
                    reload = lgb.Booster(model_file=str(OUT / f"{name}_full_model.txt"))
                    check = reload.predict(b.iloc[:1000])
                elif config["kind"] == "xgb":
                    reload = XGBClassifier(enable_categorical=True)
                    reload.load_model(OUT / f"{name}_full_model.json")
                    reload.set_params(device="cpu")
                    check = reload.predict_proba(b.iloc[:1000])[:, 1]
                else:
                    reload = CatBoostClassifier()
                    reload.load_model(str(OUT / f"{name}_full_model.cbm"))
                    check = reload.predict_proba(b.iloc[:1000])[:, 1]
                error = float(np.abs(check - predictions[name][:1000]).max())
                if error >= 1e-6:
                    raise ValueError(f"Model export mismatch: {name} {error}")
                write(OUT / f"{name}_reload_check.json", {"rows": len(check), "max_error": error})
        if confirm:
            scores[name] = float(roc_auc_score(y.iloc[va], predictions[name]))
        np.save(OUT / f"{name}_{'confirm' if confirm else 'test'}.npy", predictions[name])
    prediction = base.combine(predictions, meta["selected"]["weights"], meta["selected"]["method"])
    if confirm:
        score = float(roc_auc_score(y.iloc[va], prediction))
        write(OUT / "confirmation.json", {"seed": 2026, "validation_size": 0.2,
            "fixed_selection_before_evaluation": True, "model_auc": scores, "combined_auc": score,
            "goal_reached_on_confirmation_split": score >= 0.963,
            "note": "Additional split of the same dataset, not an independent test score"})
        print("CONFIRMATION", score, scores, flush=True)
    else:
        if not np.isfinite(prediction).all() or not ((prediction >= 0) & (prediction <= 1)).all():
            raise ValueError("Invalid predictions")
        submission = pd.read_csv(ROOT / "sample_submission.csv")
        assert submission.id.equals(other.id)
        submission["satisfaction"] = prediction
        submission.to_csv(OUT / "submission_candidate.csv", index=False)
        print("SAVED", OUT / "submission_candidate.csv", flush=True)


def verify():
    data = pd.read_csv(OUT / "submission_candidate.csv")
    test = pd.read_csv(ROOT / "test.csv", usecols=["id"])
    assert data.columns.tolist() == ["id", "satisfaction"]
    assert data.id.equals(test.id) and data.id.is_unique
    assert data.satisfaction.between(0, 1).all() and np.isfinite(data.satisfaction).all()
    meta = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selected"]
    predictions = {n: np.load(OUT / f"{n}_test.npy") for n in meta["weights"]}
    error = float(np.abs(base.combine(predictions, meta["weights"], meta["method"]) - data.satisfaction).max())
    assert error < 1e-10
    result = {"rows": len(data), "ids_match": True, "missing_predictions": 0, "export_max_error": error,
        "min_probability": float(data.satisfaction.min()), "max_probability": float(data.satisfaction.max())}
    write(OUT / "verification.json", result)
    print("VERIFIED", result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "tune", "select", "confirm", "refit", "verify"], default="tune")
    parser.add_argument("--candidates", choices=list(CONFIGS), nargs="+", default=["lgb_original_deep"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "train.csv", ROOT / "test.csv", ORIGINAL / "original_airline.zip"]}
    manifest = OUT / "data_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != hashes:
        raise ValueError("Source data changed; do not reuse cached models")
    write(manifest, hashes)
    if args.phase == "prepare":
        original_priors()
    elif args.phase == "tune":
        tune(args.candidates)
    elif args.phase == "select":
        _, y = base.load_data()
        _, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
        select(y.iloc[va])
    elif args.phase == "confirm":
        confirm_or_refit(True)
    elif args.phase == "refit":
        confirm_or_refit(False)
    else:
        verify()


if __name__ == "__main__":
    main()
