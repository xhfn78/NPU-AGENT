"""Kaggle 리더보드 0.963 목표: 동일 5-fold OOF로 세 트리 모델을 비교합니다.

폴드 학습 부분에서만 내부 5-fold 타깃 인코딩을 계산합니다.
각 모델의 test 확률은 5개 폴드 모델의 평균입니다. 대회에는 자동 제출하지 않습니다.
OOF는 early stopping/조합 선택에 사용하므로 독립 테스트 점수가 아닙니다.
"""

import argparse
import copy
import gc
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import improve_963 as previous


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "leaderboard_cv_20261028"
SEED = 20261028
N_FOLDS = 5
CONFIGS = {
    "lgb_routes_63": copy.deepcopy(previous.CONFIGS["lgb_original_deep"]),
    "xgb_routes_d6": copy.deepcopy(previous.CONFIGS["xgb_original_stats"]),
    "cat_routes_d8": {**copy.deepcopy(previous.CONFIGS["cat_original_stats"]), "route_profiles": "context"},
}
CONFIGS["lgb_routes_63"]["settings"]["n_estimators"] = 2200
CONFIGS["xgb_routes_d6"]["settings"]["n_estimators"] = 2200
CONFIGS["cat_routes_d8"]["settings"]["iterations"] = 1800


def write(path, value):
    previous.write(path, value)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_manifest():
    source = json.loads((previous.OUT / "data_manifest.json").read_text(encoding="utf-8"))
    actual = {name: digest(ROOT / name) for name in source}
    if actual != source:
        raise ValueError("Source data changed; regenerate source features before a new CV run")
    artifacts = [previous.OUT / "original_features_train.npy", previous.OUT / "original_features_test.npy",
        previous.OUT / "original_features.json", ROOT / "improve_auc.py", ROOT / "improve_963.py", Path(__file__)]
    manifest = {"source": actual, "dependencies": {str(p.relative_to(ROOT)): digest(p) for p in artifacts},
        "configs": CONFIGS, "folds": N_FOLDS, "seed": SEED, "protocol": "nested-target-encoding-v1"}
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Data, code, or configuration changed; use a new output directory")
    write(path, manifest)


def train():
    data, y = previous.base.load_data()
    test = pd.read_csv(ROOT / "test.csv")
    original, original_test = previous.priors("train"), previous.priors("test")
    fold_ids = np.full(len(data), -1, dtype="int8")
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(splitter.split(data, y)):
        fold_ids[va] = fold
        folder = OUT / f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        np.save(folder / "train_indices.npy", tr)
        np.save(folder / "validation_indices.npy", va)
        # 검증과 test를 함께 변환하되 fit/통계 계산에는 학습 부분만 전달합니다.
        other = pd.concat([data.iloc[va], test], ignore_index=True).drop(columns="satisfaction", errors="ignore")
        other_priors = pd.concat([original.iloc[va], original_test], ignore_index=True)
        a = b = cats = bundle = None
        for name, config in CONFIGS.items():
            record_path = folder / f"{name}_metrics.json"
            val_path, test_path = folder / f"{name}_validation.npy", folder / f"{name}_test.npy"
            if record_path.exists() and val_path.exists() and test_path.exists():
                p, q = np.load(val_path), np.load(test_path)
                assert len(p) == len(va) and len(q) == len(test)
                assert np.isfinite(p).all() and np.isfinite(q).all()
                print("CACHED", fold + 1, name, flush=True)
                continue
            if a is None:
                a, b, cats, bundle = previous.inputs(data.iloc[tr], other, y.iloc[tr],
                    original.iloc[tr], other_priors, CONFIGS["lgb_routes_63"])
                joblib.dump(bundle, folder / "preprocessing.joblib")
                print("CV FEATURES", fold + 1, a.shape, b.shape, flush=True)
            if config["kind"] == "cat":
                model_a, model_b, model_cats, schema = previous.base.prepare(a, b, "cat")
                joblib.dump({"schema": schema, "columns": model_a.columns.tolist()}, folder / f"{name}_schema.joblib")
            else:
                model_a, model_b, model_cats = a, b, cats
            print("CV TRAIN", fold + 1, "/", N_FOLDS, name, flush=True)
            model, best = previous.fit(config, model_a, y.iloc[tr], model_cats,
                model_b.iloc[:len(va)], y.iloc[va])
            if config["kind"] == "xgb":
                model.set_params(device="cpu")
            probability = model.predict_proba(model_b)[:, 1]
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            suffix = {"lgb": "txt", "xgb": "json", "cat": "cbm"}[config["kind"]]
            model_path = folder / f"{name}.{suffix}"
            if config["kind"] == "lgb":
                model.booster_.save_model(str(model_path))
                restored = previous.lgb.Booster(model_file=str(model_path))
                check = restored.predict(model_b.iloc[:1000])
            elif config["kind"] == "xgb":
                model.save_model(model_path)
                restored = previous.XGBClassifier(enable_categorical=True)
                restored.load_model(model_path)
                restored.set_params(device="cpu")
                check = restored.predict_proba(model_b.iloc[:1000])[:, 1]
            else:
                model.save_model(str(model_path))
                restored = previous.CatBoostClassifier()
                restored.load_model(str(model_path))
                check = restored.predict_proba(model_b.iloc[:1000])[:, 1]
            reload_error = float(np.abs(check - probability[:1000]).max())
            assert reload_error < 1e-6
            np.save(val_path, probability[:len(va)])
            np.save(test_path, probability[len(va):])
            score = float(roc_auc_score(y.iloc[va], probability[:len(va)]))
            record = {"fold": fold, "name": name, "train_rows": len(tr), "validation_rows": len(va),
                "auc": score, "iterations": best, "reload_max_error": reload_error,
                "config": config, "early_stopping_used_validation": True}
            write(record_path, record)
            print("CV RESULT", fold + 1, name, score, "trees", best, flush=True)
            del model, restored, model_a, model_b
            gc.collect()
        del a, b, bundle, other, other_priors
        gc.collect()
    assert (fold_ids >= 0).all()
    np.save(OUT / "fold_ids.npy", fold_ids)


def select_and_export():
    data, y = previous.base.load_data()
    test = pd.read_csv(ROOT / "test.csv", usecols=["id"])
    fold_ids = np.load(OUT / "fold_ids.npy")
    predictions, test_predictions, model_scores = {}, {}, {}
    for name in CONFIGS:
        oof = np.full(len(y), np.nan)
        test_values, records = [], []
        for fold in range(N_FOLDS):
            folder = OUT / f"fold_{fold}"
            va = np.load(folder / "validation_indices.npy")
            assert np.array_equal(va, np.flatnonzero(fold_ids == fold))
            assert np.isnan(oof[va]).all()
            oof[va] = np.load(folder / f"{name}_validation.npy")
            test_values.append(np.load(folder / f"{name}_test.npy"))
            records.append(json.loads((folder / f"{name}_metrics.json").read_text(encoding="utf-8")))
        assert np.isfinite(oof).all()
        predictions[name] = oof
        test_predictions[name] = np.mean(test_values, axis=0)
        np.save(OUT / f"{name}_oof.npy", oof)
        np.save(OUT / f"{name}_test_mean.npy", test_predictions[name])
        model_scores[name] = {"oof_auc": float(roc_auc_score(y, oof)),
            "fold_auc": [r["auc"] for r in records], "iterations": [r["iterations"] for r in records]}
    names = list(CONFIGS)
    candidates = [{"weights": {name: 1.0}, "method": "probability"} for name in names]
    for pair in itertools.combinations(names, 2):
        for weight in [0.25, 0.5, 0.75]:
            for method in ["probability", "logit"]:
                candidates.append({"weights": dict(zip(pair, [weight, 1-weight])), "method": method})
    for method in ["probability", "logit"]:
        candidates.append({"weights": dict.fromkeys(names, 1/3), "method": method})
        for main in names:
            for weight in [0.5, 0.75]:
                candidates.append({"weights": {n: weight if n == main else (1-weight)/2 for n in names}, "method": method})
    for candidate in candidates:
        p = previous.base.combine(predictions, candidate["weights"], candidate["method"])
        candidate["oof_auc"] = float(roc_auc_score(y, p))
        candidate["fold_auc"] = [float(roc_auc_score(y[fold_ids == f], p[fold_ids == f])) for f in range(N_FOLDS)]
    candidates.sort(key=lambda r: r["oof_auc"], reverse=True)
    selected = candidates[0]
    selection = {"target": {"metric": "Kaggle leaderboard ROC-AUC", "score": 0.963,
        "leaderboard_score_for_this_file": None, "goal_reached": None},
        "selected": selected, "models": model_scores, "candidates": candidates,
        "folds": N_FOLDS, "seed": SEED, "oof_rows": len(y),
        "note": "OOF used for early stopping and blend selection. OOF AUC does not establish leaderboard goal attainment."}
    write(OUT / "selection.json", selection)
    combined_oof = previous.base.combine(predictions, selected["weights"], selected["method"])
    np.save(OUT / "selected_oof.npy", combined_oof)
    combined_test = previous.base.combine(test_predictions, selected["weights"], selected["method"])
    assert len(combined_test) == len(test) and np.isfinite(combined_test).all()
    assert ((combined_test >= 0) & (combined_test <= 1)).all()
    submission = pd.read_csv(ROOT / "sample_submission.csv")
    assert submission.id.equals(test.id) and submission.id.is_unique
    submission["satisfaction"] = combined_test
    path = OUT / "submission_cv.csv"
    submission.to_csv(path, index=False)
    saved = pd.read_csv(path)
    assert saved.columns.tolist() == ["id", "satisfaction"] and saved.id.equals(test.id)
    error = float(np.abs(saved.satisfaction - combined_test).max())
    assert error < 1e-10
    write(OUT / "verification.json", {"rows": len(saved), "missing": int(saved.satisfaction.isna().sum()),
        "ids_match": True, "min_probability": float(saved.satisfaction.min()),
        "max_probability": float(saved.satisfaction.max()), "export_max_error": error,
        "submission_sha256": digest(path), "fold_models_per_component": N_FOLDS})
    print("CV SELECTED", selected, flush=True)
    print("CV SUBMISSION", path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "export", "all"], default="all", nargs="?")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    check_manifest()
    if args.phase in ["train", "all"]:
        train()
    if args.phase in ["export", "all"]:
        select_and_export()


if __name__ == "__main__":
    main()
