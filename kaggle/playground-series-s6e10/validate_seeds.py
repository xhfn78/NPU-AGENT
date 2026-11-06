"""고정 80:20 분할에서 모델 학습 시드 1~41을 비교합니다.

대조군 42를 같은 조건으로 다시 학습합니다. 외부 전처리, 데이터 분할, 트리 수,
조합 가중치는 고정합니다. CPU LightGBM과 GPU XGBoost/CatBoost를 병행합니다.
검증 점수로 고른 최고 시드는 탐색 결과이며 Kaggle 점수와 구분합니다.
"""

import argparse
import copy
import gc
import hashlib
import json
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import leaderboard_cv as source


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "model_seeds_1_to_41"
PREVIOUS = source.previous
SEEDS = list(range(1, 42))
REFERENCE = 42
SPLIT_SEED = 42
CPU_THREADS = 6
GPU_THREADS = 4
NAMES = ["lgb", "xgb", "cat"]
SOURCE_NAMES = dict(zip(NAMES, source.CONFIGS))
WEIGHTS = dict.fromkeys(NAMES, 1/3)
PRINT_LOCK = threading.Lock()


def log(*values):
    with PRINT_LOCK:
        print(*values, flush=True)


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(path):
    return source.digest(path)


def protocol():
    old = json.loads((PREVIOUS.OUT / "data_manifest.json").read_text(encoding="utf-8"))
    actual = {p: digest(ROOT / p) for p in old}
    if actual != old:
        raise ValueError("Source data changed")
    cv = json.loads((source.OUT / "selection.json").read_text(encoding="utf-8"))
    trees = {name: int(np.median(cv["models"][full]["iterations"])) for name, full in SOURCE_NAMES.items()}
    dependencies = [ROOT / "improve_auc.py", ROOT / "improve_963.py", ROOT / "leaderboard_cv.py",
        PREVIOUS.OUT / "original_features_train.npy", PREVIOUS.OUT / "original_features.json", source.OUT / "selection.json"]
    value = {"protocol": "fixed-holdout-model-seeds-v1", "source": actual,
        "dependencies": {str(p.relative_to(ROOT)): digest(p) for p in dependencies},
        "split_seed": SPLIT_SEED, "validation_fraction": 0.2, "seeds": SEEDS,
        "reference_seed": REFERENCE, "trees": trees, "configs": source.CONFIGS,
        "weights": WEIGHTS, "combination": "logit", "early_stopping": False,
        "target_encoding_seed": 42, "fixed_lgb_binning_seed": 42,
        "cpu_threads": CPU_THREADS, "gpu_threads": GPU_THREADS,
        "iteration_source": "median of previous 5-fold seed42 best tree counts"}
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("Protocol changed; use a new output folder")
    write(path, value)
    return value


def prepare():
    path = OUT / "features.joblib"
    if path.exists():
        result = joblib.load(path)
        assert len(result["y_train"]) == 559708 and len(result["y_valid"]) == 139927
        return result
    data, y = PREVIOUS.base.load_data()
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    assert np.array_equal(va, np.load(PREVIOUS.OLD / "validation_indices.npy"))
    original = PREVIOUS.priors("train")
    config = source.CONFIGS["lgb_routes_63"]
    a, b, cats, bundle = PREVIOUS.inputs(data.iloc[tr], data.iloc[va], y.iloc[tr], original.iloc[tr], original.iloc[va], config)
    result = {"a": a, "b": b, "cats": cats, "y_train": y.iloc[tr].to_numpy(),
        "y_valid": y.iloc[va].to_numpy(), "train_indices": tr, "valid_indices": va,
        "validation_ids": data.id.iloc[va].to_numpy()}
    joblib.dump(bundle, OUT / "preprocessing.joblib")
    joblib.dump(result, path)
    np.save(OUT / "train_indices.npy", tr)
    np.save(OUT / "validation_indices.npy", va)
    log("FIXED FEATURES", a.shape, b.shape, "split seed", SPLIT_SEED)
    return result


def paths(name, seed):
    prefix = OUT / f"{name}_seed_{seed:02d}"
    return (Path(str(prefix) + "_metrics.json"), Path(str(prefix) + "_validation.npy"),
        Path(str(prefix) + {"lgb": ".txt", "xgb": ".json", "cat": ".cbm"}[name]))


def cached(name, seed, trees):
    record, prediction, model = paths(name, seed)
    if all(p.exists() for p in [record, prediction, model]):
        result = json.loads(record.read_text(encoding="utf-8"))
        assert result["model_seed"] == seed and result["iterations"] == trees
        p = np.load(prediction, mmap_mode="r")
        assert len(p) == 139927 and np.isfinite(p).all()
        return True
    return False


def save_result(name, seed, trees, p, target, start, reload_error, actual_seed, params):
    assert actual_seed == seed
    assert len(p) == len(target) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all() and reload_error < 1e-6
    record, prediction, _ = paths(name, seed)
    np.save(prediction, p)
    value = {"model": name, "model_seed": seed, "verified_training_seed": actual_seed,
        "iterations": trees, "auc": float(roc_auc_score(target, p)),
        "seconds": time.perf_counter()-start, "reload_max_error": reload_error,
        "validation_rows": len(target), "split_seed": SPLIT_SEED, "early_stopping": False,
        "params": params, "runner_sha256": digest(Path(__file__))}
    write(record, value)
    completed = len(list(OUT.glob("*_metrics.json")))
    log("SEED RESULT", name, seed, f"AUC={value['auc']:.9f}",
        f"seconds={value['seconds']:.1f}", f"completed={completed}/126")


def cpu_worker(features, meta, seeds):
    cfg = source.CONFIGS["lgb_routes_63"]["settings"]
    fixed = {"objective": "binary", "metric": "auc", "learning_rate": cfg["learning_rate"],
        "num_leaves": cfg["num_leaves"], "min_data_in_leaf": cfg["min_child_samples"],
        "lambda_l2": cfg["reg_lambda"], "bagging_fraction": cfg["subsample"],
        "bagging_freq": cfg["subsample_freq"], "feature_fraction": cfg["colsample_bytree"],
        "max_bin": cfg["max_bin"], "data_random_seed": 42, "deterministic": True,
        "force_col_wise": True, "num_threads": CPU_THREADS, "verbosity": -1}
    dataset = PREVIOUS.lgb.Dataset(features["a"], label=features["y_train"], free_raw_data=False,
        params={k: fixed[k] for k in ["max_bin", "data_random_seed", "min_data_in_leaf", "num_threads", "verbosity"]})
    dataset.construct()
    trees = meta["trees"]["lgb"]
    for seed in seeds:
        if cached("lgb", seed, trees):
            continue
        start = time.perf_counter()
        params = {**fixed, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
        model = PREVIOUS.lgb.train(params, dataset, num_boost_round=trees)
        p = model.predict(features["b"], num_threads=CPU_THREADS)
        model.save_model(str(paths("lgb", seed)[2]))
        reloaded = PREVIOUS.lgb.Booster(model_file=str(paths("lgb", seed)[2]))
        error = float(np.abs(reloaded.predict(features["b"].iloc[:1000], num_threads=CPU_THREADS) - p[:1000]).max())
        save_result("lgb", seed, trees, p, features["y_valid"], start, error, int(model.params["seed"]), params)
        del model, reloaded
    del dataset
    gc.collect()


def gpu_worker(features, meta, seeds):
    cfg = source.CONFIGS["xgb_routes_d6"]["settings"]
    dtrain = xgb.QuantileDMatrix(features["a"], label=features["y_train"], enable_categorical=True,
        max_bin=cfg["max_bin"], nthread=GPU_THREADS)
    dvalid = xgb.DMatrix(features["b"], enable_categorical=True, nthread=GPU_THREADS)
    params = {k: cfg[k] for k in ["learning_rate", "max_depth", "min_child_weight", "subsample",
        "colsample_bytree", "reg_lambda", "max_bin"]}
    params.update(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda", nthread=GPU_THREADS)
    trees = meta["trees"]["xgb"]
    for seed in seeds:
        if cached("xgb", seed, trees):
            continue
        start = time.perf_counter()
        settings = {**params, "seed": seed}
        model = xgb.train(settings, dtrain, num_boost_round=trees, verbose_eval=False)
        actual_seed = int(json.loads(model.save_config())["learner"]["generic_param"]["seed"])
        model.set_param({"device": "cpu"})
        p = model.predict(dvalid)
        model.save_model(paths("xgb", seed)[2])
        reloaded = xgb.Booster()
        reloaded.load_model(paths("xgb", seed)[2])
        reloaded.set_param({"device": "cpu", "nthread": GPU_THREADS})
        error = float(np.abs(reloaded.predict(dvalid.slice(np.arange(1000))) - p[:1000]).max())
        save_result("xgb", seed, trees, p, features["y_valid"], start, error, actual_seed, settings)
        del model, reloaded
    del dtrain, dvalid
    gc.collect()
    a, b, cats, _ = PREVIOUS.base.prepare(features["a"], features["b"], "cat")
    train_pool = Pool(a, label=features["y_train"], cat_features=cats)
    valid_pool = Pool(b, cat_features=cats)
    check_pool = Pool(b.iloc[:1000], cat_features=cats)
    settings = copy.deepcopy(source.CONFIGS["cat_routes_d8"]["settings"])
    trees = meta["trees"]["cat"]
    settings.update(iterations=trees, task_type="GPU", devices="0", thread_count=GPU_THREADS,
        allow_writing_files=False, verbose=False)
    for seed in seeds:
        if cached("cat", seed, trees):
            continue
        start = time.perf_counter()
        params = {**settings, "random_seed": seed, "loss_function": "Logloss"}
        model = CatBoostClassifier(**params)
        model.fit(train_pool)
        p = model.predict_proba(valid_pool, thread_count=GPU_THREADS)[:, 1]
        model.save_model(str(paths("cat", seed)[2]))
        reloaded = CatBoostClassifier()
        reloaded.load_model(str(paths("cat", seed)[2]))
        error = float(np.abs(reloaded.predict_proba(check_pool, thread_count=GPU_THREADS)[:, 1] - p[:1000]).max())
        save_result("cat", seed, trees, p, features["y_valid"], start, error,
            int(model.get_all_params()["random_seed"]), params)
        del model, reloaded
    del train_pool, valid_pool, check_pool, a, b
    gc.collect()


def report(features, meta):
    target = features["y_valid"]
    records, predictions, reference, statistics = [], {}, {}, {}
    for name in NAMES:
        scores, values = [], []
        for seed in SEEDS + [REFERENCE]:
            record, prediction, model = paths(name, seed)
            if not all(p.exists() for p in [record, prediction, model]):
                raise ValueError(f"Missing result: {name}, seed={seed}")
            value = json.loads(record.read_text(encoding="utf-8"))
            assert value["model_seed"] == value["verified_training_seed"] == seed
            p = np.load(prediction)
            assert len(p) == len(target) and np.isfinite(p).all()
            assert abs(float(roc_auc_score(target, p))-value["auc"]) < 1e-12
            records.append(value)
            if seed == REFERENCE:
                reference[name] = p
            else:
                scores.append(value["auc"])
                values.append(p)
                predictions.setdefault(seed, {})[name] = p
        average = np.mean(values, axis=0)
        np.save(OUT / f"{name}_all_41_mean_validation.npy", average)
        statistics[name] = {"min_auc": float(np.min(scores)), "mean_auc": float(np.mean(scores)),
            "std_auc": float(np.std(scores, ddof=1)), "max_auc": float(np.max(scores)),
            "best_seed": SEEDS[int(np.argmax(scores))], "reference42_auc": float(roc_auc_score(target, reference[name])),
            "all_41_prediction_mean_auc": float(roc_auc_score(target, average))}
    baseline = PREVIOUS.base.combine(reference, WEIGHTS, "logit")
    baseline_auc = float(roc_auc_score(target, baseline))
    rows = []
    ensemble_predictions = []
    for seed in SEEDS:
        p = PREVIOUS.base.combine(predictions[seed], WEIGHTS, "logit")
        ensemble_predictions.append(p)
        row = {"seed": seed, **{f"{name}_auc": float(roc_auc_score(target, predictions[seed][name])) for name in NAMES},
            "ensemble_auc": float(roc_auc_score(target, p))}
        row["ensemble_delta_vs_seed42"] = row["ensemble_auc"]-baseline_auc
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "seed_scores.csv", index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k != "params"} for r in records]).to_csv(OUT / "model_scores.csv", index=False)
    averages = {name: np.load(OUT / f"{name}_all_41_mean_validation.npy") for name in NAMES}
    all_seeds = PREVIOUS.base.combine(averages, WEIGHTS, "logit")
    np.save(OUT / "all_41_seed_ensemble_validation.npy", all_seeds)
    np.save(OUT / "reference42_ensemble_validation.npy", baseline)
    result = {"split_seed": SPLIT_SEED, "train_rows": len(features["y_train"]), "validation_rows": len(target),
        "seeds_tested": SEEDS, "seeds_per_model": len(SEEDS), "models_per_seed": NAMES,
        "reference_seed": REFERENCE, "total_fits_including_reference": len(records), "trees": meta["trees"],
        "model_statistics": statistics, "reference42_ensemble_auc": baseline_auc,
        "seed_ensemble_auc": float(roc_auc_score(target, all_seeds)),
        "seed_ensemble_delta_vs_reference42": float(roc_auc_score(target, all_seeds))-baseline_auc,
        "ensemble_score_mean": float(table.ensemble_auc.mean()), "ensemble_score_std": float(table.ensemble_auc.std()),
        "ensemble_score_min": float(table.ensemble_auc.min()), "ensemble_score_max": float(table.ensemble_auc.max()),
        "best_seeds": table.sort_values("ensemble_auc", ascending=False).head(10).to_dict(orient="records"),
        "max_model_reload_error": max(r["reload_max_error"] for r in records),
        "leaderboard_goal": 0.963, "leaderboard_score": None, "goal_reached": None,
        "note": "Fixed holdout used for seed comparison and best-seed selection. Local validation only; not 41 repeated 5-fold runs. Best seed can be a selection effect, not a leaderboard gain."}
    write(OUT / "summary.json", result)
    log("ALL SEEDS COMPLETE", json.dumps({k:result[k] for k in ["total_fits_including_reference", "reference42_ensemble_auc",
        "seed_ensemble_auc", "seed_ensemble_delta_vs_reference42", "ensemble_score_min", "ensemble_score_max"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "pilot", "run", "report"], default="run", nargs="?")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = protocol()
    features = prepare()
    if args.phase in ["pilot", "run"]:
        seeds = [REFERENCE] if args.phase == "pilot" else [REFERENCE] + SEEDS
        with ThreadPoolExecutor(max_workers=2) as pool:
            cpu = pool.submit(cpu_worker, features, meta, seeds)
            gpu = pool.submit(gpu_worker, features, meta, seeds)
            cpu.result()
            gpu.result()
    if args.phase in ["report", "run"]:
        report(features, meta)


if __name__ == "__main__":
    main()
