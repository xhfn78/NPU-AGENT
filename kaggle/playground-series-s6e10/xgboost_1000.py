"""고정 분할/시드에서 XGBoost 트리 수를 1,000개까지 비교한 후 학습 시드를 검증합니다."""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import validate_seeds as fixed


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "xgboost_1000_trees"
N_TREES = 1000
COUNTS = [100, 200, 300, 400, 500, 600, 700, 786, 800, 900, 1000]
SEEDS = list(range(1, 42))
REFERENCE = 42


def write(path, value):
    fixed.write(path, value)


def prepare():
    fixed.protocol()
    features = fixed.prepare()
    reference = json.loads((fixed.OUT / "xgb_seed_42_metrics.json").read_text(encoding="utf-8"))
    assert reference["iterations"] == 786 and reference["verified_training_seed"] == 42
    params = copy.deepcopy(reference["params"])
    manifest = {"source_manifest_sha256": fixed.digest(fixed.OUT / "manifest.json"),
        "features_sha256": fixed.digest(fixed.OUT / "features.joblib"),
        "reference_record_sha256": fixed.digest(fixed.OUT / "xgb_seed_42_metrics.json"),
        "params": params, "split_seed": 42, "reference_seed": 42, "trees": N_TREES,
        "seeds": SEEDS, "early_stopping": False, "protocol": "xgb-fixed-1000-v1"}
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Experiment inputs changed; use a new output directory")
    write(path, manifest)
    dtrain = xgb.QuantileDMatrix(features["a"], label=features["y_train"],
        max_bin=params["max_bin"], enable_categorical=True, nthread=params["nthread"])
    dvalid = xgb.DMatrix(features["b"], enable_categorical=True, nthread=params["nthread"])
    return features, params, reference, dtrain, dvalid


def train_seed(seed, features, params, dtrain, dvalid):
    prefix = OUT / f"seed_{seed:02d}"
    record_path = Path(str(prefix) + "_metrics.json")
    probability_path = Path(str(prefix) + "_validation.npy")
    model_path = Path(str(prefix) + ".json")
    if record_path.exists() and probability_path.exists() and model_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["verified_training_seed"] == seed and record["trees"] == N_TREES
        model = xgb.Booster()
        model.load_model(model_path)
        model.set_param({"device": "cpu", "nthread": params["nthread"]})
        assert model.num_boosted_rounds() == N_TREES
        return model, np.load(probability_path), record
    start = time.perf_counter()
    settings = {**params, "seed": seed}
    model = xgb.train(settings, dtrain, num_boost_round=N_TREES, verbose_eval=False)
    actual_seed = int(json.loads(model.save_config())["learner"]["generic_param"]["seed"])
    assert actual_seed == seed and model.num_boosted_rounds() == N_TREES
    model.set_param({"device": "cpu"})
    p = model.predict(dvalid)
    assert len(p) == len(features["y_valid"]) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    model.save_model(model_path)
    restored = xgb.Booster()
    restored.load_model(model_path)
    restored.set_param({"device": "cpu", "nthread": params["nthread"]})
    error = float(np.abs(restored.predict(dvalid.slice(np.arange(1000))) - p[:1000]).max())
    assert error < 1e-6
    np.save(probability_path, p)
    record = {"model_seed": seed, "verified_training_seed": actual_seed, "trees": N_TREES,
        "auc": float(roc_auc_score(features["y_valid"], p)), "seconds": time.perf_counter()-start,
        "train_rows": len(features["y_train"]), "validation_rows": len(p), "split_seed": 42,
        "early_stopping": False, "reload_max_error": error, "params": settings}
    write(record_path, record)
    print("XGB 1000 RESULT", seed, f"AUC={record['auc']:.9f}", f"seconds={record['seconds']:.1f}", flush=True)
    del restored
    return model, p, record


def validate_trees(features, params, reference, dtrain, dvalid):
    model, _, record = train_seed(REFERENCE, features, params, dtrain, dvalid)
    rows = []
    for count in COUNTS:
        p = model.predict(dvalid, iteration_range=(0, count))
        rows.append({"trees": count, "auc": float(roc_auc_score(features["y_valid"], p))})
        if count in [786, 800, 900, 1000]:
            np.save(OUT / f"trees_{count}_seed42_validation.npy", p)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tree_comparison.csv", index=False)
    baseline = float(table.loc[table.trees == 786, "auc"].iloc[0])
    old = np.load(fixed.OUT / "xgb_seed_42_validation.npy")
    prefix = np.load(OUT / "trees_786_seed42_validation.npy")
    summary = {"model_seed": REFERENCE, "split_seed": 42, "actual_trained_trees": model.num_boosted_rounds(),
        "train_rows": len(features["y_train"]), "validation_rows": len(features["y_valid"]),
        "fixed_params": params, "scores": rows, "baseline_786_auc": baseline,
        "fixed_1000_auc": record["auc"], "delta_1000_vs_786": record["auc"]-baseline,
        "best_checked_count": int(table.loc[table.auc.idxmax(), "trees"]),
        "best_checked_auc": float(table.auc.max()), "old_786_reference_auc": reference["auc"],
        "old_reference_prediction_max_difference": float(np.abs(old-prefix).max()),
        "reload_max_error": record["reload_max_error"], "leaderboard_score": None,
        "note": "Same 1000-tree model prefixes compared on fixed local holdout. Validation used for comparison; no Kaggle score established."}
    write(OUT / "tree_comparison.json", summary)
    print("TREE COMPARISON", json.dumps(summary, ensure_ascii=False), flush=True)
    del model
    gc.collect()


def validate_seeds(features, params, dtrain, dvalid):
    _, baseline, control = train_seed(REFERENCE, features, params, dtrain, dvalid)
    values, records = [], []
    prefix_values = {count: [] for count in [786, 800, 900, 1000]}
    prefix_rows = []
    for seed in SEEDS:
        model, p, record = train_seed(seed, features, params, dtrain, dvalid)
        values.append(p)
        records.append(record)
        prefix_row = {"model_seed": seed}
        for count in prefix_values:
            prefix_p = p if count == N_TREES else model.predict(dvalid, iteration_range=(0, count))
            prefix_values[count].append(prefix_p)
            prefix_row[f"auc_trees_{count}"] = float(roc_auc_score(features["y_valid"], prefix_p))
            if count != N_TREES:
                np.save(OUT / f"seed_{seed:02d}_trees_{count}_validation.npy", prefix_p)
        prefix_rows.append(prefix_row)
        del model
        gc.collect()
    assert [r["verified_training_seed"] for r in records] == SEEDS
    table = pd.DataFrame([{k:r[k] for k in ["model_seed", "trees", "auc", "seconds", "reload_max_error"]} for r in records])
    table = table.merge(pd.DataFrame(prefix_rows), on="model_seed", validate="one_to_one")
    table["delta_vs_seed42"] = table.auc-control["auc"]
    table.to_csv(OUT / "seed_scores.csv", index=False)
    average = np.mean(values, axis=0)
    np.save(OUT / "all_41_seed_mean_validation.npy", average)
    prefix_statistics = {}
    for count, predictions in prefix_values.items():
        prefix_average = np.mean(predictions, axis=0)
        np.save(OUT / f"all_41_seed_mean_trees_{count}_validation.npy", prefix_average)
        scores = table[f"auc_trees_{count}"]
        prefix_statistics[str(count)] = {"mean_single_seed_auc": float(scores.mean()),
            "std_single_seed_auc": float(scores.std()), "seed_ensemble_auc": float(roc_auc_score(features["y_valid"], prefix_average))}
    paired_delta = table.auc_trees_1000-table.auc_trees_800
    summary = {"seeds_tested": SEEDS, "count": len(SEEDS), "trees_per_model": N_TREES,
        "split_seed": 42, "reference_seed": REFERENCE, "reference42_auc": control["auc"],
        "min_auc": float(table.auc.min()), "mean_auc": float(table.auc.mean()),
        "std_auc": float(table.auc.std()), "max_auc": float(table.auc.max()),
        "seed_ensemble_auc": float(roc_auc_score(features["y_valid"], average)),
        "ensemble_delta_vs_reference42": float(roc_auc_score(features["y_valid"], average))-control["auc"],
        "best_seeds": table.sort_values("auc", ascending=False).head(10).to_dict(orient="records"),
        "max_reload_error": max(r["reload_max_error"] for r in records),
        "tree_statistics_across_seeds": prefix_statistics,
        "paired_1000_vs_800_mean_delta": float(paired_delta.mean()),
        "seeds_where_1000_beats_800": int((paired_delta > 0).sum()),
        "validation_rows": len(features["y_valid"]), "leaderboard_score": None,
        "note": "XGBoost only. Fixed 80:20 holdout and 1000 trees. Best seed selected on this validation; not a leaderboard result or a 5-fold evaluation."}
    write(OUT / "seed_summary.json", summary)
    print("XGB SEEDS COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["trees", "seeds", "all"], default="trees", nargs="?")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    features, params, reference, dtrain, dvalid = prepare()
    if args.phase in ["trees", "all"]:
        validate_trees(features, params, reference, dtrain, dvalid)
    if args.phase in ["seeds", "all"]:
        validate_seeds(features, params, dtrain, dvalid)


if __name__ == "__main__":
    main()
