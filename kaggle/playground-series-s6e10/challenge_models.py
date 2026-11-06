"""조건별 전문 모델과 신경망의 후속 조합 실험. 기존 산출물은 수정하지 않습니다."""

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

import improve_963 as previous


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "challenge_963"
SOURCE = ROOT / "outputs" / "improve_963_gpu"
SEGMENTS = {"travel_experts": "Type of Travel", "loyalty_experts": "Customer Type", "class_experts": "Class"}
CONFIG = {
    **previous.CONFIGS["lgb_original_deep"],
    "settings": {**previous.CONFIGS["lgb_original_deep"]["settings"],
        "n_estimators": 2500, "num_leaves": 31, "reg_lambda": 20.0, "min_child_samples": 150},
}
NEIGHBOR_CONFIGS = {
    "neighbor_lgb": previous.CONFIGS["lgb_original_deep"],
    "neighbor_lgb_small": previous.CONFIGS["lgb_original_prior"],
}
SOFT_TARGETS = {"soft_prior_10": 0.1, "soft_prior_30": 0.3}
TREE_CONFIGS = {
    "lgb_leaves127": {**previous.CONFIGS["lgb_original_deep"], "settings": {
        **previous.CONFIGS["lgb_original_deep"]["settings"], "num_leaves": 127,
        "learning_rate": 0.025, "min_child_samples": 200, "reg_lambda": 25.0,
        "n_estimators": 2200}},
    "lgb_leaves255": {**previous.CONFIGS["lgb_original_deep"], "settings": {
        **previous.CONFIGS["lgb_original_deep"]["settings"], "num_leaves": 255,
        "learning_rate": 0.025, "min_child_samples": 400, "reg_lambda": 30.0,
        "n_estimators": 1800}},
    "lgb_extra_trees": {**previous.CONFIGS["lgb_original_deep"], "settings": {
        **previous.CONFIGS["lgb_original_deep"]["settings"], "num_leaves": 127,
        "learning_rate": 0.03, "min_child_samples": 250, "reg_lambda": 30.0,
        "extra_trees": True, "feature_fraction_bynode": 0.8, "n_estimators": 2500}},
}


def write(path, data):
    previous.write(path, data)


def incumbent(part):
    selected = json.loads((SOURCE / "selection.json").read_text(encoding="utf-8"))["selected"]
    return previous.base.combine({n: np.load(SOURCE / f"{n}_{part}.npy") for n in selected["weights"]},
        selected["weights"], selected["method"])


def arrays(seed):
    data, y = previous.base.load_data()
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    original = previous.priors("train")
    a, b, cats, bundle = previous.inputs(data.iloc[tr], data.iloc[va], y.iloc[tr], original.iloc[tr], original.iloc[va], CONFIG)
    return data, y, tr, va, a, b, cats, bundle


def fit_specialists(name, data_train, data_other, a, b, target, cats, fixed=None):
    column = SEGMENTS[name]
    values = sorted(data_train[column].unique().tolist())
    result = np.full(len(b), np.nan)
    iterations = {}
    for i, value in enumerate(values):
        mask_train = data_train[column].to_numpy() == value
        mask_other = data_other[column].to_numpy() == value
        print(name, value, "train", int(mask_train.sum()), "other", int(mask_other.sum()), flush=True)
        if not mask_other.any():
            continue
        if fixed is None:
            model, best = previous.fit(CONFIG, a.loc[mask_train], target[mask_train], cats,
                b.loc[mask_other], data_other.satisfaction.astype(int).to_numpy()[mask_other])
        else:
            model, best = previous.fit(CONFIG, a.loc[mask_train], target[mask_train], cats,
                iterations=fixed[str(value)])
        result[mask_other] = model.predict_proba(b.loc[mask_other])[:, 1]
        iterations[str(value)] = best
        model.booster_.save_model(str(OUT / f"{name}_{'tune' if fixed is None else 'fixed'}_{i}.txt"))
    if not np.isfinite(result).all():
        raise ValueError("Unseen segment or invalid expert predictions")
    return result, iterations


def tune_experts(names):
    data, y, tr, va, a, b, cats, bundle = arrays(42)
    joblib.dump(bundle, OUT / "expert_validation_preprocessing.joblib")
    for name in names:
        path = OUT / f"{name}_record.json"
        if path.exists():
            print("CACHED", name, flush=True)
            continue
        prediction, best = fit_specialists(name, data.iloc[tr], data.iloc[va], a, b, y.iloc[tr].to_numpy(), cats)
        score = float(roc_auc_score(y.iloc[va], prediction))
        np.save(OUT / f"{name}_validation.npy", prediction)
        write(path, {"name": name, "column": SEGMENTS[name], "iterations": best, "validation_auc": score, "config": CONFIG})
        print("EXPERT RESULT", name, score, flush=True)


def attention(train, other, target, folder, epochs=None, y_valid=None):
    folder.mkdir(parents=True, exist_ok=True)
    a, b = pd.get_dummies(train, dtype="float32"), pd.get_dummies(other, dtype="float32")
    b = b.reindex(columns=a.columns, fill_value=0)
    median = a.median().fillna(0)
    scaler = QuantileTransformer(n_quantiles=256, subsample=100000, output_distribution="normal", random_state=42)
    a, b = a.fillna(median), b.fillna(median)
    np.save(folder / "X_train.npy", scaler.fit_transform(a).astype("float32"))
    np.save(folder / "X_other.npy", scaler.transform(b).astype("float32"))
    np.save(folder / "y_train.npy", np.asarray(target))
    joblib.dump({"columns": a.columns.tolist(), "median": median, "scaler": scaler}, folder / "dense_preprocessing.joblib")
    if y_valid is not None:
        np.save(folder / "y_valid.npy", np.asarray(y_valid))
    cmd = [r"C:\Users\Admin\miniconda3\envs\tf29x-gpu\python.exe", "-u", str(ROOT / "keras_attention.py"), "--folder", str(folder)]
    if epochs is not None:
        cmd += ["--fixed", "--epochs", str(epochs)]
    subprocess.run(cmd, check=True)
    return np.load(folder / "probability.npy"), json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["epochs"]


def tune_attention():
    if (OUT / "attention_record.json").exists():
        return
    data, y, tr, va, a, b, _, bundle = arrays(42)
    probability, epochs = attention(a, b, y.iloc[tr], OUT / "attention_tune", y_valid=y.iloc[va])
    np.save(OUT / "attention_validation.npy", probability)
    joblib.dump(bundle, OUT / "attention_validation_preprocessing.joblib")
    score = float(roc_auc_score(y.iloc[va], probability))
    write(OUT / "attention_record.json", {"name": "attention", "epochs": epochs, "validation_auc": score})
    print("ATTENTION RESULT", score, "epochs", epochs, flush=True)


def add_neighbors(a, b, tr, va=None):
    training = np.load(OUT / "neighbor_train.npy")
    other = np.load(OUT / "neighbor_test.npy") if va is None else training[va]
    a, b = a.copy(), b.copy()
    for i in range(training.shape[1]):
        a[f"NEIGH_{i}"] = training[tr, i]
        b[f"NEIGH_{i}"] = other[:, i]
    return a, b


def tune_neighbors():
    data, y, tr, va, a, b, cats, bundle = arrays(42)
    a, b = add_neighbors(a, b, tr, va)
    bundle = {**bundle, "columns": a.columns.tolist(), "neighbor_features": True}
    joblib.dump(bundle, OUT / "neighbor_validation_preprocessing.joblib")
    for name, config in NEIGHBOR_CONFIGS.items():
        if (OUT / f"{name}_record.json").exists():
            continue
        print("TUNING NEIGHBORS", name, a.shape, flush=True)
        model, iterations = previous.fit(config, a, y.iloc[tr], cats, b, y.iloc[va])
        prediction = model.predict_proba(b)[:, 1]
        np.save(OUT / f"{name}_validation.npy", prediction)
        model.booster_.save_model(str(OUT / f"{name}_validation_model.txt"))
        score = float(roc_auc_score(y.iloc[va], prediction))
        write(OUT / f"{name}_record.json", {"name": name, "config": config,
            "iterations": iterations, "validation_auc": score})
        print("NEIGHBOR RESULT", name, score, flush=True)


def fit_soft(a, b, y, prior, alpha, y_valid=None, iterations=None):
    soft_y = (1-alpha) * np.asarray(y) + alpha * np.asarray(prior)
    settings = {**previous.CONFIGS["lgb_original_deep"]["settings"]}
    settings["n_estimators"] = iterations or 2500
    model = previous.lgb.LGBMRegressor(**settings, objective="cross_entropy", metric="auc",
        random_state=42, n_jobs=8, verbosity=-1, force_col_wise=True)
    if iterations is None:
        model.fit(a, soft_y, eval_set=[(b, y_valid)], callbacks=[previous.lgb.early_stopping(100), previous.lgb.log_evaluation(200)])
        best = model.best_iteration_
    else:
        model.fit(a, soft_y)
        best = iterations
    return model, int(best)


def tune_soft():
    data, y, tr, va, a, b, _, _ = arrays(42)
    original = previous.priors("train")
    teacher = (original.ORIG_xgb_probability.to_numpy() + original.ORIG_cat_probability.to_numpy()) / 2
    for name, alpha in SOFT_TARGETS.items():
        if (OUT / f"{name}_record.json").exists():
            continue
        model, best = fit_soft(a, b, y.iloc[tr], teacher[tr], alpha, y.iloc[va])
        prediction = model.predict(b)
        np.save(OUT / f"{name}_validation.npy", prediction)
        model.booster_.save_model(str(OUT / f"{name}_validation_model.txt"))
        score = float(roc_auc_score(y.iloc[va], prediction))
        write(OUT / f"{name}_record.json", {"name": name, "alpha": alpha, "iterations": best, "validation_auc": score})
        print("SOFT-TARGET RESULT", name, score, flush=True)


def tune_trees():
    data, y, tr, va, a, b, cats, bundle = arrays(42)
    joblib.dump(bundle, OUT / "tree_validation_preprocessing.joblib")
    for name, config in TREE_CONFIGS.items():
        if (OUT / f"{name}_record.json").exists():
            record = json.loads((OUT / f"{name}_record.json").read_text(encoding="utf-8"))
            if record["config"] != config:
                raise ValueError("Tree configuration changed; use a new experiment name")
            continue
        print("TUNING TREE", name, a.shape, flush=True)
        model, iterations = previous.fit(config, a, y.iloc[tr], cats, b, y.iloc[va])
        prediction = model.predict_proba(b)[:, 1]
        np.save(OUT / f"{name}_validation.npy", prediction)
        model.booster_.save_model(str(OUT / f"{name}_validation_model.txt"))
        score = float(roc_auc_score(y.iloc[va], prediction))
        write(OUT / f"{name}_record.json", {"name": name, "config": config,
            "iterations": iterations, "validation_auc": score})
        print("TREE RESULT", name, score, flush=True)


def blend(predictions, weights, method):
    if method == "rank":
        from scipy.stats import rankdata
        predictions = {n: rankdata(predictions[n], method="average") / (len(predictions[n]) + 1) for n in weights}
        method = "probability"
    return previous.base.combine(predictions, weights, method)


def select(selection_file="selection.json"):
    data, y = previous.base.load_data()
    _, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    predictions = {"incumbent": incumbent("validation")}
    records = [json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("*_record.json")]
    for record in records:
        predictions[record["name"]] = np.load(OUT / f"{record['name']}_validation.npy")
    choices = []
    names = list(predictions)
    for name in names:
        choices.append({"weights": {name: 1.0}, "method": "probability", "auc": float(roc_auc_score(y.iloc[va], predictions[name]))})
    for count in [2, 3]:
        for members in itertools.combinations(names, count):
            candidates = [dict.fromkeys(members, 1/count)] if count == 3 else [dict(zip(members, [w, 1-w])) for w in [0.2, 0.4, 0.5, 0.6, 0.8]]
            for weights in candidates:
                for method in ["probability", "logit", "rank"]:
                    score = float(roc_auc_score(y.iloc[va], blend(predictions, weights, method)))
                    choices.append({"weights": weights, "method": method, "auc": score})
    choices.sort(key=lambda c:c["auc"], reverse=True)
    result = {"selected": choices[0], "records": records, "all_blends": choices,
        "seed": 42, "goal": 0.963, "goal_reached": choices[0]["auc"] >= 0.963,
        "note": "This split was used for candidate selection."}
    write(OUT / selection_file, result)
    print("CHALLENGE SELECTED", choices[0], flush=True)


def evaluate_or_refit(confirm, selection_file="selection.json"):
    meta = json.loads((OUT / selection_file).read_text(encoding="utf-8"))
    records = {r["name"]: r for r in meta["records"]}
    data, y = previous.base.load_data()
    original = previous.priors("train")
    if confirm:
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=2026)
        other, other_priors = data.iloc[va], original.iloc[va]
    else:
        tr = np.arange(len(y))
        other, other_priors = pd.read_csv(ROOT / "test.csv"), previous.priors("test")
    need_new = any(n != "incumbent" and not n.startswith("tabicl_") for n in meta["selected"]["weights"])
    if need_new:
        a, b, cats, bundle = previous.inputs(data.iloc[tr], other, y.iloc[tr], original.iloc[tr], other_priors, CONFIG)
        if not confirm:
            joblib.dump(bundle, OUT / "full_preprocessing.joblib")
    predictions, scores = {}, {}
    for name in meta["selected"]["weights"]:
        if name == "incumbent":
            predictions[name] = incumbent("confirm" if confirm else "test")
        elif name == "attention":
            predictions[name], _ = attention(a, b, y.iloc[tr], OUT / f"attention_{'confirm' if confirm else 'full'}", epochs=records[name]["epochs"])
        elif name in NEIGHBOR_CONFIGS:
            na, nb = add_neighbors(a, b, tr, va if confirm else None)
            model, _ = previous.fit(records[name]["config"], na, y.iloc[tr], cats, iterations=records[name]["iterations"])
            predictions[name] = model.predict_proba(nb)[:, 1]
            if not confirm:
                model_path = OUT / f"{name}_full_model.txt"
                model.booster_.save_model(str(model_path))
                reloaded = previous.lgb.Booster(model_file=str(model_path))
                error = float(np.abs(reloaded.predict(nb.iloc[:1000]) - predictions[name][:1000]).max())
                assert error < 1e-6
                write(OUT / f"{name}_reload_check.json", {"max_error": error, "rows": 1000})
        elif name in SOFT_TARGETS:
            teacher = (original.ORIG_xgb_probability.to_numpy()[tr] + original.ORIG_cat_probability.to_numpy()[tr]) / 2
            model, _ = fit_soft(a, b, y.iloc[tr], teacher, records[name]["alpha"], iterations=records[name]["iterations"])
            predictions[name] = model.predict(b)
            if not confirm:
                model_path = OUT / f"{name}_full_model.txt"
                model.booster_.save_model(str(model_path))
                reloaded = previous.lgb.Booster(model_file=str(model_path))
                error = float(np.abs(reloaded.predict(b.iloc[:1000]) - predictions[name][:1000]).max())
                assert error < 1e-6
                write(OUT / f"{name}_reload_check.json", {"rows": 1000, "max_error": error})
        elif name in TREE_CONFIGS:
            record = records[name]
            model, _ = previous.fit(record["config"], a, y.iloc[tr], cats, iterations=record["iterations"])
            predictions[name] = model.predict_proba(b)[:, 1]
            if not confirm:
                model_path = OUT / f"{name}_full_model.txt"
                model.booster_.save_model(str(model_path))
                reloaded = previous.lgb.Booster(model_file=str(model_path))
                error = float(np.abs(reloaded.predict(b.iloc[:1000]) - predictions[name][:1000]).max())
                assert error < 1e-6
                write(OUT / f"{name}_reload_check.json", {"rows": 1000, "max_error": error})
        elif records[name].get("kind") == "finetuned_tabicl":
            record = records[name]
            command = [str(ROOT / ".tfm_venv/Scripts/python.exe"), "-u", str(ROOT / "finetune_tabicl.py"),
                "--phase", "confirm" if confirm else "test", "--context", str(record["context"]),
                "--epochs", str(record["epochs"]), "--learning-rate", str(record["learning_rate"])]
            subprocess.run(command, check=True)
            predictions[name] = np.load(OUT / f"{name}_{'confirm' if confirm else 'test'}.npy")
        elif name.startswith("tabicl_"):
            record = records[name]
            command = [str(ROOT / ".tfm_venv/Scripts/python.exe"), "-u", str(ROOT / "tabicl_candidate.py"),
                "--phase", "confirm" if confirm else "test", "--context", str(record["context"]),
                "--members", str(record["members"]), "--cache", record.get("cache", "repr"),
                "--offload", record.get("offload", "auto"), "--predict-batch", "50000",
                "--ffn-chunk", str(record.get("ffn_chunk", 0))]
            subprocess.run(command, check=True)
            predictions[name] = np.load(OUT / f"{name}_{'confirm' if confirm else 'test'}.npy")
        else:
            predictions[name], _ = fit_specialists(name, data.iloc[tr], other, a, b, y.iloc[tr].to_numpy(), cats, records[name]["iterations"])
        if confirm:
            scores[name] = float(roc_auc_score(y.iloc[va], predictions[name]))
        np.save(OUT / f"{name}_{'confirm' if confirm else 'test'}.npy", predictions[name])
    combined = blend(predictions, meta["selected"]["weights"], meta["selected"]["method"])
    if confirm:
        score = float(roc_auc_score(y.iloc[va], combined))
        write(OUT / "confirmation.json", {"seed": 2026, "combined_auc": score, "models": scores,
            "goal_reached": score >= 0.963, "settings_frozen": True,
            "selected": meta["selected"], "selection_file": selection_file,
            "note": "Reused exploratory split, not an independent test set."})
        print("CONFIRMED", score, scores, flush=True)
    else:
        assert len(combined) == len(other) and np.isfinite(combined).all()
        assert ((combined >= 0) & (combined <= 1)).all()
        submission = pd.read_csv(ROOT / "sample_submission.csv")
        assert submission.id.equals(other.id) and submission.id.is_unique
        submission["satisfaction"] = combined
        submission.to_csv(OUT / "submission_challenge.csv", index=False)
        exported = pd.read_csv(OUT / "submission_challenge.csv")
        assert exported.id.equals(other.id)
        error = float(np.abs(exported.satisfaction - combined).max())
        assert error < 1e-10
        write(OUT / "verification.json", {"rows": len(exported), "ids_match": True,
            "missing": int(exported.satisfaction.isna().sum()), "export_max_error": error,
            "selected": meta["selected"], "selection_file": selection_file})
        print("SAVED", OUT / "submission_challenge.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["experts", "attention", "neighbors", "soft", "trees", "select", "confirm", "refit"])
    parser.add_argument("--names", choices=list(SEGMENTS), nargs="+", default=list(SEGMENTS))
    parser.add_argument("--selection-file", default="selection.json",
        help="출력 폴더 내 선택 기록 파일명. 다른 실험의 고정된 선택을 보존할 때 사용합니다.")
    args = parser.parse_args()
    if Path(args.selection_file).name != args.selection_file:
        raise ValueError("--selection-file must be a filename within the output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["train.csv", "test.csv"]}
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Dataset changed")
    write(path, manifest)
    if args.phase == "experts":
        tune_experts(args.names)
    elif args.phase == "attention":
        tune_attention()
    elif args.phase == "neighbors":
        tune_neighbors()
    elif args.phase == "soft":
        tune_soft()
    elif args.phase == "trees":
        tune_trees()
    elif args.phase == "select":
        select(args.selection_file)
    else:
        evaluate_or_refit(args.phase == "confirm", args.selection_file)


if __name__ == "__main__":
    main()
