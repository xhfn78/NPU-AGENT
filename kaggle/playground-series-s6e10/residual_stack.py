"""기존 조합의 logit에 작은 교차검증 보정 모델을 더하는 실험."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".local_packages"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import challenge_models as challenge


def main():
    data, target = challenge.previous.base.load_data()
    _, indices = train_test_split(np.arange(len(target)), test_size=0.2, stratify=target, random_state=42)
    y = target.iloc[indices].to_numpy()
    X = data.iloc[indices].drop(columns=["id", "satisfaction"]).reset_index(drop=True)
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("category")
    original = challenge.previous.priors("train").iloc[indices]
    for col in original:
        X[col] = original[col].to_numpy()
    neighbor = np.load(challenge.OUT / "neighbor_train.npy")[indices]
    for i in range(neighbor.shape[1]):
        X[f"NEIGH_{i}"] = neighbor[:, i]
    baseline = challenge.incumbent("validation")
    margin = logit(np.clip(baseline, 1e-6, 1-1e-6))
    X["base_margin"] = margin
    results = []
    for trees in [100, 250]:
        prediction = np.zeros(len(y))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=733).split(X, y)):
            model = lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.02, num_leaves=7,
                min_child_samples=1000, reg_lambda=100.0, max_bin=511, n_jobs=8,
                random_state=42, objective="binary", boost_from_average=False,
                verbosity=-1, force_col_wise=True)
            model.fit(X.iloc[tr], y[tr], init_score=margin[tr])
            prediction[va] = expit(margin[va] + model.predict(X.iloc[va], raw_score=True))
            print("RESIDUAL FOLD", trees, fold, flush=True)
        score = float(roc_auc_score(y, prediction))
        results.append({"trees": trees, "oof_auc": score})
        np.save(challenge.OUT / f"residual_{trees}_oof.npy", prediction)
        print("RESIDUAL RESULT", trees, score, flush=True)
    result = {"baseline_auc": float(roc_auc_score(y, baseline)), "results": results,
        "note": "Meta corrections are cross-fitted. Base settings used this split for earlier selection; not an independent test."}
    (challenge.OUT / "residual_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
