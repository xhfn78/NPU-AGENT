"""저장된 검증 예측으로 조합의 여지를 살펴봅니다. OOF 메타 예측을 계산합니다."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "improve_963_gpu"
OUT = ROOT / "outputs" / "challenge_963"


def features(predictions, rows, contextual):
    p = np.clip(predictions, 1e-6, 1-1e-6)
    logits = np.log(p / (1-p))
    blocks = [logits, p, p * (1-p)]
    if contextual:
        context = rows["Class"].astype(str) + "|" + rows["Type of Travel"].astype(str) + "|" + rows["Customer Type"].astype(str)
        onehot = pd.get_dummies(context, dtype="float64").to_numpy()
        blocks += [onehot, (logits[:, :, None] * onehot[:, None, :]).reshape(len(p), -1)]
    return np.column_stack(blocks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "train.csv")
    y = data.satisfaction.astype(int).to_numpy()
    _, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    names = ["lgb_original_deep", "cat_original_stats", "keras_original_prior", "previous_960"]
    predictions = np.column_stack([np.load(SOURCE / f"{name}_validation.npy") for name in names])
    target = y[va]
    results = []
    best = -1
    for contextual in [False, True]:
        X = features(predictions, data.iloc[va], contextual)
        for C in [0.001, 0.01, 0.1]:
            oof = np.zeros(len(target))
            for tr, valid in StratifiedKFold(5, shuffle=True, random_state=333).split(X, target):
                model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, solver="lbfgs"))
                with threadpool_limits(limits=8):
                    model.fit(X[tr], target[tr])
                    oof[valid] = model.predict_proba(X[valid])[:, 1]
            score = float(roc_auc_score(target, oof))
            result = {"contextual": contextual, "C": C, "meta_oof_auc": score}
            results.append(result)
            print("META RESULT", result, flush=True)
            if score > best:
                best = score
                np.save(OUT / "exploratory_meta_oof.npy", oof)
                chosen = result
    (OUT / "meta_diagnostics.json").write_text(json.dumps({"base_names": names, "best": chosen,
        "results": results, "note": "Base candidates used this split for earlier selection. Meta predictions are cross-fitted; this is exploratory, not an independent test score."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
