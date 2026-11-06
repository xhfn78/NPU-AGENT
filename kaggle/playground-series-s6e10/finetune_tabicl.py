"""TabICLv2를 학습 부분의 표본으로 미세조정하고 외부 검증은 평가에만 사용합니다."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "data/tfm_cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_MODE"] = "disabled"

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabicl import FinetunedTabICLClassifier

import tabicl_candidate as prep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validation", "confirm", "test"], default="validation")
    parser.add_argument("--context", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    args = parser.parse_args()
    log = logging.getLogger("tabicl._finetune.base")
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    log.propagate = False
    folder = prep.prepare(args.phase)
    X = np.load(folder / "train.npy", mmap_mode="r")
    y = np.load(folder / "y_train.npy")
    other = np.load(folder / "other.npy", mmap_mode="r")
    context, _ = train_test_split(np.arange(len(y)), train_size=args.context, stratify=y, random_state=73)
    name = f"tabicl_ft_c{args.context}_ep{args.epochs}"
    run = prep.OUT / f"{name}_{args.phase}"
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model = FinetunedTabICLClassifier(epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=0.01, grad_clip=1.0, amp=True, n_estimators_finetune=1,
        n_estimators_validation=1, n_estimators_inference=1, max_data_size=4096,
        validation_split_ratio=0.1, early_stopping=True, patience=2, min_delta=0.00001,
        time_limit=600, save_interval=0, freeze_col=True, device="cuda", random_state=73,
        verbose=False, model_path=prep.CHECKPOINT, allow_auto_download=False, wandb_kwargs=None,
        extra_classifier_kwargs={"batch_size": 1, "kv_cache": "repr", "use_fa3": False,
            "use_amp": True, "offload_mode": "auto", "n_jobs": 8})
    start = time.perf_counter()
    print("FINE-TUNE START", name, X[context].shape, flush=True)
    model.fit(X[context], y[context], output_dir=run / "checkpoints")
    print("FINE-TUNE DONE", time.perf_counter()-start, flush=True)
    prediction = np.empty(len(other))
    for begin in range(0, len(other), 10000):
        end = min(begin+10000, len(other))
        prediction[begin:end] = model.predict_proba(other[begin:end])[:, 1]
        print("FINETUNED PREDICTED", end, "/", len(other), flush=True)
    assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    np.save(prep.OUT / f"{name}_{args.phase}.npy", prediction)
    score = None if args.phase == "test" else float(roc_auc_score(np.load(folder / "y_other.npy"), prediction))
    metadata = {"name": name, "kind": "finetuned_tabicl", "context": args.context, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "phase": args.phase, "validation_auc": score,
        "total_seconds": time.perf_counter()-start, "external_validation_used_for_fit": False,
        "internal_best_auc": model._best_metric_, "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20}
    joblib.dump(model, run / "model.joblib")
    if args.phase == "validation":
        (prep.OUT / f"{name}_record.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("FINETUNED RESULT", metadata, flush=True)


if __name__ == "__main__":
    main()
