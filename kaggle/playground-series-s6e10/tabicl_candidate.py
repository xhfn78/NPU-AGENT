"""공식 TabICLv2를 로컬 GPU에서 평가합니다. 가중치는 사전 다운로드합니다.

.tfm_venv/Scripts/python.exe tabicl_candidate.py --phase pilot --context 10000
파일럿 점수는 전체 검증 점수로 취급하지 않습니다.
"""

import argparse
import hashlib
import json
import os
import time
from types import MethodType
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "data" / "tfm_cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier

import challenge_models as campaign


OUT = campaign.OUT
CHECKPOINT = ROOT / "data/tabicl_model/tabicl-classifier-v2-20260212.ckpt"


def chunked_ffn(self, values):
    """평가 전용: 행별로 독립인 FFN만 나눠 계산합니다. Attention은 바꾸지 않습니다."""
    size = self._inference_ffn_chunk
    if self.training or values.shape[-2] <= size:
        return self._original_ffn(values)
    first = self._original_ffn(values[..., :size, :])
    result = torch.empty(values.shape, dtype=first.dtype, device=values.device)
    result[..., :size, :] = first
    for begin in range(size, values.shape[-2], size):
        result[..., begin:begin+size, :] = self._original_ffn(values[..., begin:begin+size, :])
    return result


def reduce_ffn_memory(model, chunk):
    blocks = [block for block in model.model_.icl_predictor.tf_icl.modules() if hasattr(block, "_ff_block")]
    for block in blocks:
        block._original_ffn = block._ff_block
        block._inference_ffn_chunk = chunk
        block._ff_block = MethodType(chunked_ffn, block)
    # 작은 입력으로 같은 연산인지 검사합니다. 학습 가중치를 수정하지 않습니다.
    block = blocks[0]
    width = block.linear1.in_features
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        values = torch.randn(1, chunk+101, width, device="cuda", dtype=torch.float16)
        original = block._original_ffn(values)
        changed = block._ff_block(values)
        error = float((original-changed).abs().max())
        if not torch.allclose(original, changed, atol=0.003, rtol=0.005):
            raise ValueError(f"Chunked FFN mismatch: {error}")
    print("FFN CHUNK CHECK", len(blocks), "layers; max error", error, flush=True)
    return error


def prepare(phase):
    source_manifest = json.loads((campaign.SOURCE / "data_manifest.json").read_text(encoding="utf-8"))
    for path in [ROOT / "train.csv", ROOT / "test.csv", ROOT / "data/original_airline/original_airline.zip"]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_manifest[str(path.relative_to(ROOT))]:
            raise ValueError("Source data changed; do not reuse feature/model caches")
    tag = "full" if phase == "test" else "2026" if phase == "confirm" else "42"
    folder = OUT / f"tabicl_features_{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "ready.json").exists():
        return folder
    data, y = campaign.previous.base.load_data()
    original = campaign.previous.priors("train")
    if phase == "test":
        tr = np.arange(len(y))
        other = pd.read_csv(ROOT / "test.csv")
        original_other = campaign.previous.priors("test")
        va = None
    else:
        seed = 2026 if phase == "confirm" else 42
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        other, original_other = data.iloc[va], original.iloc[va]
    a, b, _, bundle = campaign.previous.inputs(data.iloc[tr], other, y.iloc[tr], original.iloc[tr], original_other, campaign.CONFIG)
    raw = campaign.previous.base.make_features(data, "raw").columns.tolist()
    extra = ["TE_Flight Distance", "TE_Age", "TE_Flight Distance_by_Class", "TE_Flight Distance_by_Type of Travel",
        "TE_Online boarding", "TE_Inflight wifi service", "ORIG_xgb_probability", "ORIG_cat_probability"]
    columns = raw + extra + [c for c in a if c.startswith("route_class_")]
    a, b = a.loc[:, columns].copy(), b.loc[:, columns].copy()
    for col in a.select_dtypes(include=["category"]).columns:
        a[col] = a[col].cat.codes.astype("float32").replace(-1, np.nan)
        b[col] = b[col].cat.codes.astype("float32").replace(-1, np.nan)
    np.save(folder / "train.npy", a.to_numpy(dtype="float32"))
    np.save(folder / "other.npy", b.to_numpy(dtype="float32"))
    np.save(folder / "y_train.npy", y.iloc[tr].to_numpy())
    np.save(folder / "train_indices.npy", tr)
    if va is not None:
        np.save(folder / "y_other.npy", y.iloc[va].to_numpy())
        np.save(folder / "validation_indices.npy", va)
    joblib.dump(bundle, folder / "preprocessing.joblib")
    (folder / "ready.json").write_text(json.dumps({"phase": phase, "columns": columns,
        "train_rows": len(a), "other_rows": len(b)}, indent=2), encoding="utf-8")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "validation", "confirm", "test"], default="pilot")
    parser.add_argument("--context", type=int, default=10000)
    parser.add_argument("--members", type=int, default=1)
    parser.add_argument("--pilot-rows", type=int, default=20000)
    parser.add_argument("--predict-batch", type=int, default=10000)
    parser.add_argument("--cache", choices=["kv", "repr", "none"], default="kv")
    parser.add_argument("--offload", choices=["auto", "cpu", "disk"], default="auto")
    parser.add_argument("--ffn-chunk", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((campaign.SOURCE / "data_manifest.json").read_text(encoding="utf-8"))
    for path in [ROOT / "train.csv", ROOT / "test.csv", ROOT / "data/original_airline/original_airline.zip"]:
        key = str(path.relative_to(ROOT))
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_manifest[key]:
            raise ValueError("Source data changed; do not reuse feature/model caches")
    folder = prepare(args.phase)
    X = np.load(folder / "train.npy", mmap_mode="r")
    other = np.load(folder / "other.npy", mmap_mode="r")
    y = np.load(folder / "y_train.npy")
    y_other = None if args.phase == "test" else np.load(folder / "y_other.npy")
    if args.context >= len(y):
        context = np.arange(len(y))
    else:
        context, _ = train_test_split(np.arange(len(y)), train_size=args.context, stratify=y, random_state=73)
    other_indices = np.arange(len(other))
    if args.phase == "pilot" and args.pilot_rows < len(other):
        other_indices, _ = train_test_split(other_indices, train_size=args.pilot_rows, stratify=y_other, random_state=777)
    name = f"tabicl_c{len(context)}_e{args.members}"
    torch.set_num_threads(8)
    model = TabICLClassifier(n_estimators=args.members, batch_size=1, kv_cache=False if args.cache == "none" else args.cache,
        model_path=CHECKPOINT, allow_auto_download=False, device="cuda", use_amp=True, use_fa3=False,
        offload_mode=args.offload, disk_offload_dir=str(OUT / "tabicl_offload"), random_state=73, n_jobs=8)
    print("TABICL FIT", name, X[context].shape, "queries", len(other_indices), flush=True)
    start = time.perf_counter()
    model.fit(X[context], y[context])
    ffn_error = reduce_ffn_memory(model, args.ffn_chunk) if args.ffn_chunk else None
    fit_seconds = time.perf_counter() - start
    print("TABICL FIT DONE", fit_seconds, flush=True)
    probability = np.empty(len(other_indices))
    for begin in range(0, len(other_indices), args.predict_batch):
        rows = other_indices[begin:begin+args.predict_batch]
        probability[begin:begin+len(rows)] = model.predict_proba(other[rows])[:, 1]
        print("TABICL PREDICTED", begin+len(rows), "/", len(other_indices), flush=True)
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite TabICL predictions")
    elapsed = time.perf_counter() - start
    score = None if y_other is None else float(roc_auc_score(y_other[other_indices], probability))
    metadata = {"name": name, "kind": "tabicl", "context": len(context), "members": args.members,
        "phase": args.phase, "rows": len(probability), "validation_auc": score,
        "cache": args.cache,
        "offload": args.offload,
        "ffn_chunk": args.ffn_chunk, "ffn_max_error": ffn_error,
        "fit_seconds": fit_seconds, "total_seconds": elapsed,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
        "torch": torch.__version__, "tabicl": version("tabicl")}
    if args.phase == "pilot":
        np.save(OUT / f"{name}_pilot.npy", probability)
        np.save(OUT / f"{name}_pilot_indices.npy", other_indices)
        incumbent = campaign.incumbent("validation")[other_indices]
        metadata["incumbent_on_same_pilot_auc"] = float(roc_auc_score(y_other[other_indices], incumbent))
        metadata["note"] = "Pilot subset only; not the full validation score."
    else:
        suffix = "validation" if args.phase == "validation" else args.phase
        np.save(OUT / f"{name}_{suffix}.npy", probability)
        model.save(OUT / f"{name}_{suffix}_model.pkl", save_model_weights=False, save_training_data=True, save_kv_cache=False)
        if args.phase == "validation":
            (OUT / f"{name}_record.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / f"{name}_{args.phase}_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("TABICL RESULT", metadata, flush=True)


if __name__ == "__main__":
    main()
