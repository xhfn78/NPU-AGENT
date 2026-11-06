"""공개 원본의 같은 비행 거리에서 유사 승객을 찾는 보조 특징. 대회 정답은 읽지 않습니다."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import improve_auc as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "challenge_963"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "train.csv", ROOT / "test.csv", ROOT / "data/original_airline/original_airline.zip"]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    path = OUT / "neighbor_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Source data changed")
        if (OUT / "neighbor_test.npy").exists() and not args.force:
            print("NEIGHBOR FEATURES CACHED", flush=True)
            return
    original = pd.concat([pd.read_csv(ROOT / "data/original_airline/raw" / name) for name in ["train.csv", "test.csv"]], ignore_index=True)
    # 대회 train의 satisfaction 열을 제외하고, 공개 원본의 레이블만 사용합니다.
    columns = [c for c in pd.read_csv(ROOT / "test.csv", nrows=1).columns if c != "id"]
    train = pd.read_csv(ROOT / "train.csv", usecols=columns)
    test = pd.read_csv(ROOT / "test.csv", usecols=columns)
    query = pd.concat([train, test], ignore_index=True)
    source_y = original.satisfaction.eq("satisfied").to_numpy(dtype="float32")
    n = len(query)
    output = np.full((n, 13), np.nan, dtype="float32")
    orig_ratings = original[base.RATING_COLUMNS].to_numpy(dtype="float32")
    query_ratings = query[base.RATING_COLUMNS].to_numpy(dtype="float32")
    cat_cols = ["Class", "Customer Type", "Type of Travel", "Gender"]
    source_cats, query_cats = [], []
    for col in cat_cols:
        values = {value: i for i, value in enumerate(original[col].unique())}
        source_cats.append(original[col].map(values).to_numpy())
        query_cats.append(query[col].map(values).fillna(-1).to_numpy())
    source_cats, query_cats = np.column_stack(source_cats), np.column_stack(query_cats)
    original_groups = original.groupby("Flight Distance").indices
    groups = query.groupby("Flight Distance").indices
    for number, (distance, indices) in enumerate(groups.items()):
        source_indices = original_groups.get(distance)
        if source_indices is None:
            output[indices] = [30, 30, 0, 0.5, 0.5, 30, 30, 30, 0, 0.5, 0.5, 30, 1]
            continue
        labels = source_y[source_indices]
        for start in range(0, len(indices), 256):
            rows = indices[start:start+256]
            shape = (len(rows), len(source_indices))
            rating_distance = np.zeros(shape, dtype="float32")
            hamming = np.zeros(shape, dtype="float32")
            mismatch = np.zeros(shape, dtype="float32")
            for i in range(len(base.RATING_COLUMNS)):
                diff = np.abs(query_ratings[rows, i, None] - orig_ratings[source_indices, i][None, :])
                rating_distance += diff / 5
                hamming += diff != 0
            for i in range(len(cat_cols)):
                mismatch += query_cats[rows, i, None] != source_cats[source_indices, i][None, :]
            age = np.abs(query.Age.to_numpy()[rows, None] - original.Age.to_numpy()[source_indices][None, :])
            d1 = rating_distance + 2 * mismatch + age / 20
            d2 = hamming + 2 * mismatch + age / 40
            values = []
            for dist in [d1, d2]:
                pos = np.min(dist[:, labels == 1], axis=1) if (labels == 1).any() else np.full(len(rows), 30)
                neg = np.min(dist[:, labels == 0], axis=1) if (labels == 0).any() else np.full(len(rows), 30)
                k = min(7, len(labels))
                nearest = np.argpartition(dist, k-1, axis=1)[:, :k]
                nearest = np.take_along_axis(nearest, np.argsort(np.take_along_axis(dist, nearest, axis=1), axis=1), axis=1)
                near_labels = labels[nearest]
                values.extend([pos, neg, neg-pos, near_labels[:, :min(3,k)].mean(axis=1),
                    near_labels.mean(axis=1), dist.min(axis=1)])
            output[rows, :12] = np.column_stack(values)
            output[rows, 12] = 0
        if number % 500 == 0:
            print("NEIGHBORS", number, "/", len(groups), flush=True)
    if not np.isfinite(output).all():
        raise ValueError("Incomplete neighbor features")
    np.save(OUT / "neighbor_train.npy", output[:len(train)])
    np.save(OUT / "neighbor_test.npy", output[len(train):])
    (OUT / "neighbor_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("NEIGHBOR FEATURES READY", output.shape, flush=True)


if __name__ == "__main__":
    main()
