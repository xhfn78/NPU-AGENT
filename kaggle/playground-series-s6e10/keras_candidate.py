"""TensorFlow 2.9 GPU 환경에서 실행하는 Keras 비교 모델.

입력 전처리와 데이터 분할은 improve_963.py에서 학습 부분만 사용해 계산합니다.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    folder = args.folder
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    X_train = np.load(folder / "X_train.npy")
    y_train = np.load(folder / "y_train.npy")
    X_other = np.load(folder / "X_other.npy")
    y_valid = None if args.fixed else np.load(folder / "y_valid.npy")

    # 익숙한 Keras의 Sequential -> compile -> fit -> predict 흐름입니다.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(256, activation="swish"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation="swish"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation="swish"),
        tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc", num_thresholds=1000)],
    )

    class ExactAUC(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.best_epoch = -1.0, 0

        def on_epoch_end(self, epoch, logs=None):
            prediction = self.model.predict(X_other, batch_size=8192, verbose=0).ravel()
            score = float(roc_auc_score(y_valid, prediction))
            logs["val_exact_auc"] = score
            if score > self.best:
                self.best, self.best_epoch = score, epoch + 1
            print(f"exact validation AUC={score:.8f}; best={self.best:.8f}", flush=True)

    # 전체 재학습에서도 같은 epoch별 학습률을 사용합니다.
    schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * (0.5 ** (epoch // 10)))
    callbacks = [schedule]
    exact = ExactAUC()
    if not args.fixed:
        callbacks += [exact, tf.keras.callbacks.EarlyStopping(
            monitor="val_exact_auc", mode="max", patience=8, restore_best_weights=True)]
    model.fit(
        X_train, y_train, batch_size=2048, epochs=args.epochs,
        validation_data=None if args.fixed else (X_other, y_valid),
        callbacks=callbacks, verbose=2,
    )
    probability = model.predict(X_other, batch_size=8192, verbose=0).ravel()
    np.save(folder / "probability.npy", probability)
    model.save(folder / "model.h5", include_optimizer=False)
    reference = model.predict(X_other[:1000], batch_size=1024, verbose=0).ravel()
    restored = tf.keras.models.load_model(folder / "model.h5", compile=False)
    error = float(np.abs(restored.predict(X_other[:1000], batch_size=1024, verbose=0).ravel() - reference).max())
    if error > 1e-6:
        raise ValueError(f"Keras model reload mismatch: {error}")
    metrics = {"epochs": args.epochs if args.fixed else exact.best_epoch,
        "validation_roc_auc": None if args.fixed else float(roc_auc_score(y_valid, probability)),
        "model_reload_max_error": error, "tensorflow": tf.__version__}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("KERAS RESULT", metrics, flush=True)


if __name__ == "__main__":
    main()
