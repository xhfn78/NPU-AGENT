"""규제를 강화한 feature-token attention 신경망. TensorFlow 2.9 환경에서 실행합니다."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


@tf.keras.utils.register_keras_serializable(package="Airline")
class NumericalTokens(tf.keras.layers.Layer):
    def __init__(self, width=32, **kwargs):
        super().__init__(**kwargs)
        self.width = width

    def build(self, input_shape):
        features = int(input_shape[-1])
        self.scale = self.add_weight("scale", shape=(features, self.width), initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.l2(1e-5))
        self.offset = self.add_weight("offset", shape=(features, self.width), initializer="glorot_uniform")
        self.cls = self.add_weight("cls", shape=(1, 1, self.width), initializer="zeros")

    def call(self, inputs):
        tokens = tf.expand_dims(inputs, -1) * self.scale[None, :, :] + self.offset[None, :, :]
        cls = tf.broadcast_to(self.cls, (tf.shape(inputs)[0], 1, self.width))
        return tf.concat([cls, tokens], axis=1)

    def get_config(self):
        return {**super().get_config(), "width": self.width}


@tf.keras.utils.register_keras_serializable(package="Airline")
class ClassToken(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, 0, :]


def make_model(features):
    inputs = tf.keras.Input((features,))
    x = tf.keras.layers.GaussianNoise(0.01)(inputs)
    x = NumericalTokens(32)(x)
    for _ in range(2):
        z = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        z = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.1,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5))(z, z)
        x = tf.keras.layers.Add()([x, tf.keras.layers.Dropout(0.15)(z)])
        z = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        z = tf.keras.layers.Dense(64, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(z)
        z = tf.keras.layers.Dropout(0.15)(z)
        z = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(z)
        x = tf.keras.layers.Add()([x, tf.keras.layers.Dropout(0.15)(z)])
    x = ClassToken()(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    x = tf.keras.layers.Dense(64, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.keras.utils.set_random_seed(117)
    folder = args.folder
    X = np.load(folder / "X_train.npy")
    y = np.load(folder / "y_train.npy")
    other = np.load(folder / "X_other.npy")
    valid_y = None if args.fixed else np.load(folder / "y_valid.npy")
    model = make_model(X.shape[1])

    class ExactAUC(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.epoch = -1.0, 0

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(other, batch_size=1024, verbose=0).ravel()
            value = float(roc_auc_score(valid_y, p))
            logs["val_exact_auc"] = value
            if value > self.best:
                self.best, self.epoch = value, epoch + 1
            print(f"attention exact AUC={value:.8f}; best={self.best:.8f}", flush=True)

    metric = ExactAUC()
    schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0006 * (0.5 ** (epoch // 10)))
    callbacks = [schedule]
    early_stop = None
    if not args.fixed:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_exact_auc", mode="max",
            patience=7, restore_best_weights=True)
        callbacks += [metric, early_stop]
    model.fit(X, y, batch_size=1024, epochs=args.epochs, callbacks=callbacks, verbose=2)
    # TensorFlow 2.9에서도 최대 epoch에 도달했을 때 최적 가중치를 복원합니다.
    if early_stop is not None and early_stop.best_weights is not None:
        model.set_weights(early_stop.best_weights)
    p = model.predict(other, batch_size=1024, verbose=0).ravel()
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite neural probabilities")
    np.save(folder / "probability.npy", p)
    model.save(folder / "model.h5", include_optimizer=False)
    reference = model.predict(other[:1000], batch_size=1024, verbose=0).ravel()
    restored = tf.keras.models.load_model(folder / "model.h5", compile=False)
    error = float(np.abs(restored.predict(other[:1000], batch_size=1024, verbose=0).ravel() - reference).max())
    assert error < 1e-6
    result = {"epochs": args.epochs if args.fixed else metric.epoch,
        "validation_auc": None if args.fixed else float(roc_auc_score(valid_y, p)),
        "reload_max_error": error, "tensorflow": tf.__version__}
    (folder / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("ATTENTION FINISHED", result, flush=True)


if __name__ == "__main__":
    main()
