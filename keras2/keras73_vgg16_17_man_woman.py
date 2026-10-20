"""VGG16 transfer learning: man-or-woman (binary labels)."""
import time
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

NP_PATH = "./_data/faces_npy/"

def prepare_images(images):
    images = images.astype("float32")
    if images.max() <= 1.0:
        images *= 255.0
    return preprocess_input(images)

x_train = prepare_images(np.load(NP_PATH + "faces_01_x_train.npy"))
y_train = np.load(NP_PATH + "faces_01_y_train.npy")
x_test = prepare_images(np.load(NP_PATH + "faces_01_x_test.npy"))
y_test = np.load(NP_PATH + "faces_01_y_test.npy")

base_model = VGG16(weights="imagenet", include_top=False, input_shape=x_train.shape[1:])
base_model.trainable = False
model = Sequential([base_model, GlobalAveragePooling2D(), Dense(128, activation="relu"),
                    Dropout(0.3), Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

start_time = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1,
          callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
y_pred = (model.predict(x_test, verbose=0).ravel() >= 0.5).astype(int)
print(f"loss: {loss:.4f}\naccuracy: {accuracy:.4f}\naccuracy_score: {accuracy_score(y_test, y_pred):.4f}")
print(f"elapsed time: {time.time() - start_time:.2f} seconds")
