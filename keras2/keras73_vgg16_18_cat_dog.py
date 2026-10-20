"""VGG16 transfer learning: cat-or-dog (binary labels)."""
import time
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

NP_PATH = "./_data/kaggle_cat_dog_npy/"

def prepare_images(images):
    images = images.astype("float32")
    if images.max() <= 1.0:
        images *= 255.0
    return preprocess_input(images)

x_train = prepare_images(np.load(NP_PATH + "keras45_03_x_train.npy"))
y_train = np.load(NP_PATH + "keras45_03_y_train.npy")
x_test = prepare_images(np.load(NP_PATH + "keras45_03_x_test.npy"))
y_test = np.load(NP_PATH + "keras45_03_y_test.npy")

base_model = VGG16(weights="imagenet", include_top=False, input_shape=x_train.shape[1:])
base_model.trainable = False
model = Sequential([base_model, GlobalAveragePooling2D(), Dense(128, activation="relu"),
                    Dropout(0.3), Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

start_time = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=1,
          callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
y_pred = (model.predict(x_test, verbose=0).ravel() >= 0.5).astype(int)
print(f"loss: {loss:.4f}\naccuracy: {accuracy:.4f}\naccuracy_score: {accuracy_score(y_test, y_pred):.4f}")
print(f"elapsed time: {time.time() - start_time:.2f} seconds")


'''
- loss: 0.0096 - accuracy: 0.9972 - val_loss: 0.1590 - val_accuracy: 0.9613
loss: 0.0695
accuracy: 0.9840
accuracy_score: 0.9840
elapsed time: 101.12 seconds

11s 55ms/step - loss: 0.0218 - accuracy: 0.9931 - val_loss: 0.1603 - val_accuracy: 0.9688
loss: 0.1438
accuracy: 0.9600
accuracy_score: 0.9600
elapsed time: 127.38 seconds


True
p - loss: 0.6931 - accuracy: 0.5063 - val_loss: 0.6928 - val_accuracy: 0.5150
loss: 0.6749
accuracy: 1.0000
accuracy_score: 1.0000
elapsed time: 255.09 seconds

'''