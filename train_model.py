"""Run this locally or in Colab to (re)train the model. NOT used by Streamlit."""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dermamnist.npz")  # rename your dataset file to this
MODEL_PATH = os.path.join(BASE_DIR, "skin_cancer_model.keras")

data = np.load(DATA_PATH)
x_train, y_train = data["train_images"], data["train_labels"]
x_val, y_val = data["val_images"], data["val_labels"]
x_test, y_test = data["test_images"], data["test_labels"]

print("Train:", x_train.shape, y_train.shape)
print("Val:  ", x_val.shape, y_val.shape)
print("Test: ", x_test.shape, y_test.shape)

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = models.Sequential([
    layers.Input(shape=(28, 28, 3)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(7, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(x_val, y_val),
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

model.save(MODEL_PATH)
size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Saved {MODEL_PATH} ({size_mb:.2f} MB)")
assert size_mb < 25.0, "Saved model exceeds 25 MB!"
