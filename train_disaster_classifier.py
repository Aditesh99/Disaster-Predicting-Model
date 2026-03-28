# train_disaster_classifier.py
# Fixed & Complete Version
#
# Requirements:
#   pip install tensorflow scikit-learn matplotlib
#
# Folder structure expected:
# DATA_DIR/
#   biological and chemical pandemic/
#   cyclone/
#   drought/
#   earthquake/
#   flood/
#   landslide/
#   tsunami/
#   wildfire/

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ============================================================
#  USER CONFIG — change only these lines
# ============================================================
DATA_DIR      = r"C:\data science\disaster_predicting\disasters"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 30
FINE_TUNE_EPOCHS = 15
LEARNING_RATE = 1e-4
MODEL_SAVE    = "disaster_efficientnetb0.h5"
CLASSES_JSON  = "model_classes.json"   # saved so predict script reads same order
# ============================================================

# ---------- 1. Detect classes from folder (alphabetical = keras default) ----------
class_names = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])
NUM_CLASSES = len(class_names)
print(f"[INFO] Detected {NUM_CLASSES} classes: {class_names}")

# Save class list so the prediction script always uses the same order
with open(CLASSES_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)
print(f"[INFO] Class names saved to {CLASSES_JSON}")

# ---------- 2. Data generators ----------
train_datagen = ImageDataGenerator(
    preprocessing_function=effnet_preprocess,
    rotation_range=20,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.8, 1.2),
    validation_split=0.15,
    fill_mode="nearest",
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,   # CRITICAL: explicit list keeps index order fixed
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,   # same order as train_gen
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

print("[INFO] Class → index mapping:", train_gen.class_indices)

# ---------- 3. Class weights (handles imbalanced data) ----------
y_train_labels = train_gen.classes
cw_array = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels,
)
class_weights = dict(enumerate(cw_array))
print("[INFO] Class weights:", class_weights)

# ---------- 4. Build model ----------
base_model = EfficientNetB0(
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    weights="imagenet",
)
base_model.trainable = False   # freeze for initial training

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)
model.summary()

# ---------- 5. Callbacks ----------
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
)
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
)

# ---------- 6. Phase 1 — train head only ----------
print("\n[PHASE 1] Training classifier head (base frozen)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop],
)

# ---------- 7. Phase 2 — fine-tune top layers ----------
print("\n[PHASE 2] Fine-tuning top 30 layers of base model...")
base_model.trainable = True
for layer in base_model.layers[:-30]:   # freeze all but last 30
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop],
)

# ---------- 8. Save final model ----------
model.save(MODEL_SAVE)
print(f"[INFO] Final model saved to {MODEL_SAVE}")

# ---------- 9. Evaluate on validation set ----------
val_loss, val_acc, val_auc = model.evaluate(val_gen, verbose=1)
print(f"\n[RESULT] val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_auc={val_auc:.4f}")

# ---------- 10. Plot training curves ----------
all_loss     = history.history.get("loss", [])     + history_ft.history.get("loss", [])
all_val_loss = history.history.get("val_loss", []) + history_ft.history.get("val_loss", [])
all_acc      = history.history.get("accuracy", []) + history_ft.history.get("accuracy", [])
all_val_acc  = history.history.get("val_accuracy", []) + history_ft.history.get("val_accuracy", [])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(all_loss, label="train")
plt.plot(all_val_loss, label="val")
plt.axvline(x=len(history.history["loss"]) - 1, color="gray", linestyle="--", label="fine-tune start")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(all_acc, label="train")
plt.plot(all_val_acc, label="val")
plt.axvline(x=len(history.history["accuracy"]) - 1, color="gray", linestyle="--", label="fine-tune start")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("[INFO] Training curves saved to training_curves.png")
