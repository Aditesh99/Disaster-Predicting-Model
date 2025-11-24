# train_disaster_classifier.py
# Requirements:
#   pip install tensorflow tensorflow-addons matplotlib
# Adjust DATA_DIR to point at a folder with subfolders per class:
# DATA_DIR/
#   wildfire/
#   tsunami/
#   flood/
#   landslide/
#   earthquake/
#   cyclone/
#   drought/
#   biological_pandemic/
#   chemical_pandemic/

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ---------- USER CONFIG ----------
DATA_DIR = "C:\data science\disaster_predicting\disasters"  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
MODEL_SAVE = "disaster_efficientnetb0.h5"
# ---------------------------------

# automatic detection of classes from folder names
class_names = ['biological and chemical pandemic', 'cyclone', 'drought', 'earthquake', 'flood', 'landslide', 'tsunami', 'wildfire']
NUM_CLASSES = len(class_names)
print("Detected classes:", class_names)

# Train/Val/Test split using ImageDataGenerator flow_from_directory
# We'll generate train+val from the directory using validation_split; keep a separate test_dir if you have one.
train_datagen = ImageDataGenerator(
    preprocessing_function=effnet_preprocess,
    rotation_range=20,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.8,1.2),
    validation_split=0.15,  # 15% val
    fill_mode='nearest'
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Compute class weights to handle imbalanced classes
y_train_labels = train_gen.classes  # integer class indices
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Build the model using EfficientNetB0 base
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), weights='imagenet')
base_model.trainable = False  # freeze for initial training

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs
x = effnet_preprocess(x)  # already done in generator, but safe
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
model.summary()

# Callbacks
checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

# Train (initial)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Fine-tune: unfreeze some layers and continue training
base_model.trainable = True
# optionally unfreeze last N layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

fine_tune_epochs = 15
history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=fine_tune_epochs,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Save final model
model.save(MODEL_SAVE)
print("Saved model to", MODEL_SAVE)

# Quick evaluation on validation set (or point to a separate test set)
val_loss, val_acc, val_auc = model.evaluate(val_gen, verbose=1)
print(f"Validation loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f}")

# Plot training curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history.get('loss', []) + history_ft.history.get('loss', []))
plt.plot(history.history.get('val_loss', []) + history_ft.history.get('val_loss', []))
plt.title('Loss')
plt.legend(['train','val'])
plt.subplot(1,2,2)
plt.plot(history.history.get('accuracy', []) + history_ft.history.get('accuracy', []))
plt.plot(history.history.get('val_accuracy', []) + history_ft.history.get('val_accuracy', []))
plt.title('Accuracy')
plt.legend(['train','val'])
plt.tight_layout()
plt.show()

# Inference helper
def predict_image(path, model, class_names):
    img = image.load_img(path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = effnet_preprocess(arr)
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    return class_names[idx], preds[idx]

# Example usage:
# print(predict_image("/path/to/sample.jpg", model, class_names))
