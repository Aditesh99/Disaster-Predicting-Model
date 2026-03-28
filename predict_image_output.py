# predict_image_output.py
# Fixed & Complete Version
#
# Usage:
#   python predict_image_output.py --image path/to/image.jpg
#   python predict_image_output.py --image path/to/image.jpg --model disaster_efficientnetb0.h5
#   python predict_image_output.py --image path/to/image.jpg --top 3

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# ============================================================
#  CONFIG — change these if your files are in different paths
# ============================================================
DEFAULT_MODEL       = "disaster_efficientnetb0.h5"
DEFAULT_CLASSES_JSON = "model_classes.json"
DATA_DIR            = r"C:\data science\disaster_predicting\disasters"
IMG_SIZE            = (224, 224)
# ============================================================


def load_class_names(json_path: str, data_dir: str) -> list:
    """
    Load class names in the EXACT same order used during training.
    Priority:
      1. model_classes.json  (written by training script — most reliable)
      2. Sorted subfolder names from DATA_DIR  (same logic as ImageDataGenerator)
      3. Hardcoded fallback
    """
    # 1. JSON file written by the training script
    if Path(json_path).exists():
        with open(json_path, "r", encoding="utf-8") as f:
            names = json.load(f)
        print(f"[INFO] Loaded {len(names)} classes from {json_path}")
        return names

    # 2. Read from DATA_DIR subfolders (alphabetical = Keras default)
    if Path(data_dir).exists():
        names = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        if names:
            print(f"[INFO] Loaded {len(names)} classes from folder: {data_dir}")
            return names

    # 3. Hardcoded fallback (alphabetical — must match training order exactly)
    names = [
        "biological and chemical pandemic",
        "cyclone",
        "drought",
        "earthquake",
        "flood",
        "landslide",
        "tsunami",
        "wildfire",
    ]
    print("[WARNING] Using hardcoded class list. Make sure this matches your training order!")
    return names


def predict_image(img_path: str, model, class_names: list, top_k: int = 1):
    """
    Predict the disaster class for a single image.

    Returns:
        results (list of dict): top_k predictions with 'class' and 'confidence'
    """
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load & preprocess
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)                     # shape: (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)                 # shape: (1, 224, 224, 3)
    arr = effnet_preprocess(arr)                      # EfficientNet normalisation

    # Predict
    preds = model.predict(arr, verbose=0)[0]          # shape: (NUM_CLASSES,)

    # Top-k results
    top_indices = np.argsort(preds)[::-1][:top_k]
    results = [
        {"class": class_names[i], "confidence": float(preds[i])}
        for i in top_indices
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="Disaster Image Classifier")
    parser.add_argument("--image",   required=True,  help="Path to input image")
    parser.add_argument("--model",   default=DEFAULT_MODEL,        help="Path to .h5 model file")
    parser.add_argument("--classes", default=DEFAULT_CLASSES_JSON, help="Path to model_classes.json")
    parser.add_argument("--top",     type=int, default=1,          help="Show top-N predictions")
    args = parser.parse_args()

    # Load model
    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            "Make sure you run train_disaster_classifier.py first."
        )
    print(f"[INFO] Loading model: {args.model}")
    model = keras.models.load_model(args.model)

    # Load class names
    class_names = load_class_names(args.classes, DATA_DIR)
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")

    # Sanity check
    last_layer_units = model.output_shape[-1]
    if last_layer_units != len(class_names):
        raise ValueError(
            f"Model output has {last_layer_units} units but "
            f"{len(class_names)} class names were loaded. "
            "Class list does not match the model — fix model_classes.json."
        )

    # Predict
    results = predict_image(args.image, model, class_names, top_k=args.top)

    print(f"\n[RESULT] Image: {args.image}")
    print("-" * 40)
    for rank, r in enumerate(results, 1):
        bar = "█" * int(r["confidence"] * 30)
        print(f"  #{rank}  {r['class']:<40}  {r['confidence']*100:5.1f}%  {bar}")
    print("-" * 40)
    print(f"  >> Predicted: {results[0]['class'].upper()}  ({results[0]['confidence']*100:.1f}% confidence)")


# -------------------------------------------------------
# Direct import usage (e.g. in a notebook or Flask app):
#
#   from predict_image_output import load_class_names, predict_image
#   from tensorflow import keras
#   model      = keras.models.load_model("disaster_efficientnetb0.h5")
#   class_names = load_class_names("model_classes.json", DATA_DIR)
#   result     = predict_image("my_photo.jpg", model, class_names, top_k=3)
#   print(result)
# -------------------------------------------------------
if __name__ == "__main__":
    main()
