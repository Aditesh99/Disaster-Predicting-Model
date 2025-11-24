import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# -------------------------
# 1. Load your saved model
# -------------------------
model = keras.models.load_model("disaster_efficientnetb0.h5")

# -------------------------
# 2. Your class names (same order as training)
# -------------------------
class_names = [
    "cyclone",
    "drought",
    "earthquake",
    "flood",
    "landslide",
    "tsunami",
    "wildfire",
    "pandemic"   # or your exact folder names
]

# -------------------------
# 3. Prediction function
# -------------------------
def predict_image(img_path):
    IMG_SIZE = (224, 224)

    # Load and resize image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    
    # Convert to array
    img_array = image.img_to_array(img)

    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess for EfficientNet
    img_array = effnet_preprocess(img_array)

    # Predict
    predictions = model.predict(img_array)[0]

    # Pick highest probability
    index = np.argmax(predictions)
    confidence = predictions[index]

    # Return result
    return class_names[index], float(confidence)

# -------------------------
# 4. Test your function
# -------------------------
label, confidence = predict_image("522.jpg")   # <-- put your image here

print(f"Predicted Class: {label}")
print(f"Confidence: {confidence:.4f}")
