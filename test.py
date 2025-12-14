import keras
import tensorflow as tf

# Load with keras (Keras 3)
model = keras.models.load_model(
    "Face_mask.keras",
    compile=False
)

# Save as H5 (legacy, most compatible)
model.save("face_mask_compatible.h5")

print("✅ Model converted to H5 format")
