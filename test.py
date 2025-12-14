import tensorflow as tf

# Load your existing model
old_model = tf.keras.models.load_model("Face_mask.keras", compile=False)

# Create a NEW Input layer (no batch_shape)
new_input = tf.keras.Input(shape=(224, 224, 3), name="input")

# Rebuild model
x = new_input
for layer in old_model.layers[1:]:
    x = layer(x)

new_model = tf.keras.Model(inputs=new_input, outputs=x)

# Save in clean Keras format
new_model.save(
    "fm1.keras",
    include_optimizer=False
)

print("✅ Model fixed and saved as Face_mask_fixed.keras")
