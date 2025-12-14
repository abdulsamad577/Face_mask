import tensorflow as tf

model = tf.keras.models.load_model("Face_mask.keras")

# Re-save model in compatible format
model.save("fm.keras", save_format="keras")
