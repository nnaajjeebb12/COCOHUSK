import tensorflow as tf

MODEL_PATH = "coconut_husk_quality_model.h5"
TFLITE_MODEL_PATH = "coconut_husk_quality_model.tflite"

model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # basic optimization
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
  f.write(tflite_model)

print("Saved", TFLITE_MODEL_PATH)