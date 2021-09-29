import tensorflow as tf
import onnxruntime
from sklearn.metrics import accuracy_score
import numpy as np
import time

(_, _), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
test_x = (test_x / 255.0).astype("float32")

tf_model = tf.keras.models.load_model("tf_model")
ort_model = onnxruntime.InferenceSession("model.onnx")

tf_start = time.time()
for i in range(100):
    tf_model(test_x)
print("tf running time: ", time.time() - tf_start)

ort_start = time.time()
for i in range(100):
    ort_model.run(None, {"flatten_input": test_x})[0]
print("onnx running time: ", time.time() - ort_start)
