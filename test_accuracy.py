import tensorflow as tf
import onnxruntime
from sklearn.metrics import accuracy_score
import numpy as np

(_, _), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
test_x = (test_x / 255.0).astype("float32")

tf_model = tf.keras.models.load_model("tf_model")
ort_model = onnxruntime.InferenceSession("model.onnx")

tf_result = tf_model(test_x)
ort_result = ort_model.run(None, {"flatten_input": test_x})[0]

print("tf result: ", accuracy_score(np.argmax(tf_result, axis=1), test_y))
print("onnx result: ", accuracy_score(np.argmax(ort_result, axis=1), test_y))
