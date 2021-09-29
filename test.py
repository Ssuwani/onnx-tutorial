import onnxruntime
from tensorflow.keras.datasets import mnist
import numpy as np

(_, _), (test_x, test_y) = mnist.load_data()
test_x = (test_x / 255.0).astype("float32")

ort_model = onnxruntime.InferenceSession("model.onnx")
result = ort_model.run(None, {"flatten_input": test_x[0:1]})[0]

print("predict: ", np.argmax(result))
print("real: ", test_y[0])
