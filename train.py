import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
input_names = [n.name for n in model.inputs]
output_names = [n.name for n in model.outputs]
print(input_names)
print(output_names)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.fit(train_x, train_y)
model.save("tf_model", include_optimizer=False)
