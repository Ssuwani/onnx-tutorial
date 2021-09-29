import tensorflow as tf

(train_x, train_y), (_, _) = tf.keras.datasets.mnist.load_data()
train_x = train_x / 255.0
train_x = train_x.reshape(-1, 28, 28, 1).astype("float32")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.fit(train_x, train_y)
model.save("tf_model_heavy", include_optimizer=False)
