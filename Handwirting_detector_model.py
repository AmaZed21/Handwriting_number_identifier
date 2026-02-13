import tensorflow as tf
from tensorflow import keras #type: ignore
from tensorflow.keras import layers #type: ignore
import pandas as pd
import matplotlib.pyplot as plt

#Loading data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


#Initialise model
model = keras.Sequential([
    layers.Reshape((28, 28, 1)),
    layers.Rescaling(1. / 255),
    layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu'),
    layers.MaxPooling2D(pool_size = (2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation = 'softmax')
])

#Optimizer
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['sparse_categorical_accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 2,
            min_delta = 0.001,
            restore_best_weights = True   
            )

#Fitting model
history = model.fit(
        x_train, y_train,
        validation_data = (x_test, y_test),
        epochs = 20,
        batch_size = 64,
        callbacks = [early_stop]
        )

#Retrieving metrics
history_df = pd.DataFrame(history.history)
history_df[['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()
history_df[['loss', 'val_loss']].plot()
plt.show()