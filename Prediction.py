import tensorflow as tf
from tensorflow import keras #type: ignore
from tensorflow.keras import layers #type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

#Training of bot done in other file

#Splitting multiple numbers into single digits
def splitting_num(image_path: str) -> str: 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img =  cv2.GaussianBlur(img, (5, 5), 0)
        binarised_img = cv2.adaptiveThreshold(
                        img, 255,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY_INV,
                        blockSize= 31,
                        C = 10
                        )
        contours, _ = cv2.findContours(binarised_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        nums = []
        for num in contours:
            x, y, w, h = cv2.boundingRect(num)
            if w > 5 and h > 5:
                cropped = binarised_img[y:y+h, x:x+w]
                border = cv2.copyMakeBorder(cropped, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                resized = cv2.resize(border, (28,28))
                resized = resized.reshape(1, 28, 28, 1)
                prob = model.predict(resized, verbose = 0)
                final = np.argmax(prob)
                nums.append(str(final))
        return ''.join(nums)
    else:
        return 'Missing image'
    
def main():
    user = input('Image path: ')
    print(splitting_num(user))

if __name__ == '__main__':
    main()
