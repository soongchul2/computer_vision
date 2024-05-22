import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()


x_train = x_train.reshape(60000, 28, 28, 1)  
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


cnn = Sequential([
    Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(16, (5, 5), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(120, (5, 5), padding='valid', activation='relu'),
    Flatten(),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])


cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


cnn.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)


score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])