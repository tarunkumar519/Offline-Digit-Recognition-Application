import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data(path='mnist.npz')
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)

for x in range(len(xTrain)):
    for i in range(28):
        for j in range(28):
            if xTrain[x][i][j] > 0:
                xTrain[x][i][j] = 1
    print(x)

#model
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=20)

p = model.predict(xTest)
model.save('digitRecognizerModel.h5')
