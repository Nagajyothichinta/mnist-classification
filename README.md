# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model


<img width="386" alt="image" src="https://github.com/Nagajyothichinta/mnist-classification/assets/94191344/0ebb09e5-c9e0-4f53-8e4c-721e079240bd">

## DESIGN STEPS

Step 1:
Start by importing all the necessary libraries. And load the Data into Test sets and Training sets.

Step 2:
Then we move to normalization and encoding of the data.

Step 3:
The Model is then built using a Conv2D layer, MaxPool2D layer, Flatten layer, and 2 Dense layers of 16 and 10 neurons respectively.

Step 4:
The necessary Validating parameters are visualized for inspection.

Step 5:
Finally, we pass handwritten digits to the model for prediction.


## PROGRAM
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[1500]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='tanh'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=64, validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/9.png')

type(img)

img = image.load_img('l1.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="425" alt="image" src="https://github.com/Nagajyothichinta/mnist-classification/assets/94191344/7311803d-a678-4c03-8a09-796b6407c403">

<img width="428" alt="image" src="https://github.com/Nagajyothichinta/mnist-classification/assets/94191344/2cd43fd8-65d5-49f1-8649-f832c7861e62">


### Classification Report

<img width="347" alt="image" src="https://github.com/Nagajyothichinta/mnist-classification/assets/94191344/22673ba3-c85c-4538-83d2-8f0ad2a33665">


### Confusion Matrix

<img width="353" alt="image" src="https://github.com/Nagajyothichinta/mnist-classification/assets/94191344/bf1c0e4a-1f42-47b2-bf0f-47d3fa2e2331">


### New Sample Data Prediction

<img width="211" alt="image" src="https://github.com/Nagajyothichinta/mnist-classification/assets/94191344/69c3f261-1b94-4540-afd9-763666ca9795">

<img width="365" alt="image" src="https://github.com/Nagajyothichinta/mnist-classification/assets/94191344/3ad78da4-19b1-42f1-b4be-60d27ebecb12">



## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
