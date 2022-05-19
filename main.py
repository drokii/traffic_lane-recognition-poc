# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import cv2

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def main():
    preprocessData()
    model = createModel()
    compileLayers(model)
    trainModel(model)
    evaluateModelAccuracy(model)
    makePrediction(model)
    recognizePiccie(model)


# Scales the original images in the dataset from 0 to 1 for them to be fed to the neural network model.


def preprocessData():
    global train_images
    global test_images

    train_images = train_images / 255.0
    test_images = test_images / 255.0

# Plots the first 25 images in the data model.


def plotImages():
    global train_images
    global class_names

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

# Defines the layers to be used in this neural network, naemly Flatten, and Dense (twice).
# Flatten unstacks the pixels in the images into a straight line of (28x28) 784 pixels.


def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

# Defines the layers to be used in this neural network, naemly Flatten, and Dense (twice).
# Flatten unstacks the pixels in the images into a straight line of (28x28) 784 pixels.


def compileLayers(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

# Starts training the model on the training dataset


def trainModel(model):
    model.fit(train_images, train_labels, epochs=10)

# Evaluates how the model performs against the test dataset


def evaluateModelAccuracy(model):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)


def makePrediction(model):
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    print(class_names[np.argmax(predictions[0])])


def recognizePiccie(model):
    img = cv2.cvtColor(iio.imread("shirt.jpg"), cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    img = img / 255.0
    img = (np.expand_dims(img, 0))

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions_single = probability_model.predict(img)
    print(class_names[np.argmax(predictions_single[0])])

    


if __name__ == "__main__":
    main()
