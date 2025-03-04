import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import tensorflow as tf #type: ignore
from keras.models import Sequential #type: ignore
from keras.layers import Dense, Input #type: ignore
from keras.utils import to_categorical #type: ignore
from keras.datasets import mnist #type: ignore
def Neural_network_keras():
    """
    Builds, trains, and evaluates a neural network using Keras and the MNIST dataset.  
    The model consists of a hidden layer with 512 neurons (ReLU) and an output layer with 10 neurons (Softmax).  
    Data is normalized, reshaped, and trained for 10 epochs before evaluation.  
    """
    # Training data
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data() # Data: input data, and labels: used to calculate the loss function

    # Information about the training data
    print(train_data_x.shape)
    print(train_labels_y[1])
    print(test_data_x.shape)
    plt.imshow(train_data_x[1])
    plt.show()

    # Neural network architecture using TensorFlow and Keras
    model = Sequential([
        Input(shape = (28*28,)),
        Dense(512, activation = 'relu'), # number of neurons
        Dense(10, activation = 'softmax')
        ])

    # Compile the model
    model.compile(
        optimizer = 'rmsprop',
        loss = 'categorical_crossentropy', #type of loss function
        metrics = ['accuracy']
        )

    # Resume of model
    model.summary()

    # Normalizaxion de nuestros datos
    x_train = train_data_x.reshape(60000, 28*28)
    x_train = x_train.astype('float32')/255 # generate of normality
    y_train = to_categorical(train_labels_y)

    # Testing data preprocessing
    x_test = test_data_x.reshape(10000, 28*28)
    x_test = x_test.astype('float32')/255 # generate of normality
    y_test = to_categorical(test_labels_y)

    # Train the model
    model.fit(x_train, y_train, epochs = 10, batch_size = 128) # generate the train five, or five cicles, and the batch is for how pase the imagenes

    # Evaluate of model, the Neural Network
    model.evaluate(x_test,y_test) #evaluate of model utilice date of test