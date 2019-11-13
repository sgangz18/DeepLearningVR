from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense 
from keras.models import Sequential
import matplotlib.pyplot as plt


def create_model(layer_sizes):
    model = Sequential()
    model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(image_size,)))
    for s in layer_sizes[1:]:
        model.add(Dense(units = s, activation = 'sigmoid'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def evaluate(model):
    # classification using the test set
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=.1, verbose=False)
    test_loss, test_accuracy  = model.evaluate(x_test, y_test, verbose=False)
    print(f'Test loss: {test_loss:.3}')
    print(f'Test accuracy: {test_accuracy:.3}')
    
    
    # classification using the training set
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=.1, verbose=False)
    training_loss, training_accuracy  = model.evaluate(x_train, y_train, verbose=False)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()
    print(f'Training loss: {training_loss:.3}')
    print(f'Training accuracy: {training_accuracy:.3}')
    
    
# load the mnist dataset and split into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = 784 # 28 x 28
x_train = x_train.reshape(x_train.shape[0], image_size) 
x_test = x_test.reshape(x_test.shape[0], image_size)

# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = create_model([128] * 2)
evaluate(model)