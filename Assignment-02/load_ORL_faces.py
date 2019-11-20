import numpy as np  
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense 
from keras.models import Sequential
import matplotlib.pyplot as plt
# load data

def create_model(layer_sizes):
    model = Sequential()
    model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(image_size,)))
    for s in layer_sizes[1:]:
        model.add(Dense(units = s, activation = 'sigmoid'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def evaluate(model):
#    # classification using the test set
#    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#    model.fit(x_train, y_train, batch_size=512, epochs=20, validation_split=.1, verbose=False)
#    test_loss, test_accuracy  = model.evaluate(x_test, y_test, verbose=False)
#    print(f'Test loss: {test_loss:.3}')
#    print(f'Test accuracy: {test_accuracy:.3}')
        
    
    # classification using the training set
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=512, epochs=950, validation_split=.1, verbose=False)
    training_loss, training_accuracy  = model.evaluate(x_train, y_train, verbose=False)
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()
    print(f'Training loss: {training_loss:.3}')
    print(f'Training accuracy: {training_accuracy:.3}')
    





data = np.load('ORL_faces.npz')
x_train = data['trainX']
y_train = data['trainY']
x_test = data['testX']
y_test = data['testY']
 

image_size = 10304 
x_train = x_train.reshape(x_train.shape[0], image_size) 
x_test = x_test.reshape(x_test.shape[0], image_size)
x_train = x_train / 255
x_test = x_test / 255
x_train.astype(float)
x_test.astype(float)
y_train.astype(float)
y_test.astype(float)

# convert class vectors to binary class matrices
num_classes = 20
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = create_model([512])
evaluate(model)
#temptrainX= np.reshape(trainX,(240,92,112))

 