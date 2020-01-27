import pandas as pd
import numpy as np
import keras
import glob
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from mlxtend.preprocessing import minmax_scaling
from sklearn.metrics import roc_curve, auc

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Input, BatchNormalization, Multiply, Activation
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras import backend as K

import os

#Confusion Matrix
def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix',fontsize=15)
    plt.colorbar()
    classes = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito'] 
    plt.xticks([0,1,2,3,4,5,6,7,8,9], classes, fontsize=10)
    plt.yticks([0,1,2,3,4,5,6,7,8,9], classes, fontsize=10,rotation=90,verticalalignment="center")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > np.max(cm)/2. else "black")
    plt.xlabel('Predicted label',fontsize=15)
    plt.ylabel('True label',fontsize=15)
    
def plot_roc(fpr,tpr,roc_auc):
    plt.figure(figsize=(15,10))
    plt.plot(fpr[0], tpr[0], color='C1', lw=3, label='ROC curve of apple_pie (AUC = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='C2', lw=3, label='ROC curve of baby_pork_ribs (AUC = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='C3', lw=3, label='ROC curve of baklava (AUC = %0.2f)' % roc_auc[2])
    plt.plot(fpr[3], tpr[3], color='C4', lw=3, label='ROC curve of beef_carpaccio (AUC = %0.2f)' % roc_auc[3])
    plt.plot(fpr[4], tpr[4], color='C5', lw=3, label='ROC curve of beef_tartare (AUC = %0.2f)' % roc_auc[4])
    plt.plot(fpr[5], tpr[5], color='C6', lw=3, label='ROC curve of beet_salad (AUC = %0.2f)' % roc_auc[5])
    plt.plot(fpr[6], tpr[6], color='C7', lw=3, label='ROC curve of beignets (AUC = %0.2f)' % roc_auc[6])
    plt.plot(fpr[7], tpr[7], color='C8', lw=3, label='ROC curve of bibimbap (AUC = %0.2f)' % roc_auc[7])
    plt.plot(fpr[8], tpr[8], color='C9', lw=3, label='ROC curve of bread_pudding (AUC = %0.2f)' % roc_auc[8])
    plt.plot(fpr[9], tpr[9], color='C10', lw=3, label='ROC curve of breakfast_burrito (AUC = %0.2f)' % roc_auc[9])

    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--',alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('Receiver Operating Characteristics Curve',fontsize=30)
    plt.legend(loc="lower right",fontsize=15)
    plt.show()

# find file paths
food = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito'] 
f_apple = glob.glob('food-101/train/'+food[0]+'/*')
f_baby = glob.glob('food-101/train/'+food[1]+'/*')
f_baklava = glob.glob('food-101/train/'+food[2]+'/*')
f_beef_carpaccio = glob.glob('food-101/train/'+food[3]+'/*')
f_beef_tartare = glob.glob('food-101/train/'+food[4]+'/*')
f_beet_salad = glob.glob('food-101/train/'+food[5]+'/*')
f_beignets = glob.glob('food-101/train/'+food[6]+'/*')
f_bibimbap = glob.glob('food-101/train/'+food[7]+'/*')
f_bread_pudding = glob.glob('food-101/train/'+food[8]+'/*')
f_breakfast_burrito = glob.glob('food-101/train/'+food[9]+'/*')


f_apple_test = glob.glob('food-101/test/'+food[0]+'/*')
f_baby_test = glob.glob('food-101/test/'+food[1]+'/*')
f_baklava_test = glob.glob('food-101/test/'+food[2]+'/*')
f_beef_carpaccio_test = glob.glob('food-101/test/'+food[3]+'/*')
f_beef_tartare_test = glob.glob('food-101/test/'+food[4]+'/*')
f_beet_salad_test = glob.glob('food-101/test/'+food[5]+'/*')
f_beignets_test = glob.glob('food-101/test/'+food[6]+'/*')
f_bibimbap_test = glob.glob('food-101/test/'+food[7]+'/*')
f_bread_pudding_test = glob.glob('food-101/test/'+food[8]+'/*')
f_breakfast_burrito_test = glob.glob('food-101/test/'+food[9]+'/*')
"""# total 1000 files for each category
print('Number of images per class:\n\t\ttrain\ttest \nApple_pie:\t{}\t{}\nBaby_pork_ribs:\t{}\t{}\nBaklava:\t{}\t{}'
      .format(len(f_apple),len(f_apple_test),len(f_baby),len(f_baby_test),len(f_baklava),len(f_baklava_test)))"""

"""n = 7
fig, axes = plt.subplots(3,n,figsize=(20,10))

for i in range(n):
    axes[0, i].imshow(plt.imread(f_apple[i]))
    axes[0, i].set_title('apple pie')
    axes[1, i].imshow(plt.imread(f_baby[i]))
    axes[1, i].set_title('baby back ribs')
    axes[2, i].imshow(plt.imread(f_baklava[i]))
    axes[2, i].set_title('baklava')
    

for i in range(len(f_apple)):
    h1,w1,c1 = plt.imread(f_apple[i]).shape
    h2,w2,c2 = plt.imread(f_baby[i]).shape
    h3,w3,c3 = plt.imread(f_baklava[i]).shape
    plt.scatter(h1,w1,c='r',marker='x',alpha=0.5)
    plt.scatter(h2,w2,c='c',marker='o',alpha=0.5)
    plt.scatter(h3,w3,c='b',marker='v',alpha=0.5)
plt.title('Image Size')
plt.legend(('apples','babies','baklavas'))"""

train_datagen = ImageDataGenerator(featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=5,
                 width_shift_range=0.05,
                 height_shift_range=0.05,
                 shear_range=0.2,
                 zoom_range=0.2,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=True,
                 vertical_flip=False,
                 rescale=1/255) #rescale to [0-1], add zoom range of 0.2x and horizontal flip
train_generator = train_datagen.flow_from_directory(
        "food-101/train",
        target_size=(224,224),
        batch_size=64)
test_datagen = ImageDataGenerator(rescale=1/255) # just rescale to [0-1] for testing set
test_generator = test_datagen.flow_from_directory(
        "food-101/test",
        target_size=(224,224),
        batch_size=64)

"""# preview images from train generator
r = 4; c = 7
n=0
classtolabel = {'0':'apple_pie','1':'baby_back_ribs','2':'baklava'}
for x in train_generator:
    fig, axes = plt.subplots(r,c,figsize=(20,12))
    for i in range(r):
        for j in range(c):
            axes[i,j].imshow(x[0][n])
            label = np.argmax(x[1],axis=1)[n].astype('str')
            axes[i,j].set_title(classtolabel[label])
            n+=1    
    break"""

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu', input_shape = (224,224,3), kernel_initializer='he_normal'))
model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation = "relu",kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "softmax",kernel_initializer='he_normal',kernel_regularizer=l2()))

#callbacks
checkpointer = ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='auto')

model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit_generator(train_generator,steps_per_epoch=2250/64,
                              validation_data=test_generator,validation_steps=750/64, 
                              epochs=1, callbacks=[checkpointer, reduceLR, earlystopping])

# create another generator for all test images in a single batch 
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = test_datagen.flow_from_directory(
        "food-101/test",
        target_size=(224,224),
        batch_size=750)

x_test, y_test = val_generator.next()
y_pred_conf = model.predict(x_test) #return probabilities of each class
y_pred = np.argmax(y_pred_conf,axis=1)
y_label = np.argmax(y_test,axis=1)

print('Accuracy score: {:.1f}%'.format(accuracy_score(y_pred,y_label)*100))

"""# Randomly check 5 predictions
ind = np.random.randint(1,len(x_test),5)
f, ax=plt.subplots(1,5,figsize=(20,10))
for i,j in enumerate(ind):
    ax[i].imshow(x_test[j])
    ax[i].set_title("Pred :{}({:.2f})\nTrue :{}({:.2f})".format
                          (classtolabel[str(y_pred[j])],np.max(y_pred_conf[j]),
                           classtolabel[str(y_label[j])],y_pred_conf[j][(y_label[j])],fontweight="bold", size=20))"""


plot_confusion_matrix(confusion_matrix(y_label,y_pred))

# ROC Curve

fpr = dict() # false positive rate
tpr = dict() # true positive rate
roc_auc = dict() # area under roc curve
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_conf[:, i]) # roc_curve function apply to binary class only
    roc_auc[i] = auc(fpr[i], tpr[i])  # using the trapezoidal rule to get area under curve
plot_roc(fpr,tpr,roc_auc)