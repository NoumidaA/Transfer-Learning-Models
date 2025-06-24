# -*- coding: utf-8 -*-
"""multilabel_res50.ipynb

Original file is located at
    https://colab.research.google.com/drive/10n4pVc_xb8LHdR6GjZkwPe5FUFgqbdov
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import PIL
import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.backend import clear_session
from keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers
from keras.models import load_model
from tensorflow.keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import glorot_uniform
from keras.utils import np_utils
from tensorflow.keras.utils import plot_model
from __future__ import print_function
from keras.layers import *
from keras.models import Model

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/MULTILABEL _PAPER_DATA/transfer_learning

os.listdir('./train_data')

os.listdir('./validation_data')

# Check the number of images in training, validation and evaluation dataset
train = []
valid = []
test = []
# os.listdir returns the list of files in the folder, in this case image class names
for i in os.listdir('./train_data'):
   train.extend(os.listdir(os.path.join('train_data',i)))
   valid.extend(os.listdir(os.path.join('validation_data',i)))
   #test.extend(os.listdir(os.path.join('testdata',i)))

# train

print('Number of train images: {} \nNumber of validation images: {} \nNumber of test images: {}'.format(len(train),len(valid),len(test)))

# check the number of images in each class in the training dataset

No_images_per_class = []
Class_name = []
for i in os.listdir('./train_data'):
  Class_name.append(i)
  train_class = os.listdir(os.path.join('train_data',i))
  print('Number of images in {}={}\n'.format(i,len(train_class)))
  No_images_per_class.append(len(train_class))

train_datagen = ImageDataGenerator(rescale=1./255)
#normalize the data.
test_datagen = ImageDataGenerator(rescale=1./255)
# Create data generator for training, validation dataset.
train_generator = train_datagen.flow_from_directory(
        'train_data',
        target_size=(256,256),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
       'validation_data',
        target_size=(256,256),
        batch_size=32,
        class_mode='categorical')

# Image Dimensions
INPUT_SHAPE = (256,256, 3)
DROPOUT_RATE = 0.5
def create_model():
    model = Sequential()
    model.add(InputLayer(INPUT_SHAPE))
    model.add(ResNet50(weights='imagenet', include_top=False))
    model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(DROPOUT_RATE))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(10, activation='softmax'))

    return model

clear_session()

model = create_model()
model.summary()
adam = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="weightsres50.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(train_generator, steps_per_epoch= train_generator.n //32, epochs = 30, validation_data= validation_generator, validation_steps= validation_generator.n // 32,  callbacks=[checkpointer])

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/MULTILABEL _PAPER_DATA/transfer_learning

#2 birds
import glob
from keras.preprocessing import image
import cv2
model.load_weights('weightsres50.hdf5')
test_path=r'/content/drive/MyDrive/MULTILABEL _PAPER_DATA/SAM_CNN/2_species/spec_2'
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
original=[]
pred_label = []
def labels_test(test_path):                        #run for new_model
    '''Read and returns the image'''

    print("Reading...")
    count=0

    original=[]
    pred_label = []
    folders=os.listdir(test_path)
    #print(folders)
    for i in range(len(folders)):
        #file_label=i
        #print('bird: '+folders[i])
       # print('file_label: '+str(i))
        files=os.path.join(test_path,folders[i])
        spect=os.listdir(files)
      #  print(spect)

        for birds in glob.glob(test_path+'/'+folders[i]+'/*'):

                pred2=np.full((10,1), 0)
                #img = cv2.imread(os.path.join(test_path,folders[i],birds), cv2.IMREAD_COLOR)
               # img = cv2.resize(img,(400,400))
                img = image.load_img(os.path.join(test_path,folders[i],birds), target_size=(256,256))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
               # images = np.vstack([x])
                img = x.astype('float32')
                img = img / 255.0
                predict_value = model.predict(img)
                predict=predict_value .T
               # print(predict)

                pred2=pred2+predict
                pred_out=pred2*1./np.max( pred2, axis=0)

                #pred2 /= predict.shape[0]
        # print(pred_out)

        pred_final1 = np.argmax(pred_out)
       # print(pred_final1)
       # print(type(pred_final1))
        pred_final2=np.argsort(np.max(pred_out, axis=1))[-2]
      #  print(pred_final2)
       # print(type(pred_final2))
        pf1=[]
        pf2=[]
        pf1.append(pred_final1)
        pf2.append(pred_final2)
        pred_label_ad=pf1+pf2
       # print(pred_label_add)
        pred_label_add=sorted(pred_label_ad)


        pred_label.append(pred_label_add)
     #   print(pred_label)
        p = label = birds.split(os.path.sep)[-2].split("_")
        print(p)
        del p[-1]

        l=[]
        labels = {'Asiankoel': 0 , 'bluejay': 1, 'crow': 2,'duck': 3, 'goaway':4,'lapwing': 5,'owl': 6,'peafowl': 7,'sparrow': 8,'woodpeewe': 9}
        for num in range(len(p)):
           for keys,values in labels.items():
              if p[num]==keys:
                #  print(labels[keys])
                 l.append(labels[keys])
                 v=sorted(l)

        # print(l)
        original.append(v)

       # print(original)
               # predict = np.argmax(predict)
              #  print(predict)
        #  print(pred_label)


    return original,pred_label
original,pred_label=labels_test(test_path)
print(original,pred_label)
count=0
for i in range(len(original)):
    if (original[i]== pred_label[i]):
       count=count+1

total_correctly_predicted = print(count)
def Extract(lst):
    return [item[0] for item in lst]
y_ts=Extract(original)
#print(y_ts)
def Extract1(lst):
    return [item[1] for item in lst]
y_ts2=Extract1(original)
#print(y_ts2)
y_ts.extend(y_ts2)
print(y_ts)
def Extract3(lst):
    return [item[0] for item in lst]
y_pred=Extract3(pred_label)
#print(y_pred)
def Extract4(lst):
    return [item[1] for item in lst]
y_pred2=Extract4(pred_label)
#print(y_pred2)
y_pred.extend(y_pred2)
print(y_pred)
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
cm=confusion_matrix(y_ts,y_pred)
sn.set(font_scale=1.4)
plt.figure(figsize=(15,10))# for label size
sn.heatmap(cm, annot=True,cmap='magma') # font size
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
precision_recall_fscore_support(y_ts, y_pred, average='micro') #precision,recall,f1score
target_names=['Asiankoel','bluejay','crow','duck','goaway','lapwing','owl','peafowl','sparrow','woodpeewe']
dict=classification_report(y_ts, y_pred, target_names=target_names, digits=4,output_dict=True)
table=pd.DataFrame(dict)
table

#3 birds
import glob
from keras.preprocessing import image
import cv2
model.load_weights('weightsres50.hdf5')
test_path=r'/content/drive/MyDrive/MULTILABEL _PAPER_DATA/SAM_CNN/3_species/spec_3'
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
original=[]
pred_label = []


def labels_test(test_path):                        #run for new_model
    '''Read and returns the image'''

    print("Reading...")
    count=0

    original=[]
    pred_label = []
    folders=os.listdir(test_path)
    #print(folders)
    for i in range(len(folders)):
        #file_label=i
        #print('bird: '+folders[i])
       # print('file_label: '+str(i))
        files=os.path.join(test_path,folders[i])
        spect=os.listdir(files)
      #  print(spect)

        for birds in glob.glob(test_path+'/'+folders[i]+'/*'):

                pred2=np.full((10,1), 0)
                #img = cv2.imread(os.path.join(test_path,folders[i],birds), cv2.IMREAD_COLOR)
               # img = cv2.resize(img,(400,400))
                img = image.load_img(os.path.join(test_path,folders[i],birds), target_size=(256,256))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
               # images = np.vstack([x])
                img = x.astype('float32')
                img = img / 255.0
                predict_value = model.predict(img)
                predict=predict_value .T
               # print(predict)

                pred2=pred2+predict
                pred_out=pred2*1./np.max( pred2, axis=0)

                #pred2 /= predict.shape[0]
        # print(pred_out)

        pred_final1 = np.argmax(pred_out)
       # print(pred_final1)
       # print(type(pred_final1))
        pred_final2=np.argsort(np.max(pred_out, axis=1))[-2]
        pred_final3=np.argsort(np.max(pred_out, axis=1))[-3]
      #  print(pred_final2)
       # print(type(pred_final2))
        pf1=[]
        pf2=[]
        pf3=[]
        pf1.append(pred_final1)
        pf2.append(pred_final2)
        pf3.append(pred_final3)
        pred_label_ad=pf1+pf2+pf3
       # print(pred_label_add)
        pred_label_add=sorted(pred_label_ad)


        pred_label.append(pred_label_add)
     #   print(pred_label)
        p = label = birds.split(os.path.sep)[-2].split("_")
        print(p)
        del p[-1]

        l=[]
        labels = {'Asiankoel': 0 , 'bluejay': 1, 'crow': 2,'duck': 3, 'goaway':4,'lapwing': 5,'owl': 6,'peafowl': 7,'sparrow': 8,'woodpeewe': 9}
        for num in range(len(p)):
           for keys,values in labels.items():
              if p[num]==keys:
                #  print(labels[keys])
                 l.append(labels[keys])
                 v=sorted(l)

        # print(l)
        original.append(v)

       # print(original)
               # predict = np.argmax(predict)
              #  print(predict)
        #  print(pred_label)


    return original,pred_label
original,pred_label=labels_test(test_path)
print(original,pred_label)
count=0
for i in range(len(original)):
    if (original[i]== pred_label[i]):
       count=count+1

total_correctly_predicted = print(count)
print(len(original))




