import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
import os
from tqdm import tqdm

TRAIN_DIR=("D:/Projects/Data/images/train")
TEST_DIR=("D:/Projects/Data/images/validation")

def dataframe(dir):
    image_path=[]
    labels=[]
    for label in os.listdir(dir):
        for img in os.listdir(os.path.join(dir,label)):
            image_path.append(os.path.join(dir,label,img))
            labels.append(label)
        print(label,"completed")
    return image_path,labels

train=pd.DataFrame()
train['images'],train['label']=dataframe(TRAIN_DIR)

test=pd.DataFrame()
test['images'],test['label']=dataframe(TEST_DIR)

def extract_features(images):
    features = []
    for image in tqdm(images):
        try:
            img = load_img(image, color_mode='grayscale')
            img = np.array(img)
            features.append(img)
        except Exception as e:
            print(f"Error loading image {image}: {e}")
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

train_features=extract_features(train['images'])
test_features=extract_features(test['images'])

x_train=train_features/255.0
x_test=test_features/255.0

le=LabelEncoder()
le.fit(train['label'])

y_train=le.transform(train['label'])
y_test=le.transform(test['label'])

y_train=to_categorical(y_train,7)
y_test=to_categorical(y_test,7)

model=Sequential()

model.add(Conv2D(128,(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
model.fit(x=x_train,y=y_train,batch_size=128,epochs=60,validation_data=(x_test,y_test))

model.save('senti.h5')
