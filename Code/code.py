
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import pathlib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout, Flatten,Activation, BatchNormalization,MaxPooling2D
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
apple = list(data_dir.glob('apple fruit/*'))
banana = list(data_dir.glob('banana fruit/*'))
cherry = list(data_dir.glob('cherry fruit/*'))
chickoo = list(data_dir.glob('chickoo fruit/*'))
grapes = list(data_dir.glob('grapes fruit/*'))
orange = list(data_dir.glob('orange fruit/*'))
strawberry = list(data_dir.glob('strawberry fruit/*'))
fruit_images_dict = {
    'mango': list(data_dir.glob('mango fruit/*')),
    'kiwi': list(data_dir.glob('kiwi fruit/*')),
    'apple': list(data_dir.glob('apple fruit/*')),
    'banana': list(data_dir.glob('banana fruit/*')),
    'cherry': list(data_dir.glob('cherry fruit/*')),
    'chickoo': list(data_dir.glob('chickoo fruit/*')),
    'grapes': list(data_dir.glob('grapes fruit/*')),
    'orange': list(data_dir.glob('orange fruit/*')),
    'strawberry': list(data_dir.glob('strawberry fruit/*'))
}
fruit_labels_dict = {
    'mango': 0,
    'kiwi': 1,
    'apple': 2,
    'banana': 3,
    'cherry': 4,
    'chickoo': 5,
    'grapes': 6,
    'orange': 7,
    'strawberry': 8
}
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_CHANNELS = 3
X, Y = [], []

for fruit_name, images in fruit_images_dict.items():
    print(fruit_name)
    for image in images:
        img = cv2.imread(str(image))
        if isinstance(img,type(None)): 
            #print('image not found')
            continue
        elif ((img.shape[0] >= IMAGE_HEIGHT) and  (img.shape[1] >=IMAGE_WIDTH)):
            resized_img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
            X.append(resized_img)
            Y.append(fruit_labels_dict[fruit_name])
        else:
            #print("Invalid Image")
            Continue
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
print(len(X_train),len(Y_train))
print(len(X_test),len(Y_test))
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
IMAGE_CHANNELS=3
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),
Conv2D(64, (3, 3), activation='relu'),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),

Conv2D(128, (3, 3), activation='relu'),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),
Flatten(),
Dense(512, activation='relu'),
BatchNormalization(),
Dropout(0.5),
Dense(1, activation='softmax'), 
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
classes = ["mango","kiwi","apple","banana","cherry","chickoo","grapes","orange","strawberry"]
from PIL import Image
fileImage = Image.open(r"C:\Users\saksh\Downloads\kaggle\input\fruits\pictures\kiwi fruit\Image_1.png").convert("RGB").resize([IMAGE_WIDTH,IMAGE_HEIGHT],Image.LANCZOS)
image = np.array(fileImage)
myimage = image.reshape(1, IMAGE_WIDTH,IMAGE_HEIGHT,3)
# prepare pixel data
#myimage = myimage.astype('float32')
#myimage = myimage/255.
plt.figure(figsize = (4,2))
plt.imshow(image)

my_predicted_image = model.predict(myimage)
print(my_predicted_image)
if (my_predicted_image < 0.40):
    y_class=0
else:
    y_class=1
print("class:",y_class,"name=",classes[y_class])
import mysql.connector
mydb = mysql.connector.connect(
host="localhost",
user="root",
password=" ",
database="classify"
)
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM fruits where name=classes[yclass]")
myresult = mycursor.fetchall()
for x in myresult:
  print(x)

