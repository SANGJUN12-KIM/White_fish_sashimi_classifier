from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG16
import numpy as np
import os, glob, random
import pandas as pd
import cv2
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def load_data(image_file_dir, label):
    shuffled_image_data=[]
    shuffled_label_data = []
    output = []

    shuffled_train_label = label.iloc[np.random.permutation(label.index)].reset_index(drop=True)

    for i, row in shuffled_train_label.iterrows():
        image = cv2.imread(image_file_dir+row['file_name'])
        shuffled_image_data.append(image)
        shuffled_label_data.append(row['fish_name'])

    images = np.array(shuffled_image_data, dtype='float32')
    labels = pd.get_dummies(shuffled_label_data)

    return images, labels

train_files_dir = "./Dataset_proc/train/"
test_files_dir = "./Dataset_proc/valid/"
train_label = pd.read_csv("./Dataset_proc/trainLabel.csv", header=None, names=['file_name', 'fish_name'])
test_label = pd.read_csv("./Dataset_proc/validLabel.csv", header=None,  names=['file_name', 'fish_name'])

train_inputs, train_label = load_data(train_files_dir,train_label)

transfer_model = VGG16(weights= 'imagenet', include_top=False, input_shape=(128, 128, 3))
transfer_model.trainable = False


model = Sequential([
    transfer_model,
    #Flatten(),
    #GlobalAveragePooling2D(),
    #Conv2D(32, kernel_size=(3, 3), input_shape=(128, 128,3), activation='relu'),
    #Conv2D(64, kernel_size=(3,3), activation='relu'),
    #MaxPooling2D(pool_size=(2)),
    GlobalAveragePooling2D(),
    Dense(1024, activation= 'relu'),
    Dropout(0.25),
    Dense(512, activation= 'relu'),
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation= 'relu'),
    Dense(64, activation= 'relu'),
    Dense(8, activation= 'softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) # 평가 척도는 정확도입니다

model_file_name = "best.h5"
checkpointer = ModelCheckpoint(filepath=model_file_name,
                               monitor='val_accuracy',
                               verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience= 10)

history = model.fit(train_inputs, train_label,
          batch_size=50,
          epochs=100,
          validation_split = 0.2, callbacks=[checkpointer, early_stopping])


print("성능 검증")
test_inputs, test_label = load_data(test_files_dir,test_label)

loss_and_metrics = model.evaluate(test_inputs, test_label, batch_size=16)