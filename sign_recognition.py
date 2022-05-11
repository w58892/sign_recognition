import os
import pandas as pd
import imageio
import math
import numpy as np
import cv2
import sklearn
import time
import keras
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

### Otwarcie katalogu ze zdjęciami
data_dir = os.path.abspath('GTSRB/Final_Training/Images')
os.path.exists(data_dir)

### funkcja zmiany rozmiaru zdjęć do rozmiaru 64px x 64px używając open cv
def resize_cv(im):
    return cv2.resize(im, (64, 64), interpolation = cv2.INTER_LINEAR)

### Wczytanie bazy 39209 zdjęć
list_images = []
output = []
for dir in os.listdir(data_dir):
    if dir == '.DS_Store' :
        continue
    
    inner_dir = os.path.join(data_dir, dir)
    csv_file = pd.read_csv(os.path.join(inner_dir,"GT-" + dir + '.csv'), sep=';')
    for row in csv_file.iterrows() :
        img_path = os.path.join(inner_dir, row[1].Filename)
        img = imageio.imread(img_path)
        img = img[row[1]['Roi.X1']:row[1]['Roi.X2'],row[1]['Roi.Y1']:row[1]['Roi.Y2'],:]
        img = resize_cv(img)
        list_images.append(img)
        output.append(row[1].ClassId)



input_array = np.stack(list_images)

train_y = keras.utils.np_utils.to_categorical(output)

### przetasowanie zbioru
randomize = np.arange(len(input_array))
np.random.shuffle(randomize)
x = input_array[randomize]
y = train_y[randomize]

### podział zbioru na  treningowy, walidacyjny i testowy
split_size = int(x.shape[0]*0.6)
train_x, val_x = x[:split_size], x[split_size:] # zbiór treningowy zawiera 23525 zdjęć
train1_y, val_y = y[:split_size], y[split_size:]

split_size = int(val_x.shape[0]*0.5)
val_x, test_x = val_x[:split_size], val_x[split_size:] # zbiór testowy zawiera 7842 zdjęć
val_y, test_y = val_y[:split_size], val_y[split_size:]


### CNN
### budowanie modelu
hidden_num_units = 2048
hidden_num_units1 = 1024
hidden_num_units2 = 128
output_num_units = 43

epochs = 10
batch_size = 16
pool_size = (2, 2)
input_shape = Input(shape=(32, 32,3))

model = Sequential([

 Conv2D(16, (3, 3), activation='relu', input_shape=(64,64,3), padding='same'),
 BatchNormalization(),

 Conv2D(16, (3, 3), activation='relu', padding='same'),
 BatchNormalization(),
 MaxPooling2D(pool_size=pool_size),
 Dropout(0.2),
    
 Conv2D(32, (3, 3), activation='relu', padding='same'),
 BatchNormalization(),
    
 Conv2D(32, (3, 3), activation='relu', padding='same'),
 BatchNormalization(),
 MaxPooling2D(pool_size=pool_size),
 Dropout(0.2),
    
 Conv2D(64, (3, 3), activation='relu', padding='same'),
 BatchNormalization(),
    
 Conv2D(64, (3, 3), activation='relu', padding='same'),
 BatchNormalization(),
 MaxPooling2D(pool_size=pool_size),
 Dropout(0.2),

 Flatten(),

 Dense(units=hidden_num_units, activation='relu'),
 Dropout(0.3),
 Dense(units=hidden_num_units1, activation='relu'),
 Dropout(0.3),
 Dense(units=hidden_num_units2, activation='relu'),
 Dropout(0.3),
 Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

### Trenowanie modelu
trained_model_conv = model.fit(train_x.reshape(-1,64,64,3), train1_y, epochs=epochs, batch_size=batch_size, validation_data=(val_x, val_y))                               

start1 = time.process_time()

### Predykcja klas
pred = model.predict(test_x)
np.argmax(pred)

### Ocena modelu
model.evaluate(test_x, test_y)

end1 = time.process_time()

#dokładność CNN ~99%
print("czas klasyfikacji CNN") # ~10s
print(end1 - start1)
print("")


### k-NN
### zmiana kształtu macierzy z 3 do 2 wymiarów
train1_x = (train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2] * train_x.shape[3]))
test1_x = (test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2] * test_x.shape[3]))

### Trenowanie k-NN
clf = KNeighborsClassifier(p=1)
clf.fit(train1_x, train1_y)

start2 = time.process_time()

### Predykcja klas
predictions = clf.predict(test1_x)

### Mierzenie dokładności
knn_accuracy = metrics.accuracy_score(predictions, test_y)

end2 = time.process_time()

print("dokladność k-NN") # ~85%
print(knn_accuracy)
print("czas klasyfikacji k-NN") # ~1205s
print(end2 - start2)

