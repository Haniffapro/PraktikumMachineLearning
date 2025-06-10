# 2. Import TensorFlow
import tensorflow as tf

# 3. Import Modul dari TensorFlow dan Keras
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Activations
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.metrics as metrics
import tensorflow.keras.utils as utils
import cv2


# 4. Fungsi Tambahan untuk Visualisasi dan Arsitektur Model
from tensorflow.keras.utils import model_to_dot
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model

# 5. Visualisasi Data
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec

# 6. Manipulasi Data Numerik
import numpy as np

# 7. Evaluasi dan Pembagian Data
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as cm

# 8. Utilitas tambahan
from IPython.display import display_svg
import os

img_size = 200

def get_images(directory):
    Images = []
    Labels = []
    label = 0

    for labels in os.listdir(directory):
        if labels == 'dogs':
            label = 0
        elif labels == 'cats':
            label = 1

        for image_file in os.listdir(directory + labels):
            image = cv2.imread(directory + labels + '/' + image_file)
            image = cv2.resize(image, (img_size, img_size))
            Images.append(image)
            Labels.append(label)

    return shuffle(Images, Labels, random_state=817328462)

def get_classlabel(class_code):
    labels = {0: 'dogs', 1: 'cats'}
    return labels[class_code]

Images, Labels = get_images('/content/dog vs cat/dataset/training_set/')
Images = np.array(Images)
Labels = np.array(Labels)

print("Ukuran gambar dan jumlah", Images.shape)
print("Labels", Labels.shape)

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


# Pastikan img_size sudah didefinisikan, contoh: img_size = 200, Membuat model
model = models.Sequential([
    # Blok 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'), # Tambah 1 Conv layer
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Blok 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'), # Tambah 1 Conv layer
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Klasifikasi
    layers.Flatten(),
    layers.Dense(256, activation='relu'), # Tingkatkan neuron di Dense layer
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

# Compile model dengan loss yang sesuai
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy", # Cocok dengan Dense(2, 'softmax')
    metrics=["accuracy"]
)

model.summary()

from livelossplot.keras import PlotLossesCallback
import efficientnet.keras as efn
from keras.callbacks import CSVLogger

# Menyimpan nama file untuk ringkasan model dan log pelatihan
MODEL_SUMMARY_FILE = "model_summary.txt"
TRAINING_LOGS_FILE = "training_logs.csv"

# Menyimpan struktur model ke dalam file .txt
with open(MODEL_SUMMARY_FILE, 'w') as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

# Melatih model dengan logging dan visualisasi
trained = model.fit(
    Images, Labels,
    epochs=10,
    validation_split=0.18,
    batch_size=32,
    callbacks=[
        PlotLossesCallback(),
        CSVLogger(TRAINING_LOGS_FILE, append=False, separator=";")
    ]
)

test_images, test_labels = get_images("/content/dataset/test_set/")
test_images = np.array(test_images)
test_labels = np.array(test_labels)

model.evaluate(test_images, test_labels, verbose=1)

pred_images, no_labels = get_images("/content/dataset/test_set/")
pred_images = np.array(pred_images)
pred_images.shape

#Predict
from random import randint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fig =  plt.figure(figsize=(30,30))
outer = gridspec.GridSpec(5,5, wspace=0.2, hspace=0.2)
for i in range(20):
    inner = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=outer[i], wspace=0.2,hspace=0.2)
    rnd_number = randint(0,len(pred_images))
    pred_image = np.array([pred_images[rnd_number]])
    pred_class = get_classlabel(np.argmax(model.predict(pred_image), axis=1)[0])
    pred_prob = model.predict(pred_image).reshape(2)
    for j in range(2):
        if(j%2) == 0:
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(pred_image[0])
            ax.set_title(pred_class)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plt.Subplot(fig, inner[j])
            ax.bar([0,1],pred_prob)
            fig.add_subplot(ax)
fig.show()
