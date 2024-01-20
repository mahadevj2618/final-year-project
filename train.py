import pathlib

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# Setting up the data directory and getting the count of images in the directory:

data_dir = pathlib.Path("C:/Users/Dell/Downloads/archive/Data_set")
image_count = len(list(data_dir.glob('*/*.PNG')))
# Defining batch size, image height, and image width:
batch_size = 32
img_height = 180
img_width = 180
# Printing the available utilities in tf.keras.utils:
print(tf.keras.utils)
# Loading the training dataset using image_dataset_from_directory function:
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# Loading the validation dataset using image_dataset_from_directory function:
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# Getting the class names from the training dataset:
class_names = train_ds.class_names
print(class_names)
# Setting up data prefetching and caching for better performance:
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# Defining the CNN model architecture using Sequential:
num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
# Compiling the model by specifying the optimizer, loss function, and evaluation metrics
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Printing a summary of the model architecture:
model.summary()
# Training the model using the training and validation datasets:
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
# Saving the trained model to a file:
model.save("detect_image.h5")