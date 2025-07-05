import os
import random
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_datasets as tfds


train_url = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
train_zip_path = tf.keras.utils.get_file("rps.zip", origin=train_url, extract=False)
train_dir = os.path.join(os.path.dirname(train_zip_path), "rps")
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(train_dir))

val_url = "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip"
val_zip_path = tf.keras.utils.get_file("rps-test-set.zip", origin=val_url, extract=False)
val_dir = os.path.join(os.path.dirname(val_zip_path), "rps-test-set")
with zipfile.ZipFile(val_zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(val_dir))

rock_dir = os.path.join(train_dir, 'rock')
paper_dir = os.path.join(train_dir, 'paper')
scissors_dir = os.path.join(train_dir, 'scissors')

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(150,150,3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='categorical'
    )

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='categorical'
    )

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = (train_dataset
                       .cache()
                       .shuffle(SHUFFLE_BUFFER_SIZE)
                       .prefetch(PREFETCH_BUFFER_SIZE)
                       )

validation_dataset_final = (validation_dataset
                            .cache()
                            .prefetch(PREFETCH_BUFFER_SIZE)
                            )

data_augmentation = tf.keras.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.4),
    tf.keras.layers.RandomTranslation(0.2,0.2),
    tf.keras.layers.RandomContrast(0.4),
    tf.keras.layers.RandomZoom(0.2)
    ])

model_with_aug = tf.keras.models.Sequential([
    data_augmentation,
    model
])

model_with_aug.compile(loss = 'categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['accuracy'])

history = model_with_aug.fit(
    train_dataset_final,
    epochs=25,
    validation_data = validation_dataset_final,
    verbose = 1
)

def plot_loss_acc(history):
  '''Plots the training and validation loss and accuracy from a history object'''
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  fig, ax = plt.subplots(1,2, figsize=(12, 6))
  ax[0].plot(epochs, acc, 'bo', label='Training accuracy')
  ax[0].plot(epochs, val_acc, 'b', label='Validation accuracy')
  ax[0].set_title('Training and validation accuracy')
  ax[0].set_xlabel('epochs')
  ax[0].set_ylabel('accuracy')
  ax[0].legend()

  ax[1].plot(epochs, loss, 'bo', label='Training Loss')
  ax[1].plot(epochs, val_loss, 'b', label='Validation Loss')
  ax[1].set_title('Training and validation loss')
  ax[1].set_xlabel('epochs')
  ax[1].set_ylabel('loss')
  ax[1].legend()

  plt.show()

plot_loss_acc(history)

def predict_image(img_path):
    # Load and preprocess the image
    image = tf.keras.utils.load_img(img_path, target_size=(150,150))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # (1, 150, 150, 3)

    prediction = model.predict(image_array, verbose=0)[0]

    print(f'\nmodel output: {prediction}')
    prediction_index = np.argmax(prediction)
    classes = ["paper", "rock", "scissors"]
    predicted_class = classes[prediction_index]
    print(f'{img_path} is {predicted_class}')
    plt.imshow(image)
    plt.show()

predict_image("rock.png")
predict_image("paper.png")
predict_image("scissors.png")