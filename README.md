## ✊📄✂️ Rock-Paper-Scissors Image Classifier

This project trains a Convolutional Neural Network (CNN) using TensorFlow to classify images of hand gestures representing **rock**, **paper**, and **scissors**. The dataset is sourced from the official [TensorFlow rock-paper-scissors dataset](https://storage.googleapis.com/download.tensorflow.org/data/rps.zip).

---

## 🧠 Model Overview

- Input size: `150x150x3` (RGB images)
- Layers:
  - 4 Convolution + MaxPooling layers
  - Dropout for regularization
  - Dense layers ending with `softmax` for 3-class classification
- Data augmentation:
  - Random flip, rotation, translation, zoom, contrast

---


## 🧪 Predicting Custom Images
You can test the trained model with your own hand gesture images:
predict_image("path_to_image.jpg")

## 📊 Training Results
Training and validation accuracy/loss is plotted using matplotlib. This helps evaluate the effect of augmentation and generalization.

## 📦 Technologies Used

TensorFlow 2.x

TensorFlow Datasets

Python 3.x

Matplotlib & NumPy

## 📌 Notes
The model uses categorical_crossentropy as it’s a 3-class classification problem.

Labels are in categorical mode (one-hot encoded).

RandomContrast and RandomZoom improve robustness to lighting and scale.

## 👩‍💻 Author
This project was developed as part of a TensorFlow learning course. It demonstrates practical application of CNNs and data augmentation for image classification.
