# 🖼️ MNIST Image Classification using CNN

A Convolutional Neural Network (CNN) model built with TensorFlow/Keras to classify handwritten digits from the **MNIST dataset**. The model uses a Softmax output layer to predict digits from 0 to 9.

---

## 🧠 Model Architecture

The model is a Sequential CNN with the following layers:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')     # 10 classes (digits 0-9)
])
```

### Layer Breakdown

| Layer | Output Shape | Details |
|---|---|---|
| Input | (28, 28, 1) | Grayscale MNIST image |
| Conv2D | (26, 26, 32) | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | (13, 13, 32) | 2×2 pool size |
| Conv2D | (11, 11, 64) | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | (5, 5, 64) | 2×2 pool size |
| Flatten | (1600,) | Converts 2D → 1D |
| Dense | (128,) | 128 units, ReLU activation |
| Dense (Output) | (10,) | Softmax — 10 digit classes |

### Why These Design Choices?

- **Conv filter (3×3) + Pool (2×2)** — Keeps spatial dimensions reducing gradually, preserving enough features for deeper layers
- **ReLU activation** — Avoids vanishing gradient problem in hidden layers
- **Softmax output** — Converts raw scores into probabilities that sum to 1.0, ideal for multi-class classification
- **Two Conv+Pool blocks** — Extracts both low-level (edges) and high-level (shapes) features

---

## 📊 How Softmax Works

```
Raw output  →  [2.1,  0.5,  8.3,  1.2, ...]
Softmax     →  [0.02, 0.01, 0.95, 0.01, ...]   ← probabilities sum to 1.0
Prediction  →  Class 2 (highest probability = 95%)
```

```python
import numpy as np

prediction      = model.predict(img)
predicted_digit = np.argmax(prediction)     # gets index of highest probability
confidence      = prediction[0][predicted_digit] * 100

print(f"Predicted Digit : {predicted_digit}")
print(f"Confidence      : {confidence:.2f}%")
```

---

## 📦 Requirements

```
tensorflow
opencv-python
numpy
```

Install with:
```bash
pip install tensorflow opencv-python numpy
```

---

## 🚀 Quick Start

### 1. Load MNIST Dataset
```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test  = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)
```

### 2. Train the Model
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_test, y_test))
```

### 3. Predict from a Custom Image
```python
import cv2
import numpy as np

img = cv2.imread('your-image.webp')
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # convert BEFORE normalizing
img = img.astype('float32') / 255
img = img.reshape((1, 28, 28, 1))             # (batch, height, width, channels)

prediction      = model.predict(img)
predicted_digit = np.argmax(prediction)

print(f"Predicted Digit : {predicted_digit}")
print(f"Confidence      : {prediction[0][predicted_digit] * 100:.2f}%")
```

---

## 📈 Sample Output

```
Epoch 10/10 - accuracy: 0.9934 - val_accuracy: 0.9912

Predicted Probabilities : [0.00, 0.00, 0.95, 0.02, 0.00, 0.01, 0.00, 0.01, 0.00, 0.01]
Predicted Digit         : 2
Confidence              : 95.00%
```

---

## 🛠️ Built With

- [TensorFlow / Keras](https://www.tensorflow.org/) — Deep learning framework
- [OpenCV](https://opencv.org/) — Image preprocessing
- [NumPy](https://numpy.org/) — Numerical computation
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — Handwritten digit dataset

---

## 👤 Author

**Your Name**
- GitHub: [@VenkatakrishnanSambath](https://github.com/VenkatakrishnanSambath)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
