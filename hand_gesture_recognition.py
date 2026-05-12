# =============================
# Hand Gesture Recognition Model
# Internship Task-04
# ProDigy InfoTech
# =============================

# Import Libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =============================
# Dataset Path
# =============================

dataset_path = "leapGestRecog"

# =============================
# Load Images and Labels
# =============================

data = []
labels = []

IMG_SIZE = 64

for subject in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject)

    if os.path.isdir(subject_path):

        for gesture in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture)

            if os.path.isdir(gesture_path):

                for img_name in os.listdir(gesture_path):

                    img_path = os.path.join(gesture_path, img_name)

                    img = cv2.imread(img_path)

                    if img is not None:

                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        data.append(img)
                        labels.append(gesture)

# Convert to NumPy Arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# =============================
# Encode Labels
# =============================

encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

labels_categorical = to_categorical(labels_encoded)

# =============================
# Train-Test Split
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels_categorical,
    test_size=0.2,
    random_state=42
)

# =============================
# Build CNN Model
# =============================

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(labels_categorical[0]), activation='softmax'))

# =============================
# Compile Model
# =============================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =============================
# Train Model
# =============================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =============================
# Evaluate Model
# =============================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# =============================
# Save Model
# =============================

model.save("hand_gesture_model.h5")

print("Model Saved Successfully!")

# =============================
# Plot Accuracy Graph
# =============================

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()