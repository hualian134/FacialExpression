import numpy as np
import pandas as pd
import tensorflow as tf
from accuary import show_accuary
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Load the FER-2013 dataset
data = pd.read_csv('Data/fer2013/fer2013.csv')

# Preprocess the data
pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))
X = np.array(pixels.tolist(), dtype='float32').reshape(-1, 48, 48, 1)  # Reshape to 48x48 grayscale images
X = X / 255.0  # Normalize pixel values to [0, 1]

# Convert labels to one-hot encoding
y = to_categorical(data['emotion'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 output classes for emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test)
)


model.save('models/my_model.keras')

show_accuary(history)

