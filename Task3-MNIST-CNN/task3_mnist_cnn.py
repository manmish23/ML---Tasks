print("Task 3: Handwritten Character Recognition Started...")

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. LOAD MNIST DATASET - built into tensorflow
print("\nLoading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Training data shape: {X_train.shape}") # (60000, 28, 28)
print(f"Test data shape: {X_test.shape}") # (10000, 28, 28)

# 2. VISUALIZE SAMPLE DIGITS
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle('Sample MNIST Digits')
plt.show()

# 3. PREPROCESS DATA
# Reshape for CNN: (samples, height, width, channels)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# Normalize pixel values 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels: 5 → [0,0,0,0,0,1,0,0]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. BUILD CNN MODEL
model = models.Sequential([
    # Conv Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Block 3
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # Prevent overfitting
    layers.Dense(10, activation='softmax') # 10 classes: 0-9
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5. TRAIN MODEL
print("\nTraining CNN model...")
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    batch_size=64, 
                    validation_data=(X_test, y_test),
                    verbose=1)

# 6. EVALUATE MODEL
print("\n--- Evaluation ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 7. CLASSIFICATION REPORT
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# 8. CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - MNIST Digit Recognition')
plt.show()

# 9. PLOT TRAINING HISTORY
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.show()

# 10. TEST ON RANDOM IMAGES
plt.figure(figsize=(12, 6))
for i in range(10):
    idx = np.random.randint(0, X_test.shape[0])
    img = X_test[idx]
    pred = np.argmax(model.predict(img.reshape(1, 28, 28, 1)))
    true = np.argmax(y_test[idx])
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"True: {true}, Pred: {pred}", color='green' if pred==true else 'red')
    plt.axis('off')
plt.suptitle('Model Predictions on Test Images')
plt.show()

# 11. SAVE MODEL
model.save('mnist_cnn_model.h5')
print("\nModel saved as mnist_cnn_model.h5")
print("Task 3 Complete ✅")