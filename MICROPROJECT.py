# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Create a generator without augmentation (only rescale)
no_aug_datagen = ImageDataGenerator(rescale=1./255)
# Flow from directory (only for visualization)
no_aug_generator = no_aug_datagen.flow_from_directory(
train_path,
target_size=(128, 128),
batch_size=10,
class_mode='binary'
)
# Function to visualize images BEFORE augmentation
def show_original_images(generator, n_images=5):
images, labels = next(generator) # take one batch
plt.figure(figsize=(12, 4))
for i in range(n_images):
plt.subplot(1, n_images, i+1)
plt.imshow(images[i])
label = "Edible" if labels[i] == 0 else "Poisonous"
plt.title(label)
plt.axis('off')
plt.show()
# Show original images from training set
show_original_images(no_aug_generator, n_images=5)
base_path = "/content/drive/MyDrive/DL/edible and poisonous mushroom"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "validation")
test_path = os.path.join(base_path, "test")
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=45, # bigger rotation
zoom_range=0.5, # zoom in/out more aggressively
width_shift_range=0.3, # shift more
height_shift_range=0.3,
horizontal_flip=True,
vertical_flip=True
)
# Validation & Test should not be augmented
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
train_path,
target_size=(128, 128), # resize all images
batch_size=32,
class_mode='binary' # edible vs poisonous
)
val_generator = val_datagen.flow_from_directory(
val_path,
target_size=(128, 128),
batch_size=32,
class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
test_path,
target_size=(128, 128),
batch_size=32,
class_mode='binary',
shuffle=False # important for evaluation
)
# Function to show augmented images for a specific class
def show_class_augmented(generator, class_name, n_images=8):
class_idx = generator.class_indices[class_name] # get edible/poisonous index
plt.figure(figsize=(14, 6))
count = 0
for images, labels in generator:
# pick images that belong only to the requested class
for i in range(len(images)):
if labels[i] == class_idx:
plt.subplot(2, 4, count+1)
plt.imshow(images[i])
plt.title(class_name)
plt.axis("off")
count += 1
if count == n_images:
plt.show()
return
show_class_augmented(train_generator, "edible mushroom", n_images=8)
show_class_augmented(train_generator, "poisonous mushroom", n_images=8)
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
model = models.Sequential([
layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
layers.MaxPooling2D((2,2)),
layers.Conv2D(64, (3,3), activation='relu'),
layers.MaxPooling2D((2,2)),
layers.Conv2D(128, (3,3), activation='relu'),
layers.MaxPooling2D((2,2)),
layers.Flatten(),
# ❌ Do NOT specify input_dim here
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(1, activation='sigmoid') # binary classification
])
model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])
model.summary()
history = model.fit(
train_generator,
epochs=15,
validation_data=val_generator
)
import matplotlib.pyplot as plt
# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# Predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices,
yticklabels=test_generator.class_indices)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# ✅ Step 1: Get y_true and y_pred from your test set
y_true = test_generator.classes # Actual labels
y_pred = (model.predict(test_generator) > 0.5).astype("int32").flatten() # Predicted labels
# ✅ Step 2: Calculate Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
metrics = [acc, prec, rec, f1]
labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
# ✅ Step 3: Plot Performance Metrics
plt.figure(figsize=(8,5))
bars = plt.bar(labels, metrics, color="teal", width=0.6)
# Add value labels on top of bars
for bar, metric in zip(bars, metrics):
yval = bar.get_height()
plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{metric:.4f}", ha="center",
va="bottom")
plt.ylim(0,1.1)
plt.title("CNN Model Performance Metrics")
plt.ylabel("Score")
plt.show()
import matplotlib.pyplot as plt
# Example counts before augmentation
edible_original = 300
poisonous_original = 200
# Example counts after augmentation
edible_augmented = 900
poisonous_augmented = 600
# Categories
labels = ["Edible", "Poisonous"]
# Values for bar chart
original_counts = [edible_original, poisonous_original]
augmented_counts = [edible_augmented, poisonous_augmented]
x = range(len(labels))
plt.bar(x, original_counts, width=0.4, label="Original", color="skyblue", align='center')
plt.bar(x, augmented_counts, width=0.4, label="Augmented", color="orange", align='edge')
plt.xticks(x, labels)
plt.ylabel("Number of Images")
plt.title("Comparison of Edible vs Poisonous (Original vs Augmented)")
plt.legend()
plt.show()
