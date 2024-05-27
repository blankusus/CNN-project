#10 classes
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

train_data = pd.read_csv(r"C:\Users\Ruxandra\Desktop\nnproj\train.csv") #60k images
test_data = pd.read_csv(r"C:\Users\Ruxandra\Desktop\nnproj\test.csv") #10k images

#first column is unnamed
train_data = train_data.drop('Unnamed: 0', axis=1)
test_data = test_data.drop('Unnamed: 0', axis=1)

X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  #reshape and convert to 28x28 images
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
y_test = test_data.iloc[:, 0].values

X_train = X_train.astype('float32') / 255.0  #normalize pixel values
X_test = X_test.astype('float32') / 255.0

'''---------------------------------------'''

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) #if val loss does not improve for 3 epochs, it stops training
model_checkpoint = callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss') #when val loss improves, save the model in a file

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

'''---------------------------------------'''

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

'''----------------------PLOTS----------------------'''

img = [str(i) for i in range(10)]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].reshape((28, 28)), cmap=plt.cm.binary)
    plt.xlabel(img[y_train[i]])

'''---------------------------------------'''

tacc = history.history['accuracy']
tloss = history.history['loss']
vacc = history.history['val_accuracy']
vloss = history.history['val_loss']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(tloss, label='Training Loss', color='red')
plt.plot(vloss, label='Validation Loss', color='green')
plt.scatter(np.argmin(vloss), np.min(vloss), s=150, c='blue', label=f'Best Epoch = {np.argmin(vloss) + 1}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(tacc, label='Training Accuracy', color='red')
plt.plot(vacc, label='Validation Accuracy', color='green')
plt.scatter(np.argmax(vacc), np.max(vacc), s=150, c='blue', label=f'Best Epoch = {np.argmax(vacc) + 1}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

'''---------------------------------------'''

y_pred_probs = model.predict(X_test) #probabilities for each class
y_pred = np.argmax(y_pred_probs, axis=1) #class with highest probability
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()