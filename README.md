- This project is based on a script that trains a Convolutional Neural Network (CNN) on a
dataset of 28x28 grayscale images of hiragana characters for multi-class recognition. The dataset
is imported from Kaggle and it consists of 10 classes, a training set which contains 60,000
images and a testing set, containing 10,000 images.
- The model architecture includes convolutional layers to detect image features, pooling
layers to reduce the size of the feature maps and dense layers for the final classification. ReLU
activation functions are used for the intermediate layers, while the Softmax function handles the
final output. EarlyStopping and ModelCheckpoint callbacks were implemented to prevent
overfitting and save the best version of the model. The training loss and accuracy provide details
about how well the model is fitting the training data. The validation loss and accuracy provide
details about how well the model is performing on data that the model has not been exposed to
during the training. The training data was split into 80% for training and 20% for validation, with
the model training for up to 10 epochs.
- The results were good, with the model achieving a test accuracy of 92.7% and a
validation accuracy that peaked at 97.2%. I used a confusion matrix to analyze
misclassifications, graphs which included training and validation accuracy and loss trends over
the epochs, as well as sample images from the dataset.
