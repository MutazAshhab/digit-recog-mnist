from functools import partial

import keras
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Process followed:
# Data preparation
#   - Loading the data
#   - Check for missing values
#   - Check if the data is normalized
#   - Reshape data
#   - Label encoding
#   - Split training and testing dataset
# CNN
#   - Define the model
#   - Set the optimizer
# Evaluate the model
#   - Training and validation curves
# Prediction and submition
#   - Predict

# Defining constants
BATCH_SIZE = 32
EPOCHS = 4
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
LABELS = 10

# Loading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

labels = train['label']
train = train.drop(labels=['label'], axis=1)

# Checking for missing values
train.isnull().any().describe()
test.isnull().any().describe()

# Check if the data is normalized
max(train.loc[0])

# Normalizing the data
train = train / 255
max(train.loc[0])

# Reshape
# images are 28x28
train = train.values.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
test = test.values.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

# Transform to one hot encoding
labels = tf.keras.utils.to_categorical(labels, num_classes=LABELS)

# Split training and testing dataset
training_data, training_val, training_labels, labels_val = train_test_split(train, labels, test_size=0.1,
                                                                            random_state=42)

# Creating the CNN model
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False)


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


# Setting the model
model = keras.models.Sequential()
model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                        input_shape=[28, 28, 1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Fitting the model
history = model.fit(training_data, training_labels, batch_size=BATCH_SIZE,
                    epochs=EPOCHS, validation_data=(training_val, labels_val), verbose=2)

# Epoch 1/4
# 1182/1182 - 84s - loss: 0.3406 - accuracy: 0.9036 - val_loss: 0.1151 - val_accuracy: 0.9643
# Epoch 2/4
# 1182/1182 - 37s - loss: 0.0844 - accuracy: 0.9746 - val_loss: 0.1185 - val_accuracy: 0.9602
# Epoch 3/4
# 1182/1182 - 36s - loss: 0.0543 - accuracy: 0.9825 - val_loss: 0.0567 - val_accuracy: 0.9824
# Epoch 4/4
# 1182/1182 - 37s - loss: 0.0444 - accuracy: 0.9855 - val_loss: 0.0437 - val_accuracy: 0.9848
# Output when ran on google colab ^

# Evaluating the model
# Plotting the loss curves
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

# Plotting the accuracy curve
ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# predict the unseen data
results = model.predict(test)

