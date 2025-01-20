# Batch-Normalization-Better-neural-network-convergence-by-standardizing-hidden-states
Batch Normalization (BN) is a technique used to improve the training of deep neural networks by standardizing the inputs to each layer during training. This helps in stabilizing the learning process and often leads to faster convergence. In essence, BN normalizes the activations of a given layer by scaling and shifting them based on their statistics (mean and variance) across the mini-batch.

In this code, I’ll show you how to apply Batch Normalization to a simple neural network using TensorFlow and Keras.
Steps:

    Create a neural network using Keras.
    Apply Batch Normalization to hidden layers.
    Train the model on a sample dataset (e.g., MNIST).
    Compare performance with and without Batch Normalization.

Code Implementation (Batch Normalization with TensorFlow/Keras):

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load and preprocess MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshaping the data to include a channel dimension (1 for grayscale images)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Normalize the images to the range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Step 2: Create a simple neural network model with Batch Normalization
def create_model_with_batch_norm():
    model = models.Sequential()

    # Convolutional layer followed by Batch Normalization
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())  # Batch Normalization layer
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())  # Batch Normalization layer
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())  # Batch Normalization layer
    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())  # Batch Normalization layer
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 3: Create and train the model
model_with_bn = create_model_with_batch_norm()
history_with_bn = model_with_bn.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Step 4: Create a neural network model without Batch Normalization
def create_model_without_batch_norm():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 5: Create and train the model without Batch Normalization
model_without_bn = create_model_without_batch_norm()
history_without_bn = model_without_bn.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Step 6: Compare the results
def plot_history(history_with_bn, history_without_bn):
    # Plot training & validation accuracy values
    plt.plot(history_with_bn.history['accuracy'], label='With Batch Normalization (train)')
    plt.plot(history_without_bn.history['accuracy'], label='Without Batch Normalization (train)')
    plt.plot(history_with_bn.history['val_accuracy'], label='With Batch Normalization (validation)')
    plt.plot(history_without_bn.history['val_accuracy'], label='Without Batch Normalization (validation)')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history_with_bn.history['loss'], label='With Batch Normalization (train)')
    plt.plot(history_without_bn.history['loss'], label='Without Batch Normalization (train)')
    plt.plot(history_with_bn.history['val_loss'], label='With Batch Normalization (validation)')
    plt.plot(history_without_bn.history['val_loss'], label='Without Batch Normalization (validation)')
    plt.title('Model Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

plot_history(history_with_bn, history_without_bn)

Explanation:

    Data Preprocessing: The MNIST dataset is loaded, reshaped to include a channel dimension, normalized to the range [0, 1], and one-hot encoded.

    Model with Batch Normalization: The create_model_with_batch_norm() function defines a CNN with Batch Normalization layers added after each convolutional layer and dense layer.

    Model without Batch Normalization: The create_model_without_batch_norm() function defines a similar CNN, but without Batch Normalization.

    Training the Models: Both models are trained on the MNIST dataset for 5 epochs, using the Adam optimizer and categorical cross-entropy loss.

    Plotting the Results: The training and validation accuracy and loss for both models (with and without Batch Normalization) are plotted for comparison.

Output:

    With Batch Normalization: You should notice that the model converges faster, and the validation accuracy improves more rapidly.
    Without Batch Normalization: The model may take longer to converge, and the accuracy may be lower, especially as the network depth increases.

Key Takeaways:

    Batch Normalization helps stabilize the learning process by reducing internal covariate shift and improving convergence.
    It normalizes the inputs to each layer by adjusting and scaling them based on the batch statistics (mean and variance).
    It can lead to faster training and potentially higher accuracy, especially in deeper networks.

Conclusion:

In this example, we demonstrated how to implement Batch Normalization in a deep neural network using Keras and TensorFlow. Batch Normalization is particularly useful in deeper networks, and it improves the training stability and speed of convergence.
