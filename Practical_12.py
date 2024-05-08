import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


"""
Importing Libraries:
import tensorflow as tf: Imports TensorFlow, a popular deep learning framework.
from tensorflow.keras.datasets import mnist: Imports the MNIST dataset from the Keras library, which provides easy access to common datasets for machine learning.
from tensorflow.keras.models import Sequential: Imports the Sequential model from Keras, which allows us to build neural networks layer by layer.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense: Imports specific layers needed for building the CNN architecture.
from tensorflow.keras.utils import to_categorical: Imports the to_categorical function, which converts class labels to one-hot encoded vectors.

Loading and Preprocessing the Data:
mnist.load_data(): Loads the MNIST dataset, consisting of handwritten digit images and their corresponding labels.
X_train, y_train, X_test, y_test: Split the dataset into training and testing sets.
X_train.reshape(-1, 28, 28, 1) / 255.0: Reshapes the training images to have a single channel (grayscale) and normalizes pixel values to the range [0, 1].
y_train and y_test are converted to one-hot encoded format using to_categorical.

Building the CNN Model:
model = Sequential([...]): Initializes a sequential model.
Conv2D layers with ReLU activation: Perform convolutional operations on the input images, extracting features with filters of size 3x3.
MaxPooling2D layers: Perform max pooling to reduce spatial dimensions and extract dominant features.
Flatten: Flatten the feature maps into a vector to feed into the fully connected layers.
Dense layers with ReLU and softmax activations: Implement fully connected layers for classification, with the final layer using softmax activation for multiclass classification.

Compiling the Model:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']): Compiles the model with the Adam optimizer, categorical cross-entropy loss function (suitable for multi-class classification), and accuracy as the evaluation metric.

Training the Model:
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1): Trains the model on the training data for 10 epochs, using a batch size of 64.

Evaluating the Model:
model.evaluate(X_test, y_test): Evaluates the trained model on the test data.
Prints the test loss and accuracy.

"""