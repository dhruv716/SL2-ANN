import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 input images to a 1D array
    Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 units for 10 classes and softmax activation
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy for integer labels
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


"""
TensorFlow and Keras are imported to build and train the neural network model.
The MNIST dataset containing handwritten digits is loaded using mnist.load_data(). The dataset is divided into training and testing sets.
The pixel values of the images are normalized to the range [0, 1] by dividing by 255.0.
The model architecture is defined using the Sequential API. It consists of three layers:
The input layer, Flatten, converts the 28x28 input images into a 1D array.
The first hidden layer, Dense, has 128 units with ReLU activation function.
The output layer, Dense, has 10 units (one for each class) with softmax activation function.
The model is compiled using the Adam optimizer with a learning rate of 0.001, sparse categorical cross-entropy loss function (suitable for integer labels), and accuracy as the metric to monitor during training.
The model is trained on the training data for 10 epochs with a batch size of 64.
Finally, the model is evaluated on the test data, and the test loss and accuracy are printed.

"""