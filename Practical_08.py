import numpy as np

# Define sigmoid activation function 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid function 
def sigmoid_derivative(x):
    return x * (1 - x)

# Define input dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define output dataset
y = np.array([[0], [1], [1], [0]])

# Define hyperparameters
learning_rate = 0.1
num_epochs = 100000

# Initialize weights randomly with mean 0 
hidden_weights = 2 * np.random.random((2, 2)) - 1 
output_weights = 2 * np.random.random((2, 1)) - 1

# Train the neural network
for i in range(num_epochs):
    # Forward propagation
    hidden_layer = sigmoid(np.dot(X, hidden_weights)) 
    output_layer = sigmoid(np.dot(hidden_layer, output_weights))

    # Backpropagation
    output_error = y - output_layer
    output_delta = output_error * sigmoid_derivative(output_layer)
    hidden_error = output_delta.dot(output_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)
    
    output_weights += hidden_layer.T.dot(output_delta) * learning_rate 
    hidden_weights += X.T.dot(hidden_delta) * learning_rate

# Display input and output
print("Input:")
print(X)
print("Output:")
print(output_layer)


"""

Importing Libraries:
import numpy as np: Imports the NumPy library for numerical computations.

Defining Activation Functions:
sigmoid(x): Defines the sigmoid activation function, which returns the sigmoid of the input.
sigmoid_derivative(x): Defines the derivative of the sigmoid function, which is used in backpropagation.

Defining Input and Output Datasets:
X: Defines the input dataset containing four sets of binary values (0 or 1).
y: Defines the corresponding output dataset (labels) for the input dataset.

Defining Hyperparameters:
learning_rate: Sets the learning rate, which determines the step size during gradient descent.
num_epochs: Specifies the number of epochs (iterations) for training the neural network.

Initializing Weights:
hidden_weights: Initializes the weights of the connections between the input and hidden layers with random values between -1 and 1.
output_weights: Initializes the weights of the connections between the hidden and output layers similarly.

Training the Neural Network:
Iterates over the specified number of epochs.
Performs forward propagation to calculate the output of the neural network.
Computes the error between the predicted output and the actual output.
Performs backpropagation to update the weights based on the error using gradient descent.
Updates the weights of both hidden and output layers based on the calculated deltas and learning rate.

Displaying Input and Output:
Prints the input dataset X.
Prints the final output layer after training, which represents the predicted outputs for the input dataset.

"""
