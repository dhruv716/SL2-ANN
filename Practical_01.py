import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def prelu(x, alpha):
    return np.maximum(alpha * x, x)

def elu(x, alpha=1.0):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)

def softmax(x):
    exp_values = np.exp(x - np.max(x)) # for numerical stability
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)

# Linear activation function
def linear(x):
    return x

# Input values
x_values = np.linspace(-5, 5, 100)

# Plotting each activation function
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.title('Sigmoid Function')
plt.plot(x_values, sigmoid(x_values))
plt.grid(True)

plt.subplot(2, 4, 2)
plt.title('Hyperbolic Tangent Function (tanh)')
plt.plot(x_values, tanh(x_values))
plt.grid(True)

plt.subplot(2, 4, 3)
plt.title('ReLU (Rectified Linear Unit)')
plt.plot(x_values, relu(x_values))
plt.grid(True)

plt.subplot(2, 4, 4)
plt.title('Leaky ReLU')
plt.plot(x_values, leaky_relu(x_values))
plt.grid(True)

plt.subplot(2, 4, 5)
plt.title('PReLU (Parametric ReLU)')
alpha_prelu = 0.01
plt.plot(x_values, prelu(x_values, alpha_prelu))
plt.grid(True)

plt.subplot(2, 4, 6)
plt.title('ELU (Exponential Linear Unit)')
plt.plot(x_values, elu(x_values))
plt.grid(True)

plt.subplot(2, 4, 7)
plt.title('Softmax')
input_softmax = np.array([2.0, 1.0, 0.1])
plt.bar(range(len(input_softmax)), softmax(input_softmax))
plt.grid(True)

plt.subplot(2, 4, 8)
plt.title('Linear')
plt.plot(x_values, linear(x_values))
plt.grid(True)

plt.tight_layout()
plt.show()

"""
Importing Libraries:
import numpy as np: Imports the NumPy library for numerical computations.
import matplotlib.pyplot as plt: Imports the Matplotlib library for plotting graphs.

Activation Functions:
sigmoid(x): Defines the sigmoid activation function.
tanh(x): Defines the hyperbolic tangent (tanh) activation function.
relu(x): Defines the Rectified Linear Unit (ReLU) activation function.
leaky_relu(x, alpha): Defines the Leaky ReLU activation function with an optional parameter alpha.
prelu(x, alpha): Defines the Parametric ReLU (PReLU) activation function with a learnable parameter alpha.
elu(x, alpha): Defines the Exponential Linear Unit (ELU) activation function with an optional parameter alpha.
softmax(x): Defines the softmax activation function, which is typically used in the output layer for multi-class classification tasks.
linear(x): Defines the linear activation function, which simply returns the input value unchanged.

Input Values:
x_values: Generates a range of input values from -5 to 5.

Plotting:
The plt.figure(figsize=(15, 10)) line creates a new figure with a specified size.
Subplots are created using plt.subplot to plot each activation function.
Each subplot displays the graph of a specific activation function using plt.plot.
Titles are added to each subplot using plt.title.
plt.grid(True) adds a grid to each subplot.
Finally, plt.tight_layout() adjusts the layout to prevent overlapping subplots, and plt.show() displays the entire plot.


"""







