import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, epochs=100):
        self.weights = np.random.randn(input_dim + 1)  # Add 1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * np.insert(X[i], 0, 1)

def plot_decision_regions(X, y, perceptron):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = np.array([perceptron.predict(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Regions')
    plt.show()

# Example data
X = np.array([[2, 3], [3, 4], [4, 3], [3, 2], [1, 3], [2, 2]])
y = np.array([1, 1, 1, 0, 0, 0])

# Create and train perceptron
perceptron = Perceptron(input_dim=2)
perceptron.train(X, y)

# Plot decision regions
plot_decision_regions(X, y, perceptron)


"""

Perceptron Class:
__init__(self, input_dim, learning_rate=0.01, epochs=100): Initializes the perceptron with random weights and specifies the learning rate 
and number of epochs for training.

activation(self, x): Defines the step function as the activation function. 
Returns 1 if the input x is greater than or equal to 0, otherwise returns 0.

predict(self, x): Computes the output of the perceptron for a given input x by adding a bias term, taking the dot product with the weights, 
and applying the activation function.

train(self, X, y): Trains the perceptron using the given training data X and labels y. 
It adjusts the weights using the perceptron learning rule for the specified number of epochs.

plot_decision_regions() Function:
Plots the decision regions of the perceptron classifier on a 2D feature space. 
It generates a meshgrid of points covering the feature space, predicts the class labels for each point using the perceptron, 
and then plots the decision boundaries and data points.

Example Data:
X: An array of feature vectors, where each row represents a sample and each column represents a feature.
y: An array of target labels corresponding to each sample in X.

Creating and Training Perceptron:
Creates an instance of the Perceptron class with an input dimension of 2.
Trains the perceptron using the example data X and y.

Plotting Decision Regions:
Calls the plot_decision_regions() function to visualize the decision regions of the trained perceptron classifier using the example data.
Visualization:
The decision regions are plotted along with the data points. 
The decision regions separate the feature space into regions corresponding to different classes, as determined by the trained perceptron classifier.

"""
