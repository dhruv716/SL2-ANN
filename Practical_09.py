import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
    
    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
    
    def predict(self, pattern):
        energy = -0.5 * np.dot(np.dot(pattern, self.weights), pattern)
        return np.sign(np.dot(pattern, self.weights) + energy)

def main():
    patterns = np.array([
        [1, 1, -1, -1],
        [-1, -1, 1, 1],
        [1, -1, 1, -1],
        [-1, 1, -1, 1]
    ])
    n_neurons = patterns.shape[1]
    network = HopfieldNetwork(n_neurons)
    network.train(patterns)
    
    for pattern in patterns:
        prediction = network.predict(pattern)
        print('Input pattern:', pattern)
        print('Predicted pattern:', prediction)

main()

"""

HopfieldNetwork class:
__init__: Initializes a Hopfield network with a specified number of neurons (n_neurons). It initializes the weights matrix to zeros.
train: Trains the network by updating the weights based on the outer product of each input pattern.
predict: Predicts the output pattern based on the trained weights and input pattern.

main():
Defines the main execution block. This ensures that the following code runs only when the script is executed directly (not imported as a module).
Defines four input patterns stored in the patterns NumPy array.
Calculates the number of neurons based on the shape of the patterns array.
Creates an instance of the HopfieldNetwork class and trains it with the input patterns.
Iterates over each input pattern, predicts the output pattern using the trained network, and prints the input and predicted patterns.

"""
