import numpy as np

class XORNetwork:
    def __init__(self):
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.random.randn(1, 2)
        self.W2 = np.random.randn(2, 1)
        self.b2 = np.random.randn(1, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backpropagation(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        z1_error = output_delta.dot(self.W2.T)
        z1_delta = z1_error * self.sigmoid_derivative(self.a1)
        
        self.W1 += X.T.dot(z1_delta)
        self.b1 += np.sum(z1_delta, axis=0, keepdims=True)
        self.W2 += self.a1.T.dot(output_delta)
        self.b2 += np.sum(output_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward_pass(X)
            self.backpropagation(X, y, output)

    def predict(self, X):
        return self.forward_pass(X)

if __name__ == "__main__":
    xor_nn = XORNetwork()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    xor_nn.train(X, y, epochs=10000)

    predictions = xor_nn.predict(X)

    print(predictions)


"""This line defines a class named XORNetwork, which represents our neural network model for solving the XOR problem.

This part defines the constructor method __init__, which initializes the weights and biases of the neural network with random values. We have weights W1 and W2 along with biases b1 and b2.

Here, we define a sigmoid activation function, which is a common activation function used in neural networks. It squashes the input values between 0 and 1.

This function computes the derivative of the sigmoid activation function. It is used during backpropagation to update the weights of the network.

This method performs the forward pass of the neural network. It takes the input X, computes the activations of the hidden layer and output layer, and returns the output predictions.

This method implements the backpropagation algorithm to update the weights and biases of the network based on the errors in the predictions. It adjusts the parameters to minimize the error between the predicted output and the true output y.

Here, we define a method train to train the neural network. It iterates over the dataset for a specified number of epochs, computes the forward pass, and then applies backpropagation to update the weights and biases.

This method predicts the output for given input data X using the trained neural network. It simply calls the forward_pass method.

Finally, this block of code checks if the script is run directly (not imported as a module). It creates an instance of the XORNetwork class, defines the input X and output y for the XOR problem, trains the network, makes predictions, and prints the output predictions.

"""