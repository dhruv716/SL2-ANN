import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros((input_size, 1))
        self.bias = 0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        return self.activation(np.dot(x, self.weights) + self.bias)

    def train(self, X, y, learning_rate=0.1, epochs=100):
        for _ in range(epochs):
            for i in range(len(y)):
                prediction = self.predict(X[i])
                self.weights += learning_rate * (y[i] - prediction) * X[i].reshape(-1, 1)
                self.bias += learning_rate * (y[i] - prediction)

def ascii_to_binary(number):
    binary_rep = bin(ord(number) - ord('0'))[2:].zfill(7)
    return [int(bit) for bit in binary_rep]
  
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def main():
    # Training data
    X_train = np.array([ascii_to_binary(str(i)) for i in range(10)])
    y_train = np.array([i % 2 for i in range(10)]).reshape(-1, 1)

    # Create and train perceptron
    perceptron = Perceptron(input_size=7)  # ASCII 0 to 9 represented in 7-bit binary
    perceptron.train(X_train, y_train)

    # Test data
    test_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    X_test = np.array([ascii_to_binary(number) for number in test_numbers])

    # Predict and print results
    print("Number\tPrediction")
    predictions = []
    for i in range(len(X_test)):
        prediction = perceptron.predict(X_test[i])
        predictions.append(prediction)
        print(f"{test_numbers[i]}\t{'Even' if prediction == 0 else 'Odd'}")

    # Calculate accuracy
    y_test = np.array([i % 2 for i in range(10)])
    accuracy = calculate_accuracy(y_test, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")


main()


"""

Perceptron Class:
__init__(self, input_size): Initializes the perceptron with zero weights and bias. The input size corresponds to the number of input features.
activation(self, x): Defines the step function as the activation function. Returns 1 if the input x is greater than or equal to 0, otherwise returns 0.
predict(self, x): Computes the output of the perceptron by taking the dot product of input x and weights, adding bias, and applying the activation function.
train(self, X, y, learning_rate=0.1, epochs=100): Trains the perceptron using the given training data X and labels y for a specified number of epochs. It adjusts the weights and bias using the perceptron learning rule.

ascii_to_binary() Function:
Converts a decimal digit (0-9) to its 7-bit binary representation. Each digit is represented as a 7-bit binary array.
calculate_accuracy() Function:
Calculates the accuracy of the model by comparing the true labels y_true with the predicted labels y_pred.

main() Function:
Prepares the training data (X_train, y_train) and test data (X_test) using ASCII representations of numbers and their binary equivalents.
Creates an instance of the Perceptron class and trains it on the training data.
Tests the trained perceptron on the test data to predict whether each number is even or odd.
Calculates and prints the accuracy of the model.

Execution:
The main() function is called to start the program.
The perceptron is trained on binary representations of numbers from 0 to 9, where the labels indicate whether the numbers are even or odd.
The trained perceptron is then used to predict whether each test number is even or odd, and the results are printed.
Finally, the accuracy of the perceptron model is calculated and displayed.

"""
