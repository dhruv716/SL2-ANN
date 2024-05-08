import numpy as np

class McCullochPittsNN:
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs)
        self.threshold = 0

    def set_weights(self, weights):
        if len(weights) != len(self.weights):
            raise ValueError("Number of weights must match number of inputs")
        self.weights = np.array(weights)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def activation_function(self, net_input):
        return 1 if net_input > self.threshold else 0

    def forward_pass(self, inputs):
        net_input = np.dot(inputs, self.weights)
        return self.activation_function(net_input)

def generate_ANDNOT():
    # Initialize McCulloch-Pitts neural network with 2 inputs
    mp_nn = McCullochPittsNN(2)
    mp_nn.set_weights([-1, 1])  # Weights for ANDNOT function
    mp_nn.set_threshold(0)
    # Generate truth table for ANDNOT function
    truth_table = [(0, 0), (0, 1), (1, 0), (1, 1)]
    print("Truth Table for ANDNOT Function:")
    print("Input1\tInput2\tOutput")
    for inputs in truth_table:
        output = mp_nn.forward_pass(inputs)
        print(f"{inputs[0]}\t{inputs[1]}\t{output}")

def main():
    while True:
        print("\nMenu:")
        print("1. Generate ANDNOT Function")
        print("2. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            generate_ANDNOT()
        elif choice == "2":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")


main()

"""

McCullochPittsNN Class:
__init__(self, num_inputs): Initializes the MP neuron with a given number of inputs. It sets the initial weights to zero and the threshold to zero.
set_weights(self, weights): Sets the weights of the neuron. It raises a ValueError if the number of weights provided doesn't match the number of inputs.
set_threshold(self, threshold): Sets the threshold of the neuron.
activation_function(self, net_input): Defines the activation function of the neuron, which returns 1 if the net input is greater than the threshold, otherwise returns 0.
forward_pass(self, inputs): Performs a forward pass through the neuron, computing the net input and applying the activation function to produce the output.

generate_ANDNOT() Function:
Initializes an MP neuron with 2 inputs and sets the weights and threshold for the ANDNOT function.
Generates the truth table for the ANDNOT function by iterating over all possible input combinations and printing the corresponding outputs.

main() Function:
Provides a menu-driven interface for the user.
Option 1 allows the user to generate the truth table for the ANDNOT function using the generate_ANDNOT() function.
Option 2 exits the program.

Execution:
The main() function is called to start the program.
The user is prompted to choose an option from the menu.
If the user selects option 1, the truth table for the ANDNOT function is generated and displayed.
If the user selects option 2, the program exits.

"""



