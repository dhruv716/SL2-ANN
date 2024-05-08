import numpy as np

class BAM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((input_size, output_size))

    def train(self, input_patterns, output_patterns):
        for x, y in zip(input_patterns, output_patterns):
            self.weights += np.outer(x, y)

    def recall(self, input_pattern):
        output_pattern = np.dot(input_pattern, self.weights)
        output_pattern[output_pattern >= 0] = 1
        output_pattern[output_pattern < 0] = -1
        return output_pattern

def train_bam(bam):
    input_patterns = []
    output_patterns = []
    num_pairs = int(input("Enter the number of input-output pairs: "))
    for _ in range(num_pairs):
        input_pattern = np.array([int(x) for x in input("Enter input pattern (space-separated values): ").split()])
        output_pattern = np.array([int(x) for x in input("Enter corresponding output pattern (space-separated values): ").split()])
        input_patterns.append(input_pattern)
        output_patterns.append(output_pattern)
  
    bam.train(input_patterns, output_patterns)
    print("BAM trained successfully.")

def recall_bam(bam):
    input_pattern = np.array([int(x) for x in input("Enter input pattern to recall (space-separated values): ").split()])
    recalled_output = bam.recall(input_pattern)
    print("Recalled Output:", recalled_output)

if __name__ == "__main__":
    input_size = int(input("Enter the input vector size: "))
    output_size = int(input("Enter the output vector size: "))
    bam = BAM(input_size, output_size)

    while True:
        print("\nMenu:")
        print("1. Train BAM")
        print("2. Recall BAM")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        if choice == '1':
            train_bam(bam)
        elif choice == '2':
            recall_bam(bam)
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            
            
"""
OUTPUT :
Enter the input vector size: 2
Enter the output vector size: 2

Menu:
1. Train BAM
2. Recall BAM
3. Exit
Enter your choice (1/2/3): 1
Enter the number of input-output pairs: 2
Enter input pattern (space-separated values): 1 -1
Enter corresponding output pattern (space-separated values): 1 -1
Enter input pattern (space-separated values): -1 1
Enter corresponding output pattern (space-separated values): -1 1
BAM trained successfully.

Menu:
1. Train BAM
2. Recall BAM
3. Exit
Enter your choice (1/2/3): 2
Enter input pattern to recall (space-separated values): 1 -1
Recalled Output: [ 1. -1.]

Menu:
1. Train BAM
2. Recall BAM
3. Exit
Enter your choice (1/2/3): 3
Exiting program. 


EXPLANATION: 

BAM Class:
__init__(self, input_size, output_size): Initializes the BAM with the specified input and output vector sizes. It also initializes the weight matrix to zeros.

train(self, input_patterns, output_patterns): Trains the BAM by updating the weight matrix based on the input-output pattern pairs. 
It computes the outer product of each input-output pair and adds it to the weight matrix.

recall(self, input_pattern): Recalls the associated output pattern for a given input pattern by computing the dot product of the input pattern with the weight matrix. 
It then threshold the output pattern values to either 1 or -1 based on their sign.

train_bam Function:
Allows the user to input multiple input-output pattern pairs to train the BAM. It prompts the user to enter the input and corresponding output patterns and then calls the train method of the BAM object to train the network.

recall_bam Function:
Allows the user to input an input pattern to recall the associated output pattern using the trained BAM. It prompts the user to enter the input pattern and then calls the recall method of the BAM object to recall the output pattern.


Main Program:
Prompts the user to enter the input and output vector sizes to initialize the BAM object.
Displays a menu with options to train the BAM, recall the BAM, or exit the program.
Based on the user's choice, it calls the corresponding functions (train_bam, recall_bam) or exits the program.

Usage:
Users can interactively train the BAM model by providing input-output pattern pairs.
They can also recall the associated output pattern for a given input pattern after training the BAM.

"""     
