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

"""     
