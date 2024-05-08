import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Define patterns for numbers 0, 1, 2, and 3
patterns = {
    0: np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]),
    1: np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]]),
    2: np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]]),
    3: np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]])
}

# Flatten the patterns and create labels
X = [pattern.flatten() for pattern in patterns.values()]
y = list(patterns.keys())

# Create neural network model
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Train the model
model.fit(X, y)
print("Model trained successfully!")

# Function to test the model
def test_model():
    test_data = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]).flatten()
    predicted_label = model.predict([test_data])[0]
    print("Predicted Label:", predicted_label)

# Function to display accuracy
def display_accuracy():
    predicted_labels = model.predict(X)
    acc = accuracy_score(y, predicted_labels)
    print("Accuracy:", acc)

# Function to predict based on user input
def predict_number():
    print("Enter the pattern for the number (0s and 1s only):")
    test_data = [list(map(int, input(f"Row {i+1}: ").strip())) for i in range(5)]
    test_data = np.array(test_data).flatten()
    predicted_label = model.predict([test_data])[0]
    print("Predicted Label:", predicted_label)

# Main menu
while True:
    print("\nMenu:")
    print("1. Test Model")
    print("2. Display Accuracy")
    print("3. Predict Number")
    print("4. Exit")
    choice = input("Enter your choice (1-4): ")
    if choice == '1':
        test_model()
    elif choice == '2':
        display_accuracy()
    elif choice == '3':
        predict_number()
    elif choice == '4':
        print("Exiting program.")
        break
    else:
        print("Invalid choice. Please enter a number from 1 to 4.")


"""Output: 

Model trained successfully!

Menu:
1. Test Model
2. Display Accuracy
3. Predict Number
4. Exit
Enter your choice (1-4): 1
Predicted Label: 0

Menu:
1. Test Model
2. Display Accuracy
3. Predict Number
4. Exit
Enter your choice (1-4): 2
Accuracy: 1.0

Menu:
1. Test Model
2. Display Accuracy
3. Predict Number
4. Exit
Enter your choice (1-4): 3
Enter the pattern for the number (0s and 1s only):
Row 1: 111 
Row 2: 101
Row 3: 101
Row 4: 101
Row 5: 111
Predicted Label: 0

Menu:
1. Test Model
2. Display Accuracy
3. Predict Number
4. Exit
Enter your choice (1-4): 3
Enter the pattern for the number (0s and 1s only):
Row 1: 101
Row 2: 111
Row 3: 101
Row 4: 111
Row 5: 101
Predicted Label: 3

Menu:
1. Test Model
2. Display Accuracy
3. Predict Number
4. Exit
Enter your choice (1-4): 4
Exiting program.


Explanation: 

Importing Libraries:
import numpy as np: Imports the NumPy library for numerical computations.
from sklearn.neural_network import MLPClassifier: Imports the Multi-Layer Perceptron (MLP) classifier from scikit-learn, which is used to train a neural network.
from sklearn.metrics import accuracy_score: Imports the accuracy_score function from scikit-learn, which is used to calculate the accuracy of the model.

Defining Patterns:
patterns: Dictionary containing patterns for numbers 0, 1, 2, and 3, represented as binary matrices.
Data Preparation:
X: Flattens the patterns into 1D arrays and stores them in a list.
y: Stores the corresponding labels for each pattern.

Creating and Training the Model:
model: Creates an MLP classifier with a single hidden layer containing 10 neurons and a maximum of 1000 iterations for training.
model.fit(): Trains the MLP classifier using the flattened patterns and their corresponding labels.

Testing the Model:
test_model(): Function to test the trained model by providing a sample pattern. It flattens the pattern, predicts the label using the trained model, and prints the predicted label.

Displaying Accuracy:
display_accuracy(): Function to calculate and display the accuracy of the trained model using the entire dataset.
Predicting Number based on User Input:
predict_number(): Function to predict the number based on user input. It prompts the user to enter the pattern for the number and predicts the label using the trained model.

Main Menu:
Displays a menu with options to test the model, display accuracy, predict a number based on user input, or exit the program.
Accepts user input for selecting an option and performs the corresponding action based on the choice.

    
"""
