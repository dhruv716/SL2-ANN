import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)




"""
Importing Libraries:
import tensorflow as tf: Imports the TensorFlow library, which is used for building and training neural networks.
from sklearn.model_selection import train_test_split: Imports the train_test_split function from scikit-learn, which is used to split the dataset into training and testing sets.
from sklearn.preprocessing import StandardScaler: Imports the StandardScaler class from scikit-learn, which is used to standardize the dataset.
from sklearn.datasets import load_breast_cancer: Imports the load_breast_cancer function from scikit-learn, which is used to load the Breast Cancer dataset.

Loading the Dataset:
data = load_breast_cancer(): Loads the Breast Cancer dataset.
X = data.data: Assigns the features (input variables) of the dataset to X.
y = data.target: Assigns the target variable (labels) of the dataset to y.

Splitting the Dataset:
train_test_split(X, y, test_size=0.20, random_state=42): Splits the dataset into training and testing sets, with 80% of the data used for training and 20% for testing.

Standardizing the Data:
scaler = StandardScaler(): Initializes a StandardScaler object.
X_train = scaler.fit_transform(X_train): Standardizes the training data.
X_test = scaler.transform(X_test): Standardizes the testing data.

Defining the Neural Network Model:
model = tf.keras.models.Sequential([...]): Defines a sequential model using Keras.
tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],)): Adds a dense layer with 1 neuron and sigmoid activation function. The input shape is determined by the number of features in the dataset.

Compiling the Model:
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']): Compiles the model with the Adam optimizer, binary cross-entropy loss function (for binary classification), and accuracy metric.

Training the Model:
model.fit(X_train, y_train, epochs=5): Trains the model on the training data for 5 epochs.

Evaluating the Model:
test_loss, test_accuracy = model.evaluate(X_test, y_test): Evaluates the trained model on the testing data.
print("Test Accuracy:", test_accuracy): Prints the test accuracy of the model.

"""