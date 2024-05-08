import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model
model = Sequential()
model.add(Input(shape=(32, 32, 3)))  
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the data
batch_size = 32
train_set = train_datagen.flow(X_train, y_train, batch_size=batch_size)
test_set = test_datagen.flow(X_test, y_test, batch_size=batch_size)

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
epochs = 5
model.fit(train_set, steps_per_epoch=len(X_train)//batch_size, epochs=epochs,
          validation_data=test_set, validation_steps=len(X_test)//batch_size)

# Evaluate the model
score = model.evaluate(test_set, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


"""

Importing Libraries:
import keras: Imports the Keras library for building and training neural networks.
from keras.datasets import cifar10: Imports the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
from keras.models import Sequential: Imports the Sequential model from Keras, which allows creating neural networks layer by layer.
from keras.layers import ...: Imports various layer types (Dense, Dropout, Flatten, Conv2D, MaxPooling2D) that will be used to define the model architecture.
from keras.optimizers import SGD: Imports the Stochastic Gradient Descent optimizer from Keras.
from tensorflow.keras.preprocessing.image import ImageDataGenerator: Imports the ImageDataGenerator class from TensorFlow's Keras implementation, which is used for real-time data augmentation.

Loading and Preprocessing Data:
cifar10.load_data(): Loads the CIFAR-10 dataset, splitting it into training and testing sets (X_train, y_train), (X_test, y_test).
keras.utils.to_categorical: Converts class vectors to binary class matrices. In this case, it converts the labels to one-hot encoded vectors.

Defining the Model:
Sequential(): Initializes a sequential model.
Layers are added one by one to the model:
Input(): Defines the input layer with shape (32, 32, 3), corresponding to 32x32 RGB images.
Convolutional layers (Conv2D) with ReLU activation functions.
Max pooling layers (MaxPooling2D) to down-sample the feature maps.
Dropout layers for regularization to prevent overfitting.
Flatten: Flattens the 2D feature maps into a 1D vector.
Fully connected (Dense) layers with ReLU activation functions.
Output layer with softmax activation for multi-class classification.

Data Augmentation:
ImageDataGenerator: Defines generators for real-time data augmentation during training. Various transformations such as rescaling, shearing, zooming, and horizontal flipping are applied to the images.
Preparing Data Generators:
train_datagen.flow(), test_datagen.flow(): Generates batches of augmented data for training and testing sets.

Compiling the Model:
model.compile(): Configures the model for training by specifying the loss function, optimizer, and evaluation metrics. Here, categorical cross-entropy loss and SGD optimizer are used.

Training the Model:
model.fit(): Trains the model on the training data. The training set is augmented in real-time using the data generators. Training is performed for a specified number of epochs.

Evaluating the Model:
model.evaluate(): Evaluates the trained model on the test set. Test loss and accuracy are printed as the evaluation metrics.

"""
