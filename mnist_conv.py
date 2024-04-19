import numpy as np
# from keras.datasets import mnist
# from keras.utils import to_categorical

from sklearn.datasets import fetch_openml

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict


def to_one_hot(y_value):
    num_classes = np.max(y_value) + 1
    one_hot_matrix = np.eye(num_classes)[y_value.reshape(-1)]
    return one_hot_matrix.reshape(list(y_value.shape) + [num_classes])


def preprocess_data(x_value, y_value, limit):
    zero_indices = np.where(y_value.values == 0)[0][:limit]
    one_indices = np.where(y_value.values == 1)[0][:limit]
    five_indices = np.where(y_value.values == 5)[0][:limit]

    all_indices = np.hstack((zero_indices, one_indices, five_indices))
    all_indices = np.random.permutation(all_indices)

    x_value = x_value.iloc[all_indices].values.astype("float32") / 255
    x_value = x_value.reshape(len(x_value), 1, 28, 28)

    y_value = y_value.iloc[all_indices].values
    y_value = to_one_hot(y_value)
    y_value = y_value.reshape(len(y_value), 6, 1)

    return x_value, y_value


# Load MNIST data from OpenML
mnist = fetch_openml('mnist_784')

# Separate features and labels
x_data, y_data = mnist['data'], mnist['target']

# Convert string labels to integers
y_data = y_data.astype(int)

# Preprocess the data
x_train, y_train = preprocess_data(x_data[:60000], y_data[:60000], 100)
x_test, y_test = preprocess_data(x_data[60000:], y_data[60000:], 100)

# Print shapes of training and testing data and labels
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 6),
    Sigmoid()
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
