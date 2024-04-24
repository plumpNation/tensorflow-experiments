import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0