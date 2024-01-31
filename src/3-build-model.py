import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Load and prepare the MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The pixel values, which are 0-255, have to be scaled to the range 0-1
# before feeding them to the neural network model.
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten: Flattens the input.
  # Does not affect the batch size.
  tf.keras.layers.Flatten(input_shape=(28, 28)),

  # Dense: A layer of neurons. Each neuron receives input from all the
  # neurons in the previous layer, thus densely connected.
  tf.keras.layers.Dense(128, activation='relu'),
  # ReLU (Rectified Linear Unit): Returns the input if it’s positive;
  # otherwise, it returns zero. It’s widely used due to its simplicity
  # and efficiency.


  # Dropout: A form of regularization that randomly sets input units to 0
  # with a frequency of rate at each step during training time, which helps
  # prevent overfitting.
  tf.keras.layers.Dropout(0.2),

  # Softmax: A function that provides probabilities for each possible
  # output class.
  tf.keras.layers.Dense(10)
])

# For each example, the model returns a vector of logits
# or log-odds scores, one for each class.
predictions = model(x_train[:1]).numpy()

# The tf.nn.softmax function converts these logits
# to probabilities for each class.
tf.nn.softmax(predictions).numpy()

# The losses.SparseCategoricalCrossentropy loss takes a vector of logits
# and a True index and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
# This loss is equal to the negative log probability of the true class:
# It is zero if the model is sure of the correct class.
loss_fn(y_train[:1], predictions).numpy()

# before training, we need to compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=5)

# check the model's performance on the training dataset
model.evaluate(x_test,  y_test, verbose=2)

# to get the probability model, wrap the trained model and attach
# the softmax layer to it.
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])