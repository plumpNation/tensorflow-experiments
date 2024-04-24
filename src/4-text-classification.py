import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

print()
print("************************************************")
print()
print("TensorFlow version:", tf.__version__)
print()
print("************************************************")
print()

# sentiment analysis IMDB dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# only download if the dataset is not already downloaded
if not os.path.exists('aclImdb'):
  dataset = tf.keras.utils.get_file(
    "aclImdb_v1",
    url,
    untar=True,
    cache_dir='.',
    cache_subdir='',
  )
else:
  dataset = os.path.join('.', 'aclImdb')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

print("Dataset directory:", os.listdir(dataset_dir))

print()

train_dir = os.path.join(dataset_dir, 'train')

print("Train directory:", os.listdir(train_dir))

print()

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')

with open(sample_file) as f:
  print(f.read())

# We don't need this directory
remove_dir = os.path.join(train_dir, 'unsup')

if os.path.exists(remove_dir):
  shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

# Create a validation set using an 80:20 split of the training data
# by using the validation_split argument below.
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

# print out a few examples of the training dataset
# for text_batch, label_batch in raw_train_ds.take(1):
#   for i in range(3):
#     print("Review", text_batch.numpy()[i])
#     print("Label", label_batch.numpy()[i])

# The labels are 0 or 1. To see which of these correspond to positive
# and negative movie reviews, you can check the class_names property
# on the dataset.
# print("Label 0 corresponds to", raw_train_ds.class_names[0])
# print("Label 1 corresponds to", raw_train_ds.class_names[1])

# When using the validation_split and subset arguments, make sure to either
# specify a random seed, or to pass shuffle=False, so that the validation
# and training splits have no overlap.

# create a validation set
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

# create a test set
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

# Now we need to go through the data and standardize, tokenize,
# and vectorize. We will do all of this using the helpful
# preprocessing.TextVectorization layer.

# To prevent training-testing skew (also known as training-serving skew),
# it is important to preprocess the data identically at train and
# test time. To facilitate this, the TextVectorization layer can be
# included directly inside your model, as shown later in this tutorial.

# Standardization refers to preprocessing the text, typically to remove
# punctuation or HTML elements to simplify the dataset.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')

  return tf.strings.regex_replace(
    stripped_html,
    '[%s]' % re.escape(string.punctuation),
    ''
  )

max_features = 10000
# will truncate or pad sequences to exactly 250 tokens
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
# It's important to only use your training data when calling adapt
# (using the test set would leak information).
vectorize_layer.adapt(train_text)

# Let's create a function to see the result of using the TextVectorization
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)

  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))

first_review, first_label = text_batch[0], label_batch[0]

print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# Apply the TextVectorization layer to the datasets
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# pick up again from here:
# https://www.tensorflow.org/tutorials/keras/text_classification#configure_the_dataset_for_performance
