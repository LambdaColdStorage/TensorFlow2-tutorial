import os
import csv 
import datetime

from skimage.transform import resize
from skimage.io import imread
import numpy as np

import tensorflow as tf
import vgg_preprocessing

HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
NUM_CLASSES = 120

BS_PER_GPU = 32
NUM_EPOCHS = 10

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 5), (0.01, 8)]
L2_WEIGHT_DECAY = 2e-4

# Mean values of the dataset.
MEAN = [103.939, 116.779, 123.68]

FLAG_RESTORE_FROM_DISK = False

# Path to train and test data. They are csv files.
path_home = os.getenv("HOME")
TRAIN_FILE = path_home + "/demo/data/StanfordDogs120/train.csv"
TEST_FILE = path_home + "/demo/data/StanfordDogs120/eval.csv"


def preprocess_train(x, y):
  """ Preprocess for training. """
  x = vgg_preprocessing.preprocess_for_train(x,
                                             HEIGHT,
                                             WIDTH)
  return x, y


def preprocess_eval(x, y):
  """ Preprocess for testing. """
  x = vgg_preprocessing.preprocess_for_eval(x,
                                             HEIGHT,
                                             WIDTH)
  return x, y



def load_csv(file):
  """ Load csv file. """
  dirname = os.path.dirname(file)
  images_path = []
  labels = []
  with open(file) as f:
    parsed = csv.reader(f, delimiter=",", quotechar="'")
    for row in parsed:
      images_path.append(os.path.join(dirname, row[0]))
      labels.append(int(row[1]))
  return images_path, labels


def schedule(epoch):
  """ Schedule learning rate. """
  initial_learning_rate = BASE_LEARNING_RATE
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

# Load training and testing images and labels. 
train_images_path, train_labels = load_csv(TRAIN_FILE)
test_images_path, test_labels = load_csv(TEST_FILE)

NUM_TRAIN_SAMPLES = len(train_images_path)
NUM_TEST_SAMPLES = len(test_images_path)

def train_generator():
  """ Generator for training samples.
  We should use yield when we want to iterate over a sequence, but don't want to store the entire sequence in memory.
  """
  for image_path, label in zip(train_images_path, train_labels):
      X = np.array(resize(imread(image_path), (HEIGHT, WIDTH)))
      y = label
      
      yield X, y
          
def test_generator():
  """ Generator for test samples.
  We should use yield when we want to iterate over a sequence, but don't want to store the entire sequence in memory.
  """
  for image_path, label in zip(test_images_path, test_labels):
      X = np.array(resize(imread(image_path), (HEIGHT, WIDTH)))
      y = label
      
      yield X, y


# Feed data into your models using generators.
# You can handle large datasets since samples are created on the fly.
train_dataset = tf.data.Dataset.from_generator(generator = train_generator,
                                            output_types = (tf.float32, tf.int8),
                                            output_shapes=(tf.TensorShape([HEIGHT, WIDTH, 3]), tf.TensorShape([])))
test_dataset = tf.data.Dataset.from_generator(generator = test_generator,
                                            output_types = (tf.float32, tf.int8),
                                            output_shapes=(tf.TensorShape([HEIGHT, WIDTH, 3]), tf.TensorShape([])))



# Preprocess data.
train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).map(preprocess_train).batch(BS_PER_GPU, drop_remainder=True)
test_dataset = test_dataset.map(preprocess_eval).batch(BS_PER_GPU, drop_remainder=True)



# Input settings.
input_shape = (HEIGHT, WIDTH, 3)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = tf.keras.optimizers.SGD()

# Restore from the disk or use existing ResNet50.
if FLAG_RESTORE_FROM_DISK:
  backbone = tf.keras.models.load_model('ResNet50.h5')

  # Backbone is not trainable.
  backbone.trainable = False
  x = backbone.layers[-3].output	

else:
  # Use the avaliable model in Keras but dont get top layer.
  # Since the top layer is classification layer.
  backbone = tf.keras.applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (WIDTH, HEIGHT, NUM_CHANNELS))

  # Backbone is not trainable.
  backbone.trainable = False
  x = backbone.output

# Add custom layers.
x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                          name='prediction')(x)      
model = tf.keras.models.Model(backbone.input, x, name='model')

# Compile the model.
model.compile(optimizer=opt,
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

# Set logging settings.
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model.
model.fit(train_dataset,
          epochs=NUM_EPOCHS,
          validation_data=test_dataset,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback])

# Evaluate the model.          
model.evaluate(test_dataset)
