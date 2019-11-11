import math
import os
import csv

from skimage.transform import resize
from skimage.io import imread
import numpy as np
import tensorflow as tf

def preprocess_train(x, y):
  """ Preprocess for training. """
  x = tf.compat.v1.read_file(x) 
  x = tf.image.decode_jpeg(x, dct_method="INTEGER_ACCURATE")

  x = vgg_preprocessing.preprocess_for_train(x,
                                             HEIGHT,
                                             WIDTH)
  return x, y


def preprocess_eval(x, y):
  """ Preprocess for testing. """
  x = tf.compat.v1.read_file(x)
  x = tf.image.decode_jpeg(x, dct_method="INTEGER_ACCURATE")  
  x = vgg_preprocessing.preprocess_for_eval(x,
                                             HEIGHT,
                                             WIDTH)
  return x, y


def augmentation(x, y):
  """ Data augmentation. """
  x = tf.image.resize_with_crop_or_pad(
  	x, HEIGHT + 32, WIDTH + 32)
  x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
  x = tf.image.random_flip_left_right(x)
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

# Custom data generator for large datasets.
# Check https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
class TrainDataGenerator(tf.keras.utils.Sequence):
    """ Custom data generator for Stanford Dogs dataset. """
    def __init__(self, train_file, test_file, batch_size, height, width):
        
        # Load training and testing images and labels. 
        self.train_images_path, self.train_labels = load_csv(train_file)
        self.test_images_path, self.test_labels = load_csv(test_file)

        self.train_images_path = np.array(self.train_images_path)
        self.train_labels = np.array(self.train_labels)

        self.batch_size = batch_size

        self.num_train_samples = len(self.train_images_path)
        self.num_test_samples = len(self.test_images_path)

        self.height = height
        self.width  = width

        self.indices = np.arange(self.num_train_samples)
        np.random.shuffle(self.indices)


    def __len__(self):
        return math.ceil(len(self.train_images_path) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = self.train_images_path[inds]
        batch_y = self.train_labels[inds]

        return np.array([
            resize(imread(file_name), (self.height, self.width))
                for file_name in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)