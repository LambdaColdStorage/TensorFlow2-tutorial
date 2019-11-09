import os
import numpy as np
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint

import resnet


NUM_GPUS = 1
BS_PER_GPU = 128

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 5), (0.01, 10)]

NUM_EPOCHS_1 = 3
NUM_EPOCHS_2 = 5
INIT_EPOCH_2 = 1

def normalize(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y 


def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate


# Custom data generator for large datasets.
# Check https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
class DataGenerator(Sequence):

    def __init__(self, path_to_images, path_to_labels, batch_size):
        self.path_to_images, self.path_to_labels = path_to_images, path_to_labels
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
                for file_name in batch_x]), np.array(batch_y)




(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).map(augmentation).map(normalize).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_dataset = test_dataset.map(normalize).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)


input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
img_input = tf.keras.layers.Input(shape=input_shape)

model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)

# define optimizer
sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# checkpoint
outputFolder = './output-cifar'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint_callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, \
                             save_best_only=False, save_weights_only=False, \
                             mode='auto', save_frequency=1)

# train the model for the first time
model.fit(train_dataset,
          epochs=NUM_EPOCHS_1, callbacks=[checkpoint_callback],
          validation_data=test_dataset,
          validation_freq=1)

# resume training from the checkpoint
model_info = model.fit(train_dataset,
                       epochs=NUM_EPOCHS_2, callbacks=[checkpoint_callback],
                       validation_data=test_dataset,
                       validation_freq=1,
                       initial_epoch = INIT_EPOCH_2)

model.evaluate(test_dataset)

model.save('./output-cifar/model.h5')