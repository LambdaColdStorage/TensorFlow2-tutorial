import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler

import resnet # Import ResNet from resnet.py


NUM_GPUS = 2 # Number of GPUs
BS_PER_GPU = 128 # Batch size per GPU
NUM_EPOCHS = 60 # Number of epochs

HEIGHT = 32 # Image height
WIDTH = 32 # Image width
NUM_CHANNELS = 3 # Number of channels (RGB -> 3)
NUM_CLASSES = 10 # Number of classes (CIFAR10 -> 10)
NUM_TRAIN_SAMPLES = 50000 # Number of training samples

BASE_LEARNING_RATE = 0.1 # Initial learning rate
LR_SCHEDULE = [(0.1, 30), (0.01, 45)] # Multiply the learning rate with 0.1 and 0.01 at epochs 30 and 45, respectively.


def normalize(x, y):
  """ Preprocess function: Image normalization """
  x = tf.image.per_image_standardization(x) # Linearly scales each image in x to have mean 0 and variance 1.
  return x, y


def augmentation(x, y):
    """ Preprocess function: Image augmentation """
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8) # Resizes an image to a target width and height by either centrally cropping the image or padding it evenly with zeros.
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS]) # Slices a shape size portion out of value at a uniformly chosen offset
    x = tf.image.random_flip_left_right(x) # Randomly flip an image horizontally
    return x, y	


def schedule(epoch):
  """ Learning rate scheduler 
      It takes an epoch index as input (integer, indexed from 0) and returns a new learning rate as output (float).
  """
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE: # Multiply learning rate when reach to target epoch.
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch) # Log for the tensorboard
  return learning_rate


# Load CIFAR10 dataset.
(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data() 

# Feed data into your models using an input pipeline to ensure that the GPU has never waits for new data to come in.
train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Operations that rely on a random seed actually derive it from two seeds: the graph-level and operation-level seeds. This sets the graph-level seed.
tf.random.set_seed(22)

# Preprocess data.
train_dataset = train_dataset.map(augmentation).map(normalize).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_dataset = test_dataset.map(normalize).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)

# Set the input layer.
input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
img_input = tf.keras.layers.Input(shape=input_shape)

# Set the SGD optimizer.
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

# Load and compile the model for 1 GPU only.
if NUM_GPUS == 1:
    model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
    model.compile(
              optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Load and compile the modal for multiple GPUs.
else:
    # Mirrors the Keras model onto multiple GPUs on a single machine.
    # The speedup of training/inference is achieved by splitting the input batches so they are spread evenly across the devices.
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # The threads will use a variable creator set by mirrored_strategy, and will enter its "cross-replica context".
    with mirrored_strategy.scope():
      model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
      model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])  

# Log directory.
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Creates a summary file writer for the given log directory.
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()

# Create tensorboard callback.
tensorboard_callback = TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

# "schedule" function takes an epoch index as input (integer, indexed from 0) and returns a new learning rate as output (float).
lr_schedule_callback = LearningRateScheduler(schedule)

# Fit function.
model.fit(train_dataset,
          epochs=NUM_EPOCHS,
          validation_data=test_dataset,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback])

# Evaluate the model.          
model.evaluate(test_dataset)

# Save the model.
model.save('model.h5')

# Load the model back.
new_model = keras.models.load_model('model.h5')

# Evaluate the model. 
new_model.evaluate(test_dataset)
