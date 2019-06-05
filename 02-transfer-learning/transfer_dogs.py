import os
import csv 
import datetime

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

MEAN = [103.939, 116.779, 123.68]

FLAG_RESTORE_FROM_DISK = False

path_home = os.getenv("HOME")
TRAIN_FILE = path_home + "/demo/data/StanfordDogs120/train.csv"
TEST_FILE = path_home + "/demo/data/StanfordDogs120/eval.csv"


def preprocess_train(x, y):
  x = tf.compat.v1.read_file(x)
  x = tf.image.decode_jpeg(x, dct_method="INTEGER_ACCURATE")

  x = vgg_preprocessing.preprocess_for_train(x,
                                             HEIGHT,
                                             WIDTH)
  return x, y


def preprocess_eval(x, y):
  x = tf.compat.v1.read_file(x)
  x = tf.image.decode_jpeg(x, dct_method="INTEGER_ACCURATE")  
  x = vgg_preprocessing.preprocess_for_eval(x,
                                             HEIGHT,
                                             WIDTH)
  return x, y


def augmentation(x, y):
  x = tf.image.resize_with_crop_or_pad(
  	x, HEIGHT + 32, WIDTH + 32)
  x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
  x = tf.image.random_flip_left_right(x)
  return x, y	


def load_csv(file):
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
  initial_learning_rate = BASE_LEARNING_RATE
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate


train_images_path, train_labels = load_csv(TRAIN_FILE)
test_images_path, test_labels = load_csv(TEST_FILE)

NUM_TRAIN_SAMPLES = len(train_images_path)
NUM_TEST_SAMPLES = len(test_images_path)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images_path, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images_path, test_labels))

train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).map(preprocess_train).batch(BS_PER_GPU, drop_remainder=True)
test_dataset = test_dataset.map(preprocess_eval).batch(BS_PER_GPU, drop_remainder=True)




input_shape = (HEIGHT, WIDTH, 3)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = tf.keras.optimizers.SGD()

if FLAG_RESTORE_FROM_DISK:
  backbone = tf.keras.models.load_model('ResNet50.h5')
  backbone.trainable = False
  x = backbone.layers[-3].output	
else:
  backbone = tf.keras.applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (WIDTH, HEIGHT, NUM_CHANNELS))
  backbone.trainable = False
  x = backbone.output

x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                          name='prediction')(x)      
model = tf.keras.models.Model(backbone.input, x, name='model')

model.compile(optimizer=opt,
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule)

model.fit(train_dataset,
          epochs=NUM_EPOCHS,
          validation_data=test_dataset,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback])
model.evaluate(test_dataset)
