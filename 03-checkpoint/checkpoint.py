import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
import np_utils


(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
_, img_rows, img_cols =  train_features.shape
num_classes = len(np.unique(train_labels))
num_input_nodes = img_rows*img_cols

# reshape images to column vectors
train_features = train_features.reshape(train_features.shape[0], img_rows*img_cols)
test_features = test_features.reshape(test_features.shape[0], img_rows*img_cols)
# convert class labels to binary class labels
# train_labels = np_utils.to_categorical(train_labels, num_classes)
# test_labels = np_utils.to_categorical(test_labels, num_classes)


def deep_nn():
    # Define a deep neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, input_dim=num_input_nodes))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('softmax'))
    return model


# define a deep neural network
model = deep_nn()
# define optimizer
sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
# checkpoint
outputFolder = './output-mnist'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, \
                             save_best_only=False, save_weights_only=True, \
                             mode='auto', period=10)
callbacks_list = [checkpoint]
# train the model
model.fit(train_features, train_labels, batch_size=128, \
          nb_epoch=80, callbacks=callbacks_list, verbose=0, \
          validation_split=0.2)

init_epoch_num = 81

model_info = model.fit(train_features, train_labels, batch_size=128, \
                       nb_epoch=100, callbacks=callbacks_list, verbose=0, \
                       validation_split=0.2, initial_epoch = init_epoch_num)

model.save('./output-mnist/model.h5')