import tensorflow as tf

model = tf.keras.applications.ResNet50(weights = "imagenet", include_top=True)
model.save('ResNet50.h5')