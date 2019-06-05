# TensorFlow 2 Tutorial I: Image Classification Basics

TensorFlow 2 is coming. If you are used to coding in TensorFlow 1.x, things are about to change. Coding in TensorFlow 2.0 is imperative, free form sessions, and a lot of Keras.

This tutorial explains the basics of TensorFlow 2 with image classification as an example. We will cover:
* Data Pipeline
* Model Pipeline
* Multiple-GPU
* Callbacks

## Reproduce the tutorial

```
python resnet_cifar.py
```

## Data

Machine learning solutions typically start from a data pipeline, which mainly concerns three things:
* Load data from storage
* An interface for feeding data into the training pipeline
* Miscellaneous tasks such as preprocessing, shuffling and batching

**Load Data**

For image classification, it is common to read the images and labels into data arrays (numpy ndarrays). The ```Oth``` dimension of these arrays is equal to the total number of samples. Customized data usually needs a customized function. In this tutorial, we leverage Keras's ```load_data``` function to read the popular CIFAR10 dataset:

```
(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()
```

We can verify the type and shape of these data arrays:

```
print(type(x), type(y))
print(x.shape, y.shape)

(<type 'numpy.ndarray'>, <type 'numpy.ndarray'>)
((50000, 32, 32, 3), (50000, 1))
```

**Interface**

Although it is possible to directly feed numpy ndarrays to the training loop, doing so makes it difficult to incorporate data augmentation, which is randomized on the fly. What is needed here is an interface that can handle both the data and the preprocesses applied to the data.

TensorFlow provides a very sophisticated [Dataset API]((https://www.tensorflow.org/api_docs/python/tf/data/Dataset)) for this purpose. A TensorFlow Dataset essentially presents two things
* A collection of elements (nested structures of tensors) 
* A "logical plan" of transformations that act on those elements, where we can apply the necessary preprocessing jobs.

A TensorFlow dataset can be directly created from the data arrays. We can use the ```take(1)``` to fetch the first element of the dataset, which is a tuple that contains the image tensor and the label tensor:

```
train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
for image, label in train_dataset.take(1):
  print(image.shape, label.shape) 

(TensorShape([32, 32, 3]), TensorShape([1]))
```

These are the first 20 images in the dataset:

![cifar10-1](https://lambdalabs.com/blog/content/images/2019/05/cifar10-1.png)

**Miscellaneous**

Thanks to TensorFlow Dataset's ability to handle transformations, we can now add the miscellaneous preprocess jobs. 

Let's first add data augmentation: We pads four black pixels to the border of the image, then randomly crops 32x32 regions from the padded image, and finally randomly flips the image horizontally. TensorFlow Dataset uses the ```map``` function to apply the augmentation to each element. 

```
def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y    
train_dataset = train_dataset.map(augmentation)
```

These are the first 20 images after the augmentation:

![augmentation](https://lambdalabs.com/blog/content/images/2019/05/augmentation.png)

Data augmentation is frequently used to "inflate" the training data and improves the generalization performance. Notice data augmentation should only be applied to the training set because the randomized nature of the data augmentation will make the inference undesirable non-deterministic. 

Next, we randomly shuffle the dataset. TensorFlow Dataset has a member ```shuffle``` function, all we need to do is append it to the Dataset object:

```
train_dataset = train_dataset.map(augmentation).shuffle(50000)
```

Notice, for perfect shuffling, a buffer size should be greater than or equal to the full size of the dataset (50000 in this case). Below are the 20 images from the Dataset after shuffling. They are not the same image as the first 20 images stored in the original dataset:

![shuffle](https://lambdalabs.com/blog/content/images/2019/05/shuffle.png)


It is also common practice to normalize the data, for example, by linearly scaling the image to have zero mean and unit variance. This can be achieved by mapping a customized ```normalize``` function to the dataset. 
```
def normalize(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y
```

Last but not the least, we need to ```batch``` the data, and set ```drop_remainder``` to ```True``` in case the number of samples in the dataset is not divisible by the ```batch_size```.

```
train_dataset = train_dataset.map(augmentation).map(normalize).shuffle(50000).batch(128, drop_remainder=True)
```


Now we have a complete data pipeline. Next, we will define the model and create a training pipeline.


## Model

**Define a Model**

TensorFlow 2 chooses Keras as its high-level API. Keras has two ways to define a model: Sequential and Functional. 

```
from tf.keras.models import Sequential, Model
from tf.keras.layers import Input, Conv2, MaxPooling2D, Flatten, Dense

# Sequential API
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

# Functional API
inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=x)
```

The above two ways define the same model. The main difference is the Sequential API requires its first layer to be provided with ```input_shape```; the Functional API requires its first layer to be a ```tf.keras.layers.Input``` layer, and needs to call the ```tf.keras.models.Model``` constructor at the end. 

Sequential API requires less typing, but functional API is more flexible -- it allows a model to be non-sequential. For example, to have the skip connection in ResNet. This tutorial adapts TensorFlow's official Keras implementation of [ResNet](https://github.com/tensorflow/models/blob/master/official/resnet/keras/resnet_cifar_model.py), which uses the functional API. 

```
input_shape = (32, 32, 3)
img_input = Input(shape=input_shape)
model = resnet_cifar_model.resnet56(img_input, classes=10)
```

A Keras model needs to be compiled before being trained. The compilation of the model essentially defines three things: the **loss function**, the **optimizer** and the **metrics** for evaluation:

```
model.compile(
          loss='sparse_categorical_crossentropy',
          optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
          metrics=['accuracy'])
```

Notice we use ```sparse_categorical_crossentropy``` and ```sparse_categorical_accuracy``` here because each label is represented by a single integer (index of the class). One should use ```categorical_crossentropy``` and ```categorical_accuracy``` if a one-hot vector represents each label.

**Train and Evaluation**

Keras uses the ```fit``` API to train a model. Optionally, one can test the model on a validation dataset at every ```validation_freq``` training epoch. Notice we use the test dataset for validation only because CIFAR10 does not natively provide a validation set. More rigorous validation of the model should be conducted on a set of data split from the training set.
```
model.fit(train_dataset,
          epochs=60,
          validation_data=test_dataset,
          validation_freq=1)
```

Notice in this example, the ```fit``` function takes TensorFlow Dataset objects (```train_dataset``` and ```test_dataset```). As previously mentioned, it can also take numpy ndarrays as the input. The downside of using arrays is the lack of flexibility to apply transformations on the dataset. 
```
model.fit(x, y,
          batch_size=128, epochs=5, shuffle=True,
          validation_data=(x_test, y_test))
```

The evaluation of the model calls the ```evaluate``` function with the test dataset:
```
model.evaluate(test_dataset)
```

**Save and Restore**

Keras models have native support for saving/Restoring model definition and weights -- all you need to do is calling the ```save``` and ```load_model``` APIs. In our case of ResNet, it also saves the moving statistics of the batch normalization layer:
```
model.save('model.h5')

new_model = keras.models.load_model('model.h5')

# Gives the same accuracy as model
new_model.evaluate(test_dataset)
```

However, there is one caveat: model created by [sub-classing](https://keras.io/models/about-keras-models/#model-subclassing) can not be saved by ```model.save()```. This is because sub-classing defines model's topology as Python code (rather than as a static graph of layers). That means the model's topology cannot be inspected or serialized. As a result, the following methods and attributes are not available for subclassed models:

```
model.inputs and model.outputs.
model.to_yaml() and model.to_json()
model.get_config() and model.save()
```

## Multi-GPU

So far, we have shown how to use TensorFlow's Dataset API to create data pipeline, and how to use the Keras API to define the model and conduct the training and evaluation. The next step is to make the code run with multiple GPUs.

In fact, Tensorflow 2 has made it very easy to convert the single-GPU implementation to run with multiple GPUs. All you need to do is define a distribute strategy and create the model under the strategy's scope: 

```
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = resnet.resnet56(classes=NUM_CLASSES)
    model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  
```

We use ```MirroredStrategy``` here, which supports synchronous distributed training on multiple GPUs on one machine. By default, it uses NVIDIA NCCL as the all-reduce implementation.

Notice the data pipeline need to scale batch size accordingly to get proper utilization of multiple-GPUs.

```
train_dataset = train_dataset.map(preprocess).shuffle(50000).batch(BS_PER_GPU*NUM_GPUS)
test_dataset = test_dataset.map(preprocess).batch(BS_PER_GPU*NUM_GPUS)
```

## Callbacks

Often we need to customize the training. For example, log statistics during the training for debugging and optimizing the network; implement bespoken learning rate schedule to improve the efficiency of training. In TensorFlow 2, customization of the training is supported via callbacks.


**Tensorboard** 

TensorBoard is mainly used for logging and visualize information during training. It is handy for examing the performance of the model. The way to add TensorBoard support in Keras is via the ```tensorflow.keras.callbacks.TensorBoard``` callback function:
```
from tensorflow.keras.callbacks import TensorBoard

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(
    log_dir=log_dir, update_freq='batch', histogram_freq=1)
  
model.fit(...,
          callbacks=[tensorboard_callback])
```

In the above example, we first create a TensorBoard callback that record data for each training step (via ```update_freq=batch```), then attach this callback to the ```fit``` function. TensorFlow will generate ``tfevents`` files, which can be visualized with TensorBoard. For example, this is the visualization of classification accuracy during  the training (blue is the training accuracy, red is the validation accuracy):

![Screenshot-from-2019-05-28-16-20-58-1-1](https://lambdalabs.com/blog/content/images/2019/05/Screenshot-from-2019-05-28-16-20-58-1-1.png)

**Learning Rate Schedule**

Often we would like to have the finest control of learning rate as the training progresses. The customized learning rate schedule can be implemented as callback functions. Here we create a customized ```schedule``` function that decreases the learning rate using a step function (at 30th epoch and 45th epoch). This schedule is converted to a ```keras.callbacks.LearningRateScheduler``` and attached to the ```fit``` function.

```
from tensorflow.keras.callbacks import LearningRateScheduler

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

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
  
lr_schedule_callback = LearningRateScheduler(schedule)

model.fit(...,
          callbacks=[..., lr_schedule_callback])
```

This is the statistics of the customized learning rate during a 60-epochs training:


![Screenshot-from-2019-05-28-16-19-23-1](https://lambdalabs.com/blog/content/images/2019/05/Screenshot-from-2019-05-28-16-19-23-1.png)


## Summary

This tutorial explains the basic of TensorFlow 2.0 with image classification as an example. We covered
* Data pipeline with TensorFlow 2's dataset API
* Train, evaluation, save and restore models with Keras (TensorFlow 2's official high-level API)
* Multiple-GPU with distributed strategy
* Customized training with callbacks