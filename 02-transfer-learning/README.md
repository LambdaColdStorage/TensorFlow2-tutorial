# TensorFlow 2 Tutorial 2: Transfer Learning

This tutorial explains how to do transfer learning with TensorFlow 2. We will cover:
* Handeling Customized Dataset
* Restore Backbone with Keras's application API
* Restore Backbone from disk

## Reproduce the tutorial

```
python demo/download_data.py \
--data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
--data_dir=~/demo/data

python transfer_dogs.py
```

## Customized Data

In this tutorial we will classify images in the [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) dataset. We re-organized the raw data with a CSV file. Below is a snippet of the CSV file. The first column is the path to the image, the second column is the class id:

