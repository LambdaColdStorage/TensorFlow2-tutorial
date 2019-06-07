# TensorFlow2-tutorial


# Installation

```
git clone https://github.com/lambdal/TensorFlow2-tutorial.git
cd TensorFlow2-tutorial
virtualenv venv-tf2
. venv-tf2/bin/activate
pip install tf-nightly-gpu-2.0-preview==2.0.0.dev20190526
```


# Tutorials Summary

See individual tutorial's README for details

### 01 Basic Image Classification

A tutorial of Image classification with ResNet. 
* Data pipeline with TensorFlow Dataset API
* Model pipeline with Keras (TensorFlow 2's offical high level API)
* Multi-GPU with distributed strategy
* Customized training with callbacks (TensorBoard, Customized learning schedule)

### 02 Transfer Learning
This tutorial explains how to do transfer learning with TensorFlow 2. We will cover:

* Handling Customized Dataset
* Restore Backbone with Keras's application API
* Restore backbone from disk

### 03 Checkpoint
This tutorial explains how use checkpoint to save and restore model during training.

* Use ```tf.keras.ModelCheckpoint``` to save checkpoint
* Resume training from a pre-saved checkpoint

### 04 Early Stopping
This tutorial explains how to implement early stopping in TensorFlow 2.

* Use ```tf.keras.EarlyStopping``` callback to achieve early stopping.

### 05 Distributed Training Across Multi-Nodes
This tutorial explains how to do distributed training across multiple nodes:

* Code boilerplate for multi-node distributed training
* Run code across multiple machines

