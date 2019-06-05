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
