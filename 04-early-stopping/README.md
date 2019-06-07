During training, weights in the neural networks are updated so the model better fit the training data. Up to a point, the improvement of the performance on the training dataset can benefit the performance on the testing dataset. After that point, overfitting occurs and the testing performance gets worse. Early stopping is introduced to terminate the training before overfitting happens. 


This tutorial explains how early stopping is implemented in TensorFlow 2. 

## Reproduce
```
python resnet_cifar_early_stopping.py
```

## Early Stopping
Early stopping is implemented in TensorFlow via the ```tf.keras.EarlyStopping``` callback function:

```
earlystop_callback = EarlyStopping(
  monitor='val_accuracy', min_delta=0.0001,
  patience=1)
```

```monitor``` keep track of the quantity that is used to decide if the training should be terminated. In this case, we use the validation accuracy. ```min_delta``` is the threshold that triggers the termination. In this case, we require that the accuracy should at least improve 0.0001. ```patience``` is the maximum number of epochs to achieve the threshold. Here we require every epoch has to achieve such improvement.

Now, we can attach the early stop callback and run training with early stopping:

```
model.fit(train_dataset,
          epochs=10, callbacks=[earlystop_callback],
          validation_data=test_dataset,
          validation_freq=1)

Epoch 1/10
390/390 [==============================] - 73s 187ms/step - loss: 2.7133 - accuracy: 0.3300 - val_loss: 6.3186 - val_accuracy: 0.1752
Epoch 2/10
390/390 [==============================] - 39s 100ms/step - loss: 2.2262 - accuracy: 0.4914 - val_loss: 2.5499 - val_accuracy: 0.4358
Epoch 3/10
390/390 [==============================] - 39s 100ms/step - loss: 1.9842 - accuracy: 0.5801 - val_loss: 2.5666 - val_accuracy: 0.4708
Epoch 4/10
390/390 [==============================] - 39s 99ms/step - loss: 1.8144 - accuracy: 0.6333 - val_loss: 2.2643 - val_accuracy: 0.5407
Epoch 5/10
390/390 [==============================] - 39s 99ms/step - loss: 1.6799 - accuracy: 0.6770 - val_loss: 2.1015 - val_accuracy: 0.5841
Epoch 6/10
390/390 [==============================] - 39s 99ms/step - loss: 1.5700 - accuracy: 0.7104 - val_loss: 2.0468 - val_accuracy: 0.6078
Epoch 7/10
390/390 [==============================] - 38s 98ms/step - loss: 1.4697 - accuracy: 0.7388 - val_loss: 2.0628 - val_accuracy: 0.5925
Epoch 00007: early stopping
```

Notice the 7th epoch resulted in better training accuracy but lower validation accuracy. Hence the training terminated at the 7th epoch despite the maximum number of epochs is set to 10.

## Summary

This tutorial explains how early stopping is implemented in TensorFlow. The key lesson is to use ```tf.keras.EarlyStopping``` callback. Early stopping is triggered by monitoring if certain quantity (for example, validation accuracy) has improved over the latest period of time (controlled by the ```patient``` argument).
