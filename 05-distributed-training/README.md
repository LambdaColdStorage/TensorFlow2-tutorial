Distributed training allows scaling up deep learning task so bigger models can be learned from more extensive data. In a previous tutorial, we discussed how to use ```MirroredStrategy``` to achieve multi-GPU training within a single node (physical machine). In this tutorial, we will explain how to do distributed training across multiple nodes:

* Code boilerplate for multi-node distributed training
* Run code across multiple machines

## Reprodution

Change the ```TF_CONFIG``` accroding to your cluster setup. 


```
# On the first node
python worker_0.py

# On the second node
python worker_1.py
```

## Code Boilerplate

Similar to multi-GPU training within a single node, multi-node training also uses a distributed strategy. In this case, ```tf.distribute.experimental.MultiWorkerMirroredStrategy```. Differently, multi-node training further requires a ```TF_CONFIG``` environment variable to be set. Notice, such an environment variable will be slightly different on each node. For example, this is the setting on ```work 0``` in a two-node distributed training job: 

```
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'worker': ["10.1.10.58:12345", "10.1.10.250:12345"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```

The ```cluster``` field is the same across all nodes. It describes how the cluster is set. In this case, our cluster only has two worker nodes, whose ```IP:port``` information is listed in the ```worker``` field. The ```task``` field varies from node to node. It specifies the type and index of the node, which can be used to fetch details from the ```cluster``` field. In this case, this config file indicates the training job runs on worker 0, which is ```"10.1.10.58:12345"```

We need to customize this python snippet for each node. So the second node will have ```'task': {'type': 'worker', 'index': 1}```. 

Then we need to create the distributed strategy:

```
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
```

Notice this line has to be right after the definition of ```TF_CONFIG``` and before the definition of data pipeline and model. Otherwise, a ```Collective ops must be configured at program startup``` Error will be triggered. 

Last bit of the code boilplate is to define the model under the strategy scope:

```
with strategy.scope():
  model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
  model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']) 
model.fit(train_dataset,
          epochs=NUM_EPOCHS)
```

## Run the training

To run distributed training, the training scripts need to be available on all nodes. To make it clean, let's have two scripts, one for each node. The only difference between these two scripts is the ```task``` field in the ```TF_CONFIG```:

```
# worker_0.py
'task': {'type': 'worker', 'index': 0}

# worker_1.py
'task': {'type': 'worker', 'index': 1}

```

In the meantime, make sure the nodes can ssh into each other without the request of the password. An easy way to do this is to use [ssh key](https://debian-administration.org/article/530/SSH_with_authentication_key_instead_of_password) for authentication.

Last but not least, run the script simultaneously on both nodes. Notice one can run the script in a python virtual environment if that is where TensorFlow is installed. 

```
# On the first node
python worker_0.py

# On the second node
python worker_1.py
```

The training is now distributed across multiple nodes. The output of the two nodes are synchronized as the ```Mirrored``` strategy is used.


## Summary

This tutorial explains how to do distributed training in TensorFlow 2. The key is to set up the ```TF_CONFIG``` environment variable and use the ```MultiWorkerMirroredStrategy``` to scope the model definition. In this tutorial, we need to run the training script manually on each node with custimized ```TF_CONFIG```. One can see this will quickly become tedious when the number of nodes increases. 

There are more advanced ways to deploy a distributed training job across a large number of nodes, for example, Horovod, or Kubernetes with TF-flow. We will have another tutorial dedicated to that topic.

