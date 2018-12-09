# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 21:32:49 2018

@author: luke
"""

## Imports ##

import os
import numpy as np
import tensorflow as tf
import vggish_slim
import vggish_params as params

from keras.models import Sequential, load_model
from keras import layers as lyr
from keras.optimizers import SGD


## Constants ##

VGGISH_DIR = os.path.join(os.path.expanduser('~'), 'tf-models','research','audioset') # TODO: at setup
checkpoint_path = os.path.join(VGGISH_DIR, 'vggish_model.ckpt')

# The fold to concentrate on: see models_with_embedding.py
FOLD_NUM = 0
DATA_DIR = os.path.join(os.path.expanduser('~'), 'find-tune', 'data', 'fold%d' % FOLD_NUM)


## Main ##

# Load training and val set for this fold
data_tr = np.load(os.path.join(DATA_DIR, 'foldwise_data_tr.npz'))
X_tr, y_tr, c_tr, s_tr, ids_tr, L_tr = data_tr['X'], data_tr['y'], data_tr['c'], data_tr['s'], data_tr['i'], data_tr['L']
data_va = np.load(os.path.join(DATA_DIR, 'foldwise_data_va.npz'))
X_va, y_va, c_va, s_va, ids_va, L_va = data_va['X'], data_va['y'], data_va['c'], data_va['s'], data_va['i'], data_va['L']


# Extract the VGGish variables from the checkpoint file
with tf.Graph().as_default(), tf.Session() as session:

    vggish_slim.define_vggish_slim(training=False)
    vggish_var_names = [v.name for v in tf.global_variables()]
    vggish_vars = [v for v in tf.global_variables() if v.name in vggish_var_names]

    saver = tf.train.Saver(vggish_vars, name='vggish_load_pretrained', write_version=1)
    saver.restore(session, checkpoint_path)

    model_wts = {}
    for var in vggish_vars:
        try:
            model_wts[var.name] = var.eval()
        except:
            print("For var={}, an exception occurred".format(var.name))


# Reconstruct the network in Keras notation
# See: https://stackoverflow.com/questions/44466066/how-can-i-convert-a-trained-tensorflow-model-to-keras/53638524#53638524
# For network definition, see vggish_slim.define_vggish_slim
vggish = Sequential()
conv1  = lyr.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv1',
                    input_shape=(params.NUM_FRAMES, params.NUM_BANDS, 1))
vggish.add(conv1)
# Set its weights, defined in model_vars
conv1.set_weights([model_wts['vggish/conv1/weights:0'], model_wts['vggish/conv1/biases:0']])
conv1.trainable = False # Freeze this layer

pool1  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool1')
vggish.add(pool1)
# Note: no weights for pooling layers

conv2  = lyr.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv2')
vggish.add(conv2)
conv2.set_weights([model_wts['vggish/conv2/weights:0'], model_wts['vggish/conv2/biases:0']])
conv2.trainable = False

pool2  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool2')
vggish.add(pool2)

conv3_1= lyr.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv3_1')
vggish.add(conv3_1)
conv3_1.set_weights([model_wts['vggish/conv3/conv3_1/weights:0'], model_wts['vggish/conv3/conv3_1/biases:0']])
conv3_1.trainable = False

conv3_2= lyr.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv3_2')
vggish.add(conv3_2)
conv3_2.set_weights([model_wts['vggish/conv3/conv3_2/weights:0'], model_wts['vggish/conv3/conv3_2/biases:0']])
conv3_2.trainable = False

pool3  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool3')
vggish.add(pool3)

conv4_1= lyr.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv4_1')
vggish.add(conv4_1)
conv4_1.set_weights([model_wts['vggish/conv4/conv4_1/weights:0'], model_wts['vggish/conv4/conv4_1/biases:0']])
conv4_1.trainable = False

conv4_2= lyr.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv4_2')
vggish.add(conv4_2)
conv4_2.set_weights([model_wts['vggish/conv4/conv4_2/weights:0'], model_wts['vggish/conv4/conv4_2/biases:0']])
conv4_2.trainable = False

pool4  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool4')
vggish.add(pool4)

vggish.add(lyr.Flatten()) # TODO: check channels_first or _last

fc1_1  = lyr.Dense(4096, activation='relu', name='fc1_1')
vggish.add(fc1_1)
fc1_1.set_weights([model_wts['vggish/fc1/fc1_1/weights:0'], model_wts['vggish/fc1/fc1_1/biases:0']])
fc1_1.trainable = False

fc1_2  = lyr.Dense(4096, activation='relu', name='fc1_2')
vggish.add(fc1_2)
fc1_2.set_weights([model_wts['vggish/fc1/fc1_2/weights:0'], model_wts['vggish/fc1/fc1_2/biases:0']])
fc1_2.trainable = False

# Note: first trainable layer in network
fc2    = lyr.Dense(params.EMBEDDING_SIZE, activation='relu', name='fc2')
vggish.add(fc2)
fc2.set_weights([model_wts['vggish/fc2/weights:0'], model_wts['vggish/fc2/biases:0']])
fc2.trainable = True

# Append our model on top of the VGGish embedding layer

# Load keras model representing the final part of classification task
model_head = load_model(os.path.join(DATA_DIR, 'nn_fold%d.model') % FOLD_NUM)

# Make a copy of fc_last layer, then add it to vggish
h_last = model_head.get_layer('fc_last').output_shape[-1]
a_last = model_head.get_layer('fc_last').activation
fc_last= lyr.Dense(h_last, activation=a_last, name='fc_last')
vggish.add(fc_last)
# We need to set the weights
fc_last.set_weights(model_head.get_layer('fc_last').get_weights())

# Make a copy of fc_last layer, then add it to vggish
a_cfy = model_head.get_layer('classify').activation
classify = lyr.Dense(1, activation=a_cfy, name='classify')
vggish.add(classify)
classify.set_weights(model_head.get_layer('classify').get_weights())

optzr = SGD(lr=params.LEARNING_RATE/10)
vggish.compile(optzr, loss='binary_crossentropy')

