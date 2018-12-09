# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 21:32:49 2018

@author: luke
"""

## Imports ##

import os
import tensorflow as tf
import vggish_slim
import vggish_params as params

from keras.models import Sequential
from keras import layers as lyr
from keras.optimizers import Adam


## Constants ##

VGGISH_DIR= os.path.join(os.path.expanduser('~'), 'tf-models','research','audioset')
checkpoint_path = os.path.join(VGGISH_DIR, 'vggish_model.ckpt')


## Main ##

# Extract the VGGish variables from the checkpoint file
with tf.Graph().as_default(), tf.Session() as session:

    vggish_slim.define_vggish_slim(training=False)
    vggish_var_names = [v.name for v in tf.global_variables()]
    vggish_vars = [v for v in tf.global_variables() if v.name in vggish_var_names]

    saver = tf.train.Saver(vggish_vars, name='vggish_load_pretrained', write_version=1)
    saver.restore(session, checkpoint_path)

    model_vars = {}
    for var in vggish_vars:
        try:
            model_vars[var.name] = var.eval()
        except:
            print("For var={}, an exception occurred".format(var.name))


# Reconstruct the network in Keras notation
# See: https://stackoverflow.com/questions/44466066/how-can-i-convert-a-trained-tensorflow-model-to-keras/53638524#53638524
# For network definition, see vggish_slim.define_vggish_slim
vggish = Sequential()
conv1  = lyr.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv1',
                    input_shape=(params.NUM_FRAMES, params.NUM_BANDS, 1))
pool1  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool1')
vggish.add(conv1)
vggish.add(pool1)

conv2  = lyr.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv2')
pool2  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool2')
vggish.add(conv2)
vggish.add(pool2)

conv3_1= lyr.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv3_1')
conv3_2= lyr.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv3_2')
pool3  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool3')
vggish.add(conv3_1)
vggish.add(conv3_2)
vggish.add(pool3)

conv4_1= lyr.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv4_1')
conv4_2= lyr.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv4_2')
pool4  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool4')
vggish.add(conv4_1)
vggish.add(conv4_2)
vggish.add(pool4)

vggish.add(lyr.Flatten()) # TODO: check channels_first or _last

fc1_1  = lyr.Dense(4096, activation='relu', name='fc1_1')
fc1_2  = lyr.Dense(4096, activation='relu', name='fc1_2')
vggish.add(fc1_1)
vggish.add(fc1_2)

fc2    = lyr.Dense(params.EMBEDDING_SIZE, activation='relu', name='fc2')
vggish.add(fc2)

# TODO: add another Dense layer corresponding to the new binary classification task

optzr  = Adam(lr=0.00000001) # TODO: make this appropriate, esp. lr
vggish.compile(optzr, loss='binary_crossentropy')


# TODO: set the weights within the layers with, e.g.:
# conv1.set_weights([model_vars['vggish/conv1/weights:0'], model_vars['vggish/conv1/biases:0']])
# Possible to shortcut this if we can prepare a set of weights like vggish.get_weights(), but beware biases