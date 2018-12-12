# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:18:47 2018

@author: luke
"""

## This script processes incoming sound data and classifies it using our tuned VGGish network. It is (heavily) adapted  ##
## from: https://github.com/devicehive/devicehive-audio-analysis/blob/master/audio/processor.py                         ##


## Imports ##

import os
import json
import numpy as np
from keras.models import model_from_json

import vggish_input


## Constants ##

# Probability of classifying as target tune (see ../fine_tune.py)
P_THRESH = 0.9

__all__ = ['WavProcessor', 'format_predictions']


cwd = os.path.dirname(os.path.realpath(__file__))


## Functions ##

def format_predictions(predictions):
    return ', '.join([str(p) for p in predictions])


## Classes ##

class WavProcessor(object):

    _tuned_vggish = None

    def __init__(self, data_dir=os.path.join(os.path.expanduser('~'), 'find-tune', 'data'), low_memory=True):
        # TODO: fix path

        # Load architecture
        with open(os.path.join(data_dir, 'my_vggish_network.json'), 'r') as j:
            model_dict = json.load(j)
        vggish = model_from_json(json.dumps(model_dict))
        
        # Load weights (for low-memory devices, it helps to go layer-by-layer)
        if low_memory:
            # Where each layer's weights are stored on disk
            with open(os.path.join(DATA_DIR, 'layers', 'layer_loc.json'), 'r') as j:
                layer_loc = json.load(j)
            for lyr, fp in layer_loc.items():
                wts = np.load(os.path.join(DATA_DIR, fp))
                    w, b = wts['w'], wts['b']
                    vggish.get_layer(lyr).set_weights([w,b])
        else:
            vggish.load_weights(os.path.join(data_dir, 'my_vggish_network.h5'))

        self._tuned_vggish = vggish

    def get_predictions(self, sample_rate, data):
        # Convert to [-1.0, +1.0]
        # See: wavfile_to_examples at https://github.com/tensorflow/models/blob/master/research/audioset/vggish_input.py
        samples = data / 32768.0
        # Convert to log-mel matrices for each sampled second (3D tensor)
        logmels = vggish_input.waveform_to_examples(samples, sample_rate)
        # Extract predictions from the network (input must be 4D => increase dimension by 1)
        predictions = self._tuned_vggish.predict(logmels[:,:,:,None])[:, 0]

        return predictions > P_THRESH

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


