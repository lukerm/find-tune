# Copyright (C) 2017 DataArt
# Modifications copyright (C) 2018 lukerm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been (extensively) modified by lukerm (2018). View the original file at:
#     https://github.com/devicehive/devicehive-audio-analysis/blob/master/audio/processor.py


## This script processes incoming sound data and classifies it using our tuned VGGish network ##


## Imports ##

import os
import json
import numpy as np
from keras.models import model_from_json

import vggish_input
import vggish_params as params

from definitions import DATA_DIR


## Constants ##

# Probability of classifying as target tune (see ../fine_tune.py)
P_THRESH = 0.9

__all__ = ['WavProcessor', 'format_predictions']


## Functions ##

def format_predictions(predictions):
    return ', '.join([str(p) for p in predictions])


## Classes ##

class WavProcessor(object):

    _tuned_vggish = None

    def __init__(self, data_dir=DATA_DIR):

        with open(os.path.join(data_dir, 'my_vggish_network.json'), 'r') as j:
            model_dict = json.load(j)
        vggish = model_from_json(json.dumps(model_dict))
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


