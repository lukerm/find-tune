import os
import json
import numpy as np
from keras.models import model_from_json

DATA_DIR = './data'

with open(os.path.join(DATA_DIR, 'my_vggish_network.json'), 'r') as j:
    model_dict = json.load(j)
vggish = model_from_json(json.dumps(model_dict))
print('architecture done')

# Where each layer's weights are stored on disk
with open(os.path.join(DATA_DIR, 'layers', 'layer_loc.json'), 'r') as j:
    layer_loc = json.load(j)
for lyr, fp in layer_loc.items():
    print('\t doing %s' % lyr)
    wts  = np.load(os.path.join(DATA_DIR, fp))
    w, b = wts['w'], wts['b']
    wts_copy = [np.copy(w), np.copy(b)]
    del w, b, wts
    vggish.get_layer(lyr).set_weights(wts_copy)
    del wts_copy
    print('\t%s done' % lyr)
print('weights done')
