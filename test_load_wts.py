import os
import json
import numpy as np
from keras.models import model_from_json

DATA_DIR = './data'

with open(os.path.join(DATA_DIR, 'my_vggish_network.json'), 'r') as j:
    model_dict = json.load(j)
vggish = model_from_json(json.dumps(model_dict))
print('architecture done', flush=True)

# Where each layer's weights are stored on disk
with open(os.path.join(DATA_DIR, 'layers', 'layer_loc.json'), 'r') as j:
    layer_loc = json.load(j)
#for lyr, fp in layer_loc.items():
for lyr in ['fc1_1']:
    print('\t doing %s' % lyr, flush=True)
    fp = 'layers/%s.npz' % lyr
    wts  = np.load(os.path.join(DATA_DIR, fp))
    print('load done', flush=True)
    w, b = wts['w'], wts['b']
    print('now setting weights', flush=True)
    vggish.get_layer(lyr).set_weights([w,b])
    del w, b, wts
    print('\t%s done' % lyr, flush=True)
print('weights done', flush=True)
