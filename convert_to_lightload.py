import os
import json
from keras.models import model_from_json
from prod.initialize import LightLoadInitializer


DATA_DIR = './data'

with open(os.path.join(DATA_DIR, 'my_vggish_network.json'), 'r') as j:
    model_dict = json.load(j)
vggish_lite = model_from_json(json.dumps(model_dict))

for lyr in vggish_lite.layers:
    weight_init = LightLoadInitializer(lyr.name, bias=False, weights_dir=os.path.join(DATA_DIR, 'layers'))
    bias_init   = LightLoadInitializer(lyr.name, bias=True, weights_dir=os.path.join(DATA_DIR, 'layers'))
    lyr.kernel_initializer = weight_init
    lyr.bias_initializer   = bias_init

# Architecture with modified initializers as JSON
model_dict = json.loads(vggish_lite.to_json())
with open(os.path.join(DATA_DIR, 'my_vggish_network_lite.json'), 'w') as j:
    json.dump(model_dict, j)

