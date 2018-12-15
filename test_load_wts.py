import os
import json
import numpy as np
from keras.models import model_from_json
from prod.initialize import LightLoadInitializer

# TODO: fix path!
DATA_DIR = './data'

with open(os.path.join(DATA_DIR, 'my_vggish_network_lite.json'), 'r') as j:
    model_dict = json.load(j)

vggish = model_from_json(
            json_string = json.dumps(model_dict),
            custom_objects = {'LightLoadInitializer': LightLoadInitializer}
         )
print('done', flush=True)
