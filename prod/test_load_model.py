import os
import json
from keras.models import load_model, model_from_json
from definitions import DATA_DIR

with open(os.path.join(DATA_DIR, 'my_vggish_network.json'), 'r') as j:
    model_dict = json.load(j)
vggish = model_from_json(json.dumps(model_dict))
vggish.load_weights(os.path.join(DATA_DIR, 'my_vggish_network.h5'))

print('loaded')
