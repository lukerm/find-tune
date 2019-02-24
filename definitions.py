import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# This path must match where you cloned tensorflow's models repository to - please adjust as required
VGGISH_DIR = os.path.join(os.path.expanduser('~'), 'tf-models', 'research', 'audioset')
#VGGISH_DIR = os.path.join(ROOT_DIR, '..', 'tf-models', 'research', 'audioset')
