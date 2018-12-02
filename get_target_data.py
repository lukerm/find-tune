
## Imports ##

import os

import soundfile as sf
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_slim


## Constants ## 

user_root = '/'.join(os.getcwd().split('/')[:3])
# Paths to downloaded VGGish files
# TODO: fix path
VGGISH_DIR= os.path.join(user_root, 'tf-models','research','audioset')
checkpoint_path = os.path.join(VGGISH_DIR, 'vggish_model.ckpt')
pca_params_path = os.path.join(VGGISH_DIR, 'vggish_pca_params.npz')

DATA_DIR = os.path.join(user_root, 'find-tune', 'data')


## Main ## 


# Get embedding features for the target file
tgt_data_lmel = vggish_input.wavfile_to_examples(os.path.join(DATA_DIR, 'target_tune.wav'))


with tf.Graph().as_default(), tf.Session() as sess:
  vggish_slim.define_vggish_slim()
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor   = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor  = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
  [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: tgt_data_lmel})


print(embedding_batch)
print(embedding_batch.shape)
