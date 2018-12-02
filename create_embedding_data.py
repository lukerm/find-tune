## Binary classification: target track (1) or other music / noise (0) ##


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

# Paths to data files
DATA_DIR = os.path.join(user_root, 'find-tune', 'data')
YT8M_DIR = os.path.join(DATA_DIR, 'youtube_clip')


## Main ## 

# Details for all the links
with open(os.path.join(DATA_DIR, 'link_details.json'), 'r') as j:
    link_details = json.load(j)

# Keep a record of the categories per sound bite
ytid_data = {}
cntr   = 0
n_cats = len(link_details) 
for cat, urls in link_details.items():
    cntr += 1
    print('category: %s (%d / %d)' % (cat, cntr, n_cats))
    for _, yt_id, _, _ in urls:
        # Skip this link if not in the YT8M_DIR
        wav_fpath = os.path.join(YT8M_DIR, 'yt8m_sound_%s.wav' % yt_id)
        if not os.path.isfile(wav_fpath):
            continue
        # Record the category and log-mel spectrogram for this file
        ytid_data[yt_id] = {}
        ytid_data[yt_id]['category'] = cat
        ytid_data[yt_id]['log_mel']  = vggish_input.wavfile_to_examples(wav_fpath)

# Make a record for the target track, too
ytid_data['target'] = {}
ytid_data['target']['category'] = 'target'
ytid_data['target']['log_mel']  = vggish_input.wavfile_to_examples(os.path.join(DATA_DIR, 'target_tune.wav'))

# Embedding features
with tf.Graph().as_default(), tf.Session() as sess:
    # Instantiate the model
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
    features_tensor  = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    # Extract 128-D embedding features for the target track
    [embedding_trgt] = sess.run([embedding_tensor], feed_dict={features_tensor: ytid_data['target']['log_mel']})
    ytid_data['target']['embedding'] = embedding_trgt
    ytid_data['target']['cat_list']  = [ytid_data['target']['category']] * len(embedding_trgt)
    embedding_label  = ['target']* len(embedding_trgt)



