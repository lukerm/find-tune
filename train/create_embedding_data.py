#  Copyright (C) 2018 lukerm
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

## Binary classification: target track (1) or other music / noise (0) ##


## Imports ##

import os
import json
import pickle
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_slim

import numpy as np

from definitions import DATA_DIR, VGGISH_DIR


## Constants ##

# Paths to downloaded VGGish files
checkpoint_path = os.path.join(VGGISH_DIR, 'vggish_model.ckpt')
pca_params_path = os.path.join(VGGISH_DIR, 'vggish_pca_params.npz')

# Paths to YouTube sound data files
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
    # Extract 128-D embedding features for each YouTube track (& target)
    cntr = 0
    for yt_id, data in ytid_data.items():
        cntr += 1
        print('track: %s (%d / %d)' % (yt_id, cntr, len(ytid_data)))
        [my_embedding] = sess.run([embedding_tensor], feed_dict={features_tensor: ytid_data[yt_id]['log_mel']})
        ytid_data[yt_id]['embedding'] = my_embedding
        ytid_data[yt_id]['cat_list']  = [ytid_data[yt_id]['category']] * len(my_embedding)
        ytid_data[yt_id]['labels']    = [1 if yt_id == 'target' else 0]* len(my_embedding)
        ytid_data[yt_id]['n_sec']     = list(range(1, len(my_embedding)+1))
        ytid_data[yt_id]['ytid']      = [yt_id] * len(my_embedding)

# Stack embeddings (X), log_mel data (L),labels (y), categories (c), number of seconds (s) & ids (i)
X = np.ndarray((0, ytid_data['target']['embedding'].shape[-1]))
L = np.ndarray((0,) + ytid_data['target']['log_mel'].shape[1:])
y = []
c = []
s = []
i = []

# We can use this method because the data is small and runs quickly
# Otherwise, you should pre-assign a matrix of correct size and use pointers
for k, v in ytid_data.items():
    X  = np.concatenate([X, v['embedding']], axis=0)
    L  = np.concatenate([L, v['log_mel']], axis=0)
    y += v['labels']
    c += v['cat_list']
    s += v['n_sec']
    i += v['ytid']

assert len(X) == len(L)
assert len(X) == len(y)
assert len(X) == len(c)
assert len(X) == len(s)
assert len(X) == len(i)

# Convert to arrays
y = np.array(y)
c = np.array(c)
s = np.array(s)
i = np.array(i)

# Save to data file
np.savez(os.path.join(DATA_DIR, 'embedding_data.npz'), X=X, y=y, c=c, s=s, i=i, L=L)
with open(os.path.join(DATA_DIR, 'yt_clips_dict.pkl'), 'wb') as f:
    pickle.dump(ytid_data, f)
