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

## Utility file for assessing model performance ##

## Imports ##

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


## Functions ##

def print_scorecard(y_true, y_pred, title, beta=1.):

    print(title + ':')
    print('Accuracy: %.3f' % accuracy_score(y_true, y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, beta=beta)
    print('precision: [%.3f, %.3f]' % (p[0], p[1]))
    print('recall:    [%.3f, %.3f]' % (r[0], r[1]))
    print('f_beta:    [%.3f, %.3f]' % (f[0], f[1]))
    print('support:   [%.3f, %.3f]' % (s[0], s[1]))
    print()


def print_negatives(y_true, y_pred, categories, ytids=None, num_secs=None):
    if ytids is not None or num_secs is not None:
        assert len(ytids) == len(num_secs)
        print_ids = True
    else:
        print_ids = False

    i_fp = np.where((y_pred == 1) & (y_true == 0))[0]
    i_fn = np.where((y_pred == 0) & (y_true == 1))[0]
    print('Number of false positives: %d' % len(i_fp))
    print(categories[i_fp])
    print('Number of false negatives: %d' % len(i_fn))
    print(categories[i_fn])

    if print_ids:
        print()
        if len(i_fp) > 0:
            print('False positive IDs:')
            for i, s, c in zip(ytids[i_fp], num_secs[i_fp], categories[i_fp]):
                print('Video ID: %s (%ds due to %s)' % (i, s, c))
        if len(i_fn) > 0:
            print('False negative IDs:')
            for i, s, c in zip(ytids[i_fn], num_secs[i_fn], categories[i_fn]):
                print('Video ID: %s (%ds due to %s)' % (i, s, c))
