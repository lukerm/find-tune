# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:20:37 2018

@author: luke
"""

## Imports ##

import os

import numpy as np
np.random.seed(2018)


## Constants ##

DATA_DIR = os.path.join(os.path.expanduser('~'), 'find-tune', 'data')


## Main ##

# Load data from file
data = np.load(os.path.join(DATA_DIR, 'embedding_data.npz'))
X, y, c = data['X'], data['y'], data['c']

# Train (80%) / val (20%) split
p = np.mean(y)
N = int(len(y)*0.2)
print('p: %.4f' % p) # ~ 1/1000
print('N: %d'   % N) # ~ 5000

# Note that whilst the mean of Binomial(5000, 0.001) is 5, there is a very real chance that
# we could have very few positive data points in the validation set if we're not careful
print('P(X < 3) = %.3f (where X ~ Bin(N, p))' % np.mean(np.random.binomial(N, p, size=100000) < 3))

# So, to ensure this doesn't happen, we will explicitly hold out 5 positive examples for testing
i_pos = np.where(y == 1)[0]
i_neg = np.where(y == 0)[0]

i_pos_tr = np.random.choice(i_pos, size=5, replace=False) # Training
i_pos_va = np.array(list(set(i_pos) - set(i_pos_tr)))     # compliment for validation

i_neg_tr = np.random.choice(i_neg, size=len(X)-N, replace=False) # Training
i_neg_va = np.array(list(set(i_neg) - set(i_neg_tr)))            # Validation

# Create split datasets
X_tr = X[np.append(i_neg_tr, i_pos_tr), :]
y_tr = y[np.append(i_neg_tr, i_pos_tr)]

X_va = X[np.append(i_neg_va, i_pos_va), :]
y_va = y[np.append(i_neg_va, i_pos_va)]

print('Positive labels in training set:   %d' % y_tr.sum())
print('Positive labels in validation set: %d' % y_va.sum())

print('Negative labels in training set:   %d' % (~y_tr.astype(bool)).sum())
print('Negative labels in validation set: %d' % (~y_tr.astype(bool)).sum())