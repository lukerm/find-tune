# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:20:37 2018

@author: luke
"""

## Imports ##

import os

import numpy as np
np.random.seed(2018)

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing  import StandardScaler

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression


## Constants ##

DATA_DIR = os.path.join(os.path.expanduser('~'), 'find-tune', 'data')


## Main ##

# Load data from file
data = np.load(os.path.join(DATA_DIR, 'embedding_data.npz'))
X, y, c, s, ids = data['X'], data['y'], data['c'], data['s'], data['i']

# Train (80%) / val (20%) split
p = np.mean(y)
N = int(len(y)*0.2)
print('p: %.4f' % p) # ~ 7.5/1000
print('N: %d'   % N) # ~ 700

# Note that whilst the mean of Binomial(N, p) is ~5, there is a very real chance that
# we could have very few positive data points in the validation set if we're not careful
print('P(X < 3) = %.3f (where X ~ Bin(N, p))' % np.mean(np.random.binomial(N, p, size=100000) < 3))

# So, to ensure this doesn't happen, we will explicitly hold out 5 positive examples for testing
i_pos = np.where(y == 1)[0]
i_neg = np.where(y == 0)[0]

i_pos_va = np.random.choice(i_pos, size=5, replace=False) # 5 for validation
i_pos_tr = np.array(list(set(i_pos) - set(i_pos_va)))     # compliment for training

i_neg_tr = np.random.choice(i_neg, size=len(X)-N, replace=False) # Training
i_neg_va = np.array(list(set(i_neg) - set(i_neg_tr)))            # Validation

# Create split datasets
X_tr = X[np.append(i_neg_tr, i_pos_tr), :]
y_tr = y[np.append(i_neg_tr, i_pos_tr)]
c_tr = c[np.append(i_neg_tr, i_pos_tr)]
s_tr = s[np.append(i_neg_tr, i_pos_tr)]
ids_tr = ids[np.append(i_neg_tr, i_pos_tr)]

X_va = X[np.append(i_neg_va, i_pos_va), :]
y_va = y[np.append(i_neg_va, i_pos_va)]
c_va = c[np.append(i_neg_va, i_pos_va)]
s_va = s[np.append(i_neg_va, i_pos_va)]
ids_va = ids[np.append(i_neg_va, i_pos_va)]

print('Positive labels in training set:   %d' % y_tr.sum())
print('Positive labels in validation set: %d' % y_va.sum())

print('Negative labels in training set:   %d' % (y_tr==0).sum())
print('Negative labels in validation set: %d' % (y_va==0).sum())

# As this is quite an imbalanced problem, we'll use SMOTE to over sample the positive class
sm = SMOTE(random_state=2018)
X_tr_bal, y_tr_bal = sm.fit_sample(X_tr, y_tr)

print('Positive labels in balanced training set: %d' % y_tr_bal.sum())
print('Negative labels in balanced training set: %d' % (y_tr_bal==0).sum())

# Centre and scale the features
ss = StandardScaler()
X_tr_bal = ss.fit_transform(X_tr_bal)
X_va = ss.transform(X_va)



## Linear classifier ##

# Fit multiple classifiers with increasing regularization strength
# Print a report card for each model
for alpha in np.logspace(0, -5, 6):
    print('alpha = %.e' % alpha)
    lr = LogisticRegression(C=1/alpha, solver='liblinear')
    lr.fit(X_tr_bal, y_tr_bal)
    y_pred_tr_bal = lr.predict(X_tr_bal)
    y_pred_tr = lr.predict(X_tr)
    y_pred_va = lr.predict(X_va)

    print('TRAIN (BAL.):')
    print('Accuracy: %.3f' % accuracy_score(y_tr_bal, y_pred_tr_bal))
    p, r, f, s = precision_recall_fscore_support(y_tr_bal, y_pred_tr_bal, beta=1.)
    print('precision: [%.3f, %.3f]' % (p[0], p[1]))
    print('recall:    [%.3f, %.3f]' % (r[0], r[1]))
    print('f_beta:    [%.3f, %.3f]' % (f[0], f[1]))
    print('support:   [%.3f, %.3f]' % (s[0], s[1]))
    print()
    print('TRAIN:')
    print('Accuracy: %.3f' % accuracy_score(y_tr, y_pred_tr))
    p, r, f, s = precision_recall_fscore_support(y_tr, y_pred_tr, beta=1.)
    print('precision: [%.3f, %.3f]' % (p[0], p[1]))
    print('recall:    [%.3f, %.3f]' % (r[0], r[1]))
    print('f_beta:    [%.3f, %.3f]' % (f[0], f[1]))
    print('support:   [%.3f, %.3f]' % (s[0], s[1]))
    print()
    print('VALIDATION:')
    print('Accuracy: %.3f' % accuracy_score(y_va, y_pred_va))
    p, r, f, s = precision_recall_fscore_support(y_va, y_pred_va, beta=1.)
    print('precision: [%.3f, %.3f]' % (p[0], p[1]))
    print('recall:    [%.3f, %.3f]' % (r[0], r[1]))
    print('f_beta:    [%.3f, %.3f]' % (f[0], f[1]))
    print('support:   [%.3f, %.3f]' % (s[0], s[1]))
    print()
    print()


# Check category of the missclassified (false positives) - are they musical?
tp = np.where((y_pred_va == 1) & (y_va == 1))[0]
fp = np.where((y_pred_va == 1) & (y_va == 0))[0]
print('Number of true positives (validation):  %d' % len(tp))
print(c_va[tp])
print('Number of false positives (validation): %d' % len(fp))
print(c_va[fp])

tp = np.where((y_pred_tr == 1) & (y_tr == 1))[0]
fp = np.where((y_pred_tr == 1) & (y_tr == 0))[0]
print('Number of true positives (train):  %d' % len(tp))
print(c_tr[tp])
print('Number of false positives (train): %d' % len(fp))
print(c_tr[fp])

for i, s, c in zip(ids_tr[fp], s_tr[fp], c_tr[fp]):
    print('Video ID: %s (%ds due to %s)' % (i, s, c))
