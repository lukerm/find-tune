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

## Imports ##

import os
import pickle
import numpy as np
np.random.seed(2018)

import sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'find-tune', 'train'))
import perf_utils as pu

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from definitions import DATA_DIR


## Main ##
print('=== Preparing data ===')

# Load data from file
# X = embedding features (output from VGGish model)
# y = binary target variable (1 = target track, 0 = other sound)
# c = sound categories (e.g. 'Jingle Bells', 'target')
# s = seconds in the original YouTube track where this clip begins
# ids = YouTube unique identifiers for the track
# L = log-mel attributes (input to VGGish model)
data = np.load(os.path.join(DATA_DIR, 'embedding_data.npz'))
X, y, c, s, ids, L = data['X'], data['y'], data['c'], data['s'], data['i'], data['L']

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
L_tr = L[np.append(i_neg_tr, i_pos_tr), :, :]

X_va = X[np.append(i_neg_va, i_pos_va), :]
y_va = y[np.append(i_neg_va, i_pos_va)]
c_va = c[np.append(i_neg_va, i_pos_va)]
s_va = s[np.append(i_neg_va, i_pos_va)]
ids_va = ids[np.append(i_neg_va, i_pos_va)]
L_va = L[np.append(i_neg_va, i_pos_va), :, :]

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
X_tr = ss.transform(X_tr)
X_va = ss.transform(X_va)



## Linear classifier ##
print('\n')
print('=== Linear classifier ===')
print()
# Yields accuracy and perfect recall on positive class
# Suffers from imperfect precision on positive class (gives false positives)

# Fit multiple classifiers with increasing regularization strength
# Print a report card for each model
for alpha in np.logspace(-4, -4, 1):
    print('alpha = %.e' % alpha)
    lr = LogisticRegression(C=1/alpha, solver='liblinear')
    lr.fit(X_tr_bal, y_tr_bal)
    y_pred_tr_bal = lr.predict(X_tr_bal)
    y_pred_tr = lr.predict(X_tr)
    y_pred_va = lr.predict(X_va)

    pu.print_scorecard(y_tr_bal, y_pred_tr_bal, title='TRAIN (BAL.)')
    pu.print_scorecard(y_tr, y_pred_tr, title='TRAIN')
    pu.print_scorecard(y_va, y_pred_va, title='VALIDATION')
    print()

# An analysis of incorrect predictions
pu.print_negatives(y_va, y_pred_va, c_va, ytids=ids_va, num_secs=s_va)


## Random Forest ##
print('\n')
print('=== Random forest classifier ===')
print()
# Performs quite well, good accuracy
# Perfect precision on positive class (desirable: don't want false positives)
# Note: overfits without cap on depth
rf = RandomForestClassifier(n_estimators = 50, max_depth = 5)
#rf.fit(X_tr_bal, y_tr_bal)
rf.fit(X_tr, y_tr)
y_pred_tr_bal = rf.predict(X_tr_bal)
y_pred_tr = rf.predict(X_tr)
y_pred_va = rf.predict(X_va)

pu.print_scorecard(y_tr_bal, y_pred_tr_bal, title='TRAIN (BAL.)')
pu.print_scorecard(y_tr, y_pred_tr, title='TRAIN')
pu.print_scorecard(y_va, y_pred_va, title='VALIDATION')
print()

pu.print_negatives(y_va, y_pred_va, c_va, ytids=ids_va, num_secs=s_va)


## Dense Neural Network ##
print('\n')
print('=== Neural network classifier ===')
print()
# Without any hyperparameter tuning, the network gets a perfect score on all metrics!

def fit_nn_model(lr0, h1, bsz, verbose=0, cp_path=None):
    """
    Generic model for fitting a 1-layer neural network
    """

    # Callbacks
    lr_red = ReduceLROnPlateau(monitor='val_loss', min_delta=0, factor=0.75, patience=2)
    e_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15)
    cbacks = [lr_red, e_stop]
    if cp_path is not None:
        cbacks.append(ModelCheckpoint(cp_path, save_best_only=True))


    # Model architecture
    ix = Input((X_tr_bal.shape[1],), name='vggish_feat_input')
    x  = Dense(h1, activation='relu', name='fc_last')(ix)
    x  = Dense(1, activation='sigmoid', name='classify')(x)

    # Compile the Model
    model = Model(ix, x)
    optzr = Adam(lr=lr0)
    model.compile(optzr, loss='binary_crossentropy', metrics=['accuracy'])
    # Pre-training metrics
    ep0_tr = model.evaluate(X_tr_bal, y_tr_bal, verbose=verbose)
    ep0_va = model.evaluate(X_va, y_va, verbose=verbose)
    # Fit the model
    history = model.fit(X_tr_bal, y_tr_bal,
                        epochs=200, batch_size=bsz,
                        callbacks=cbacks,
                        validation_data=(X_va, y_va),
                        verbose=verbose,
                        )

    return (model, history, (ep0_tr, ep0_va))

# Hyperparameters
lr0 = 0.01
h1  = 128
bsz = 64

model, history, _ = fit_nn_model(lr0, h1, bsz, verbose=0)

# Make (probability) predictions
y_pred_tr_bal = model.predict(X_tr_bal)[:, 0]
y_pred_tr = model.predict(X_tr)[:, 0]
y_pred_va = model.predict(X_va)[:, 0]

p_thresh = 0.5
pu.print_scorecard(y_tr_bal, y_pred_tr_bal > p_thresh, title='TRAIN (BAL.)')
pu.print_scorecard(y_tr, y_pred_tr > p_thresh, title='TRAIN')
pu.print_scorecard(y_va, y_pred_va > p_thresh, title='VALIDATION')


# More rigorous test: use cross-validation, 5-folds
# Again, perfect scorecards for each fold - shuffling is important for diversity of training data!
print('\n')
print('=== 5-fold CV ===')
print()
histories = []
fold_cntr = 0
skf = StratifiedKFold(5, shuffle=True, random_state=2018)
for i_tr, i_va in skf.split(X, y):

    # Storage for data / models relating to this fold
    fold_dir = os.path.join(DATA_DIR, 'fold%d' % fold_cntr)
    os.makedirs(fold_dir, exist_ok=True)

    # Create split datasets
    X_tr = X[i_tr, :]
    y_tr = y[i_tr]
    c_tr = c[i_tr]
    s_tr = s[i_tr]
    ids_tr = ids[i_tr]
    L_tr = L[i_tr, :, :]

    X_va = X[i_va, :]
    y_va = y[i_va]
    c_va = c[i_va]
    s_va = s[i_va]
    ids_va = ids[i_va]
    L_va = L[i_va, :, :]

    print('FOLD %d' % fold_cntr)
    print('======')
    print()
    print('Positive labels in training set:   %d' % y_tr.sum())
    print('Positive labels in validation set: %d' % y_va.sum())

    # over sample with SMOTE
    sm = SMOTE(random_state=2018)
    X_tr_bal, y_tr_bal = sm.fit_sample(X_tr, y_tr)
    print('Positive labels in balanced set:   %d' % y_tr_bal.sum())

    print()
    print('=== Linear classifier ===')
    alpha = 1e-4
    lr = LogisticRegression(C=1/alpha, solver='liblinear')
    lr.fit(X_tr_bal, y_tr_bal)
    y_pred_va = lr.predict(X_va)
    pu.print_scorecard(y_va, y_pred_va, title='VALIDATION')
    pu.print_negatives(y_va, y_pred_va > p_thresh, c_va, ytids=ids_va, num_secs=s_va)

    # Centre and scale the features
    ss = StandardScaler()
    X_tr_bal = ss.fit_transform(X_tr_bal)
    X_tr = ss.transform(X_tr)
    X_va = ss.transform(X_va)
    with open(os.path.join(fold_dir, 'sc_fold%d.pkl' % fold_cntr), 'wb') as f:
        pickle.dump(ss, f)

    # Save data (with input scaled features) for later reference
    np.savez(os.path.join(fold_dir, 'foldwise_data_tr.npz'), X=X_tr, y=y_tr, c=c_tr, s=s_tr, i=ids_tr, L=L_tr)
    np.savez(os.path.join(fold_dir, 'foldwise_data_va.npz'), X=X_va, y=y_va, c=c_va, s=s_va, i=ids_va, L=L_va)
    np.savez(os.path.join(fold_dir, 'foldwise_data_bal.npz'), X=X_tr_bal, y=y_tr_bal)

    # Fit the model
    save_path = os.path.join(fold_dir, 'nn_fold%d.model' % fold_cntr)
    _, history, _ = fit_nn_model(lr0, h1, bsz, verbose=0, cp_path=save_path)
    histories.append(history)

    # Load the best model and predict
    print()
    print('=== Dense neural network ===')
    model = load_model(save_path)
    y_pred_va = model.predict(X_va)[:, 0]
    pu.print_scorecard(y_va, y_pred_va > p_thresh, title='VALIDATION')
    pu.print_negatives(y_va, y_pred_va > p_thresh, c_va, ytids=ids_va, num_secs=s_va)

    print()
    fold_cntr += 1

