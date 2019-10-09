#!/usr/bin/env python3
import numpy as np
np.set_printoptions(threshold=np.inf) # setting the print threshold to infinity
import pandas as pd
from sklearn.pipeline import Pipeline
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold

# read in data
aa_train_x=pd.read_csv("/all_drug/aa_train_x.csv")
aa_val_x=pd.read_csv("/all_drug/aa_val_x.csv")
aa_test_x=pd.read_csv("/all_drug/aa_test_x.csv")

aa_train_y=pd.read_csv("/all_drug/aa_train_y.csv", header=None)
aa_val_y=pd.read_csv("/all_drug/aa_val_y.csv", header=None)
aa_test_y=pd.read_csv("/all_drug/aa_test_y.csv", header=None)


print('---------------------finish loading data -------------------------')
#-----------------set up parameters--------------
# use Theano as backend
from keras.models import model_from_json  #model save package
from keras.models import load_model    # model save package
import tensorflow as tf
import keras.backend as K
from importlib import reload
import os
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend   # this will throw a warning tensorflow:From /share/pkg/python/3.6.2/install/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:1290: calling reduce_mean (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
#Instructions for updating:
#keep_dims is deprecated, use keepdims instead

##set_keras_backend("theano") # tried other backend to prevent the above error
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import auc, f1_score
from sklearn.utils import class_weight
from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import roc_auc_score
import keras
from keras import regularizers
from dnn_accuracy_metrics import stratified_split_n_fold, auc, check_units, f1, perf_measure 

# make learning rate as a metric so it can be printed out in each epoch
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# same optimizer
opt_adam = keras.optimizers.Adam(lr=0.001105, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
lr_metric = get_lr_metric(opt_adam)



#load entire model, doesn't work, custom auc is not in the /usr3/graduate/jiayiwu/.local/lib/python3.6/site-packages/keras/utils/generic_utils.py
hdf5_file=sys.argv[1]
new_model=load_model(hdf5_file, custom_objects={'auc':auc, 'f1':f1, 'lr': lr_metric})


# evaluate loaded model on train data
new_model.compile(loss='binary_crossentropy', optimizer=opt_adam, weighted_metrics=[auc,f1, lr_metric])

#-----------------varify in train model
print('-----------------in train set--------------------')
print(new_model.evaluate(aa_train_x.values, aa_train_y.values,batch_size=len(aa_train_y.values)))
y_true=aa_train_y.values
y_pred=new_model.predict_classes(aa_train_x.values, batch_size=len(aa_train_y.values), verbose=0).flatten()
print(perf_measure(y_true, y_pred))#----------------- test model and weight on test set

# varify in validation set
print ('------------------in validation set -------------')
print(new_model.evaluate(aa_val_x.values, aa_val_y.values,batch_size=len(aa_val_y.values)))
y_true=aa_val_y.values
y_pred=new_model.predict_classes(aa_val_x.values, batch_size=len(aa_val_y.values), verbose=0).flatten()
print(perf_measure(y_true, y_pred))

# test on test set
print('------------------in test set --------------------')
print(new_model.evaluate(aa_test_x.values, aa_test_y.values,batch_size=len(aa_test_y.values)))
y_true=aa_test_y.values
y_pred=new_model.predict_classes(aa_test_x.values, batch_size=len(aa_test_x.values), verbose=0).flatten()
print(perf_measure(y_true, y_pred))
