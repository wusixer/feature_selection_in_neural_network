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
from pprint import pprint
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
#hdf5_file='AA_nodrugs0.001105layer1as180layer2as115val_auc0.92val_f10.69.hdf5'
new_model=load_model(hdf5_file, custom_objects={'auc':auc, 'f1':f1, 'lr': lr_metric})

#print model details by layers
for layer in range(len(new_model.layers)):
    print(layer)
    pprint(new_model.layers[layer].get_config())

# evaluate loaded model on train data
new_model.compile(loss='binary_crossentropy', optimizer=opt_adam, weighted_metrics=[auc,f1, lr_metric])
#-------------------------
#  0
#  {'activation': 'relu',
  #  'activity_regularizer': None,
  #  'batch_input_shape': (None, 3092),
  #  'bias_constraint': None,
  #  'bias_initializer': {'class_name': 'Zeros', 'config': {}},
  #  'bias_regularizer': None,
  #  'dtype': 'float32',
  #  'kernel_constraint': None,
  #  'kernel_initializer': {'class_name': 'VarianceScaling',
						 #  'config': {'distribution': 'uniform',
									  #  'mode': 'fan_avg',
									  #  'scale': 1.0,
									  #  'seed': None}},
  #  'kernel_regularizer': {'class_name': 'L1L2',
						 #  'config': {'l1': 0.0, 'l2': 0.0017999999690800905}},
  #  'name': 'dense_1',
  #  'trainable': True,
  #  'units': 180,
  #  'use_bias': True}
#  1
#  {'activation': 'relu',
  #  'activity_regularizer': None,
  #  'bias_constraint': None,
  #  'bias_initializer': {'class_name': 'Zeros', 'config': {}},
  #  'bias_regularizer': None,
  #  'kernel_constraint': None,
  #  'kernel_initializer': {'class_name': 'VarianceScaling',
						 #  'config': {'distribution': 'uniform',
									  #  'mode': 'fan_avg',
									  #  'scale': 1.0,
									  #  'seed': None}},
  #  'kernel_regularizer': {'class_name': 'L1L2',
						 #  'config': {'l1': 0.0, 'l2': 0.0017999999690800905}},
  #  'name': 'dense_2',
  #  'trainable': True,
  #  'units': 115,
  #  'use_bias': True}
#  2
#  {'activation': 'sigmoid',
  #  'activity_regularizer': None,
  #  'bias_constraint': None,
  #  'bias_initializer': {'class_name': 'Zeros', 'config': {}},
  #  'bias_regularizer': None,
  #  'kernel_constraint': None,
  #  'kernel_initializer': {'class_name': 'VarianceScaling',
						 #  'config': {'distribution': 'uniform',
									  #  'mode': 'fan_avg',
									  #  'scale': 1.0,
									  #  'seed': None}},
  #  'kernel_regularizer': {'class_name': 'L1L2',
						 #  'config': {'l1': 0.0, 'l2': 0.0017999999690800905}},
  #  'name': 'dense_3',
  #  'trainable': True,
  #  'units': 1,
  #  'use_bias': True}


#-------------------------
# get action potential matrix of layer 0, output = activation(dot(input, kernel) + bias), https://keras.io/layers/core/#dense
get_0th_layer_output=K.function([new_model.layers[0].input],[new_model.layers[0].output])
zerolayer_output=get_0th_layer_output([aa_train_x])[0]  # this is a list with len of 1
zerolayer_output.shape
#(903, 180)

# get weights matrix  from layer 0 
len(new_model.layers[0].get_weights()[0])   # number of rows in weight matrix 
zerolayer_weights=np.zeros((3092, 180), dtype=float)
for i in range(len(new_model.layers[0].get_weights()[0])):
    zerolayer_weights[i,:]=new_model.layers[0].get_weights()[0][i]

# the input training matrix
aa_train_x.shape   #(903, 3092)


# get pij = 1/m * sum over m (abs(wji*xi +bj)) from the paper https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7280626
# the weight matrix from the input layer to first layer has the dimension of (208, 1820) ---[n1, n0] weight matrix https://www.youtube.com/watch?v=vuYcFz86ryo
# create pij matrix filled with zeros, i is the # of variables -3092, j is the number of node in 1st layer - 180
pij=np.zeros((3092,180), dtype=float)
# calculate values in pij
for i in range(len(pij)):
    print(i)
    for j in range(len(pij[i])):
        # the operation below is to take the mutiplication of a given variable on a given node in all samples, and then take the absolute and then sum over
        pij[i][j] = np.sum(np.absolute(zerolayer_weights[i][j]*aa_train_x.iloc[:,j:j+1] + new_model.layers[0].get_weights()[1][j])) # wij*x +bias

# divide by 903, the number of people

pij=np.round(np.divide(pij, 903), decimals=4)
pij_sum_all_var = np.mean(pij, axis=0)

# get cij - the relative contribution of ith input towards the activation potential of jth hidden neuron
# cij = aij / pij_sum_all_var
cij=np.zeros((3092,180), dtype=float)
for i in range(len(cij)):
    print(i)
    for j in range(len(cij[i])):
        cij[i][j] = np.divide(np.sum(zerolayer_weights[i][j]*aa_train_x.iloc[:,j:j+1] + new_model.layers[0].get_weights()[1][j]), pij_sum_all_var[j] ) # wij*x +bias


# get ci+ the net positive contribution of input dimension i over all hidden neurons
ci_plus=np.zeros((3092,180), dtype=float)
for i in range(len(ci_plus)):
    print(i)
    for j in range(len(ci_plus[i])):
        ci_plus[i,j]=(cij[i][j])*(cij[i][j] > 0)

# get the ci_plus by summing up the number of neurons in the first hidden layer, summing across columns
ci_plus_average = np.mean(ci_plus, axis=1)

#get feature importance ranking
rank=np.column_stack((aa_train_x.columns.values, ci_plus_average))


# save pij to file
#np.save(pij, "all_drug.aa.pij_array",allow_pickle=False )

#save rank to file
np.savetxt('aa.ci_plus_average.txt', ci_plus_average)


#-------------------------------------------------
