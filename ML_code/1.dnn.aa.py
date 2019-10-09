#!/usr/bin/env python3
import numpy as np
np.set_printoptions(threshold=np.inf) # setting the print threshold to infinity
import pandas as pd
from sklearn.pipeline import Pipeline
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold


# read in data, x has header but y doesn't have header
aa_train_x=pd.read_csv("/all_drug/aa_train_x.csv")
aa_val_x=pd.read_csv("/all_drug/aa_val_x.csv")
aa_test_x=pd.read_csv("/all_drug/aa_test_x.csv")

aa_train_y=pd.read_csv("/all_drug/aa_train_y.csv", header=None)
aa_val_y=pd.read_csv("/all_drug/aa_val_y.csv", header=None)
aa_test_y=pd.read_csv("/all_drug/aa_test_y.csv", header=None)

# make the target lable a (n,) demension
aa_train_y=aa_train_y.squeeze()
aa_val_y=aa_val_y.squeeze()
aa_test_y=aa_test_y.squeeze()


print('---------------------finish loading data -------------------------')
#-----------------set up parameters--------------
# use Theano as backend
from time import time
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

#set_keras_backend("theano")
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import auc, f1_score
from sklearn.utils import class_weight
from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping, TensorBoard
from sklearn.metrics import roc_auc_score
import keras
from keras import regularizers
from dnn_accuracy_metrics import stratified_split_n_fold, auc, check_units, f1, perf_measure 


# get learning rate as a metric so it can be printed out in each epoch
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

#use optimizer
#define learning rate 10^-4 to 10^-1
#lr=10**np.random.uniform(-4,-3)
opt_adam = keras.optimizers.Adam(lr=0.001105, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
lr_metric = get_lr_metric(opt_adam)

# define model
def get_model(X_train):
       model = Sequential()
       model.add(Dense(hidden_layer1, input_dim=X_train.shape[1], activation='relu',kernel_regularizer=regularizers.l2(0.0018)))
       #maybe use BatchNormalization(axis = 3)(x)
       model.add(Dense(hidden_layer2, activation='relu',kernel_regularizer=regularizers.l2(0.0018)))
       model.add(Dense(1,activation="sigmoid",kernel_regularizer=regularizers.l2(0.0018)))
              #compile model
       model.compile(loss='binary_crossentropy', optimizer=opt_adam, weighted_metrics=[auc,f1, lr_metric])
       return model

# parameters that need to change input if run other dataset
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(aa_train_y),
                                                 aa_train_y)
class_weights=dict(enumerate(class_weights))

# define # of node in each layer, from the previous 
hidden_layer1=int(np.random.randint(200,350,1))
hidden_layer2=int(np.random.randint(100,hidden_layer1,1))
#define learning rate 10^-4 to 10^-1
#learning_rate=10**np.random.uniform(-4,-3)

# use this tuned settings
hidden_layer1=180
hidden_layer2=115

#run model use CV
model = get_model(aa_train_x)
model.summary()

# 6 fold CV
folds =stratified_split_n_fold(aa_train_x,aa_train_y,6)
for j, (train_idx, val_idx) in enumerate(folds):
       print('\nFold ',j)
       X_train_cv = aa_train_x.loc[train_idx,].values
       y_train_cv = aa_train_y.loc[train_idx,].values
       X_valid_cv = aa_train_x.loc[val_idx,].values
       y_valid_cv= aa_train_y.loc[val_idx,].values
        # list how many people are cases and controls in each fold
       unique, counts = np.unique(y_train_cv, return_counts=True)
       dict(zip(unique, counts))
       unique, counts = np.unique(y_valid_cv, return_counts=True)
       dict(zip(unique, counts))
       #define file path
       learning_rate=opt_adam.lr
       b=K.eval(learning_rate)
       #filepath="AA_nodrugs"+str(b)+"l1as"+str(hidden_layer1)+"l2as"+str(hidden_layer2)+'epoch{epoch:02d}-{val_auc:.2f}-{val_f1:.2f}'+'.hdf5'
       #print(filepath)
       callbacks = [keras.callbacks.EarlyStopping(monitor='val_weighted_f1', min_delta=0.0005, patience=10, verbose=0, mode='max', baseline=0.9), keras.callbacks.TensorBoard(log_dir ="logs/{}".format(time()))]
#,keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, monitor='f1', mode='max'),  keras.callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=8, verbose=0, min_lr=1e-10, mode='max')]
       final_model=model.fit(X_train_cv,y_train_cv,
             batch_size=len(aa_train_y.values),
             #epochs=15 and cv=10 result in perfect auc=1
             #epchos=10 and cv=3 result in perfect auc=1
             epochs=8,
             shuffle=False,
             verbose=2,
             class_weight=class_weights, # sample_weight is not used becuase all the sames are treated equal
             validation_data = (X_valid_cv, y_valid_cv),
             callbacks = callbacks)

#----------print all the results
# see if learning rate got updated
print(model.summary())
print('learning rate')
print(K.eval(learning_rate))

# evaluate on train set
print(model.evaluate(aa_train_x.values, aa_train_y.values,batch_size=len(aa_train_y.values)))
print(model.metrics_names)
#['loss', 'weighted_auc', 'weighted_f1','learning_rate']
y_true=aa_train_y.values
y_pred=model.predict_classes(aa_train_x.values, batch_size=len(aa_train_y.values), verbose=0).flatten()
print(perf_measure(y_true, y_pred))

# evaluate on validation set
result_on_val=model.evaluate(aa_val_x.values, aa_val_y.values,batch_size=len(aa_val_y.values))
print(result_on_val)
y_true=aa_val_y.values
y_pred=model.predict_classes(aa_val_x.values, batch_size=len(aa_val_y.values), verbose=0).flatten()
print(perf_measure(y_true, y_pred))

# get the accuracy parameters
val_auc=np.round(result_on_val[1],2)
val_f1=np.round(result_on_val[2],2)

#save model to file
filepath="AA_nodrugs"+str(K.eval(learning_rate))+"layer1as"+str(hidden_layer1)+"layer2as"+str(hidden_layer2)+'val_auc'+str(val_auc)+'val_f1'+str(val_f1)+'.hdf5'
model.save(filepath)  # the load_model doesn't work so will not use model.save
print("Saved weigthts to disk")



#----------------- test model and weight on test set
print(model.evaluate(aa_test_x.values, aa_test_y.values,batch_size=len(aa_test_y.values)))
model.metrics_names
#['loss', 'auc', 'f1']


y_true=aa_test_y.values
y_pred=model.predict_classes(aa_test_x.values, batch_size=64, verbose=0).flatten()

print(perf_measure(y_true, y_pred))
