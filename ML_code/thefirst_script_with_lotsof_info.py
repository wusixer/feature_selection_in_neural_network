#!/usr/bin/env python3
import numpy as np
np.set_printoptions(threshold=np.inf) # setting the print threshold to infinity
import pandas as pd
from sklearn.pipeline import Pipeline
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold

# read in data
aa_train=pd.read_csv("/aa_dev.txt", delimiter="\t")
aa_val=pd.read_csv("/aa_val.txt", delimiter="\t")
aa_test=pd.read_csv("/phenotype/aa.ml.test.z.txt", delimiter="\t")
# check demension
aa_train.shape; aa_val.shape; aa_test.shape
#(918, 3666),(154,3666),(120, 3666)

## select columns needed
# var for input
A = aa_train.columns.str.startswith('A')
B = aa_train.columns.str.startswith('B')
B_drop = aa_train.columns.str.startswith('B7')

ssadda_related = [i for i in aa_train.filter(regex=('^[A-Z][0-9]')).columns] + [i for i in aa_train.filter(regex=('^[a-z][0-9]')).columns]
disorder_related = [i for i in aa_train.columns if i not in ssadda_related ]
mental_trait = [i for i in aa_train.filter(regex=('^[I-Z][0-9]')).columns]+ [i for i in aa_train.filter(regex=('^[i-z][0-9]')).columns]
mental_trait = [i for i in mental_trait if i not in  aa_train.filter(regex=('^[V-W][0-9]')).columns]
mental_trait = [i for i in mental_trait if i not in aa_train.filter(regex=('^[v-w][0-9]')).columns]
basic = [i for i in aa_train.columns[A]]+[i for i in aa_train.columns[B] if i not in aa_train.columns[B_drop]]+disorder_related +mental_trait

# vars for drop
c = [i for i in disorder_related if 'Score' in i]
d = [i for i in disorder_related if 'Box' in i]
e = [i for i in disorder_related if 'Total' in i]
f = [i for i in disorder_related if 'ITEMS' in i]
g = [i for i in disorder_related if 'Clustering' in i]

drop = [i for i in aa_train.columns[B_drop]] + c+d+e+f+g+["opices","index","train","SSADDA_ID", "Meth_Treated"]

final = list(set([i for i in basic if i not in drop]))

# get the dataset for analyisis
aa_train_y = aa_train['opices']
aa_train_x = aa_train[final]
print("aa_train_x")
print(aa_train_x.shape)
aa_train_x.shape
# (918, 1802)
aa_val_y = aa_val['opices']
aa_val_x = aa_val[final]
print("aa_val_x")
print(aa_val_x.shape)
aa_val_x.shape
# (154,1802)
aa_test_y = aa_test['opices']
aa_test_x = aa_test[final]
print('aa_test_x')
print(aa_test_x.shape)
#(120, 1802)
aa_train = None #; aa_test = None    #remove aa_train to save space
#a=["A" in x for x in list(aa_train)]
#b=aa_train.iloc[:, np.where(a)[0]]
#a_range=([range(1,13)] +[446]+[range(461,463)]+[range(499,504)]+[range(1689,1744)])
#list(map((lambda i: b.iloc[:,i]), a_range))

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


#set_keras_backend("theano")
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import auc, f1_score
from sklearn.utils import class_weight
from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import roc_auc_score


#-----------------write basic functions
# split train,validation set into 10 folds
def stratifed_split_n_fold(X_train,Y_train, n):
	folds = list(StratifiedKFold(n_splits=n, shuffle=False,random_state=1).split(X_train,Y_train))
	return folds

# save checkpoint
def get_callbacks(filepath, patience_lr_epoch):
	   mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor=f1, mode='max')
	   #implement learning rate decay,set min learning rate as 1e-5
	   reduce_lr_loss = ReduceLROnPlateau(monitor=f1, factor=0.1, patience=patience_lr_epoch, verbose=0, min_lr=0.000001, mode='max')
	   #early stopping if f1 doesn't increase for 0.001 over 3 epoches after reaching 0.95
	   earlystopping = EarlyStopping(monitor=f1, min_delta=0.001, patience=patience_lr_epoch, verbose=0, mode='max', baseline=0.95)
        return [mcp_save, reduce_lr_loss, earlystopping]


#auc from tf, doesn't throw error like sklearn
def auc(y_true, y_pred):
	# didn't do weights=class_weights, becuase class_weights is used in model fit alraedy
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


#f1 from keras by batch, https://github.com/keras-team/keras/issues/5400
def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#----------get the correctness status
#-----------taken from https://datascience.stackexchange.com/questions/28493/confusion-matrix-get-items-fp-fn-tp-tn-python
def perf_measure(y_actual, y_pred):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for i in range(len(y_pred)): 
		if y_actual[i]==y_pred[i]==1:
			TP += 1
		if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
			FP += 1
		if y_actual[i]==y_pred[i]==0:
			TN += 1
		if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
			FN += 1
	return(TP, FP, TN, FN)



# define model
def get_model(X_train):
	   model = Sequential()
	   model.add(Dense(hidden_layer1, input_dim=X_train.shape[1], activation='relu'))
	   #maybe use BatchNormalization(axis = 3)(x)
	   model.add(Dense(hidden_layer2, activation='relu'))
	   model.add(Dense(1,activation="sigmoid"))
	   #use optimizer
	  # opt_adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	   #compile model
	   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc,f1])
	   return model


# parameters that need to change input if run other dataset
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(aa_train_y),
                                                 aa_train_y)
class_weights=dict(enumerate(class_weights))

# define # of node in each layer, from the previous 
hidden_layer1=int(np.random.randint(350,450,1))
hidden_layer2=int(np.random.randint(100,hidden_layer1,1))
#define learning rate 10^-4 to 10^-1
learning_rate=10**np.random.uniform(-4,-3)

# use this tuned settings
#hidden_layer1=350
#hidden_layer2=274
#learning_rate=0.0005

#run model use CV
model = get_model(aa_train_x)
model.summary()
# print the set parameters
print("the learning rate is",learning_rate)

# 6 fold CV
folds =stratifed_split_n_fold(aa_train_x,aa_train_y,6)
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
	   filepath="AA_nodrugs"+str(np.round(learning_rate,5))+"l1as"+str(hidden_layer1)+"l2as"+str(hidden_layer2)+".hdf5"
	   callbacks = get_callbacks(filepath = filepath, patience_lr_epoch=3)
	   final_model=model.fit(X_train_cv,y_train_cv,
			 #batch_size=64,
			 #epochs=15 and cv=10 result in perfect auc=1
			 #epchos=10 and cv=3 result in perfect auc=1
			 epochs=10,
			 shuffle=False,
			 verbose=2,
			 class_weight=class_weights,			 
			 validation_data = (X_valid_cv, y_valid_cv),
			 callbacks = callbacks)

print(model.evaluate(aa_train_x.values, aa_train_y.values,batch_size=len(aa_train_y.values)))
#print(model.evaluate(aa_train_x.values, aa_train_y.values,batch_size=64))
model.metrics_names
#['loss', 'auc', 'f1']
#1072/1072 [==============================] - 0s 19us/step
#[0.009568675421178341, 0.9606053233146667, 0.9977323412895203]
y_true=aa_train_y.values
y_pred=model.predict_classes(aa_train_x.values, batch_size=64, verbose=0).flatten()
perf_measure(y_true, y_pred)

print(model.evaluate(aa_val_x.values, aa_val_y.values,batch_size=len(aa_val_y.values)))
y_true=aa_val_y.values
y_pred=model.predict_classes(aa_val_x.values, batch_size=64, verbose=0).flatten()
perf_measure(y_true, y_pred)


model.save_weights(filepath)
#model.save(filepath)  # the load_model doesn't work so will not use model.save
print("Saved weigthts to disk")

# load entire model, doesn't work, custom auc is not in the /usr3/graduate/jiayiwu/.local/lib/python3.6/site-packages/keras/utils/generic_utils.py
#model=load_model('AA_nodrugs0.0005l1as350l2as274.hdf5')
# tried to solve it by setting $ model=load_model('AA_nodrugs0.0005l1as350l2as274.hdf5', custom_objects={'auc':auc}), $sys.setrecursionlimit(150000000) but took too long to load


#-----------------------save it to human readble file
jasonpath = "AA_nodrugs"+str(np.round(learning_rate,5))+"l1as"+str(hidden_layer1)+"l2as"+str(hidden_layer2)+".json"
with open(jasonpath, "w") as json_file:
   json_file.write(model.to_json())

#----------------------load model
# load model seprately
#load json and create model
with open('AA_nodrugs0.0005l1as350l2as274.json', 'r') as json_file:
	loaded_model=model_from_json(json_file.read())

# load weight from pre-trained model
loaded_model.load_weights('AA_nodrugs0.0005l1as350l2as274.hdf5')
print('Loaded model from disk')

# evaluate loaded model on train data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc,f1])

print(loaded_model.evaluate(aa_train_x.values, aa_train_y.values,batch_size=len(aa_train_y.values)))

#----------------- test model and weight on test set
print(model.evaluate(aa_test_x.values, aa_test_y.values,batch_size=len(aa_test_y.values)))
print(model.evaluate(aa_test_x.values, aa_test_y.values,batch_size=64))
model.metrics_names
#['loss', 'auc', 'f1']



y_true=aa_test_y.values
y_pred=model.predict_classes(aa_test_x.values, batch_size=64, verbose=0).flatten()

perf_measure(y_true, y_pred)
