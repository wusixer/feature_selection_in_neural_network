#!/usr/bin/env python3
# because loading csv with preprocessing change the column order all the time,
# a solution is we do preprocessing before hand and save it to file for future
# direct use in keras
import numpy as np
np.set_printoptions(threshold=np.inf) # setting the print threshold to infinity
import pandas as pd
from sklearn.pipeline import Pipeline
import sys
#import pdb

# read in data
aa_train=pd.read_csv("/aa_dev.txt", delimiter="\t")
aa_val=pd.read_csv("/aa_val.txt", delimiter="\t")
aa_test=pd.read_csv("/phenotype/aa.ml.test.z.2018.12.txt", delimiter="\t")
# check demension
aa_train.shape; aa_val.shape; aa_test.shape
#(918, 3666),(154,3666),(120, 3666)

## select columns needed
B_drop = [[i for i in aa_train.columns[aa_train.columns.str.startswith('B7')]][index] for index in [0,2,4,6,8,11,13,16,18,21,23,24,25,26,27,28]]

G = [i for i in aa_train.columns[aa_train.columns.str.startswith('G')]][:219]
G1 = [i for i in G if i in aa_train.columns[aa_train.columns.str.endswith('R')]]
G2 = [i for i in G if i in aa_train.filter(regex=('RC')).columns]
G3 = [i for i in G if i in aa_train.filter(regex=('Yrs')).columns]

H1 = [i for i in aa_train.columns[aa_train.columns.str.startswith('H')] if i in aa_train.columns[aa_train.columns.str.endswith('RC')]] 
H2 = [i for i in aa_train.columns[aa_train.columns.str.startswith('H')] if i in aa_train.columns[aa_train.columns.str.endswith('R')]]
H3 = [i for i in aa_train.columns[aa_train.columns.str.startswith('H')] if i in aa_train.filter(regex=('OTH')).columns]

drop = B_drop + G1 + G2+ G3 + H1 +H2+H3  + ["opices","index","train","G1C_OpiUseYr","G1C_1_OpiUse11Yr_1","G1C_1_OpiUse11Yr_5","SSADDA_ID"]


final = list(set([i for i in aa_train.columns if i not in drop]))

# get the dataset for analyisis
aa_train_y = aa_train['opices']
aa_train_x = aa_train[final]


print("aa_train_x")
print(aa_train_x.shape)
aa_train_x.shape
# (918, 3157)
aa_val_y = aa_val['opices']
aa_val_x = aa_val[final]
print("aa_val_x")
print(aa_val_x.shape)
aa_val_x.shape
# (154,3157)
aa_test_y = aa_test['opices']
aa_test_x = aa_test[final]
print('aa_test_x')
print(aa_test_x.shape)
#(120, 3157)
# save dataset to file 

aa_train_x.to_csv('/all_drug/aa_train_x.csv', na_rep='NA', index=False)
aa_val_x.to_csv('/all_drug/aa_val_x.csv', na_rep='NA', index=False)
aa_test_x.to_csv('/all_drug/aa_test_x.csv', na_rep='NA', index=False)

aa_train_y.to_csv('/all_drug/aa_train_y.csv', na_rep='NA', index=False)
aa_val_y.to_csv('/all_drug/aa_val_y.csv', na_rep='NA', index=False)
aa_test_y.to_csv('/all_drug/aa_test_y.csv', na_rep='NA', index=False)

#######-------------------------for ea

ea_train=pd.read_csv("/ea_dev.txt", delimiter="\t")
ea_val=pd.read_csv("/ea_val.txt", delimiter="\t")
ea_test=pd.read_csv("/phenotype/ea.ml.test.z.2018.12.txt", delimiter="\t")
#check demension
ea_train.shape  #, ea_test.shape
#(2310, 3856)

## select columns needed
B_drop = [[i for i in ea_train.columns[ea_train.columns.str.startswith('B7')]][index] for index in [0,2,4,6,8,10,12,14,16,18,20,22,23,24]]
G = [i for i in ea_train.columns[ea_train.columns.str.startswith('G')]][:263]
G1 = [i for i in G if i in ea_train.columns[ea_train.columns.str.endswith('R')]]
G2 = [i for i in G if i in ea_train.filter(regex=('RC')).columns]
G3 = [i for i in G if i in ea_train.filter(regex=('Yrs')).columns]

H1 = [i for i in ea_train.columns[ea_train.columns.str.startswith('H')] if i in ea_train.columns[ea_train.columns.str.endswith('RC')]]
H2 = [i for i in ea_train.columns[ea_train.columns.str.startswith('H')] if i in ea_train.columns[ea_train.columns.str.endswith('R')]]
H3 = [i for i in ea_train.columns[ea_train.columns.str.startswith('H')] if i in ea_train.filter(regex=('OTH')).columns]

drop = B_drop + G1 + G2+ G3 + H1 +H2+H3  + ["opices","index","train","G1C_OpiUseYr","G1C_1_OpiUse11Yr_1","G1C_1_OpiUse11Yr_5","SSADDA_ID"]


final = list(set([i for i in ea_train.columns if i not in drop]))

#get the dataset for analyisis
ea_train_y = ea_train['opices']
ea_train_x = ea_train[final]
print("ea_train_x")
print(ea_train_x.shape)

ea_val_y=ea_val['opices']
ea_val_x=ea_val[final]

ea_train_x.shape
ea_test_y = ea_test['opices']
ea_test_x = ea_test[final]

print('ea_test_x.shape')
print(ea_test_x.shape)
ea_train = None

ea_train_x.to_csv('/all_drug/ea_train_x.csv', na_rep='NA', index=False)
ea_val_x.to_csv('/all_drug/ea_val_x.csv', na_rep='NA', index=False)
ea_test_x.to_csv('/all_drug/ea_test_x.csv', na_rep='NA', index=False)

ea_train_y.to_csv('/all_drug/ea_train_y.csv', na_rep='NA', index=False)
ea_val_y.to_csv('/all_drug/ea_val_y.csv', na_rep='NA', index=False)
ea_test_y.to_csv('/all_drug/ea_test_y.csv', na_rep='NA', index=False)

print('---------------------finish loading data -------------------------')
