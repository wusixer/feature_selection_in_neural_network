
import matplotlib
matplotlib.use('Agg') # set matplotlib to not use the Xwindows backend to not plot interactively
import numpy as np
import pandas
import matplotlib.pyplot as plt

aa_train_val_value=pandas.read_csv("logs/4.aa.var_selection_validation.sorted.txt", delimiter='\t',header=0 )
# turn interactive plotting off
plt.ioff()
# plot overall
fig = plt.figure()
plt.plot(aa_train_val_value[['top_n_var']], aa_train_val_value[['weighted_auc_train']], color='olive', linewidth=2, label="weighted auc on training set")
plt.plot(aa_train_val_value[['top_n_var']], aa_train_val_value[['weighted_auc_val']], color='red', linewidth=2, label="weighted auc on validation set")
#plt.plot(236,0.53, marker='o', markersize=3, color='red', label="weighted auc on test set")
plt.ylim(0,1)
plt.legend()

# add axs
plt.title("AA No Drug Model Training and Validation AUC score")
plt.xlabel("Features (sorted)")
plt.ylabel("Weighted AUC Score")
plt.savefig("plots/aa_no_drug_training_and_val_weighted_auc.png")
plt.close(fig)

















