
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
plt.plot(aa_train_val_value[['top_n_var']], aa_train_val_value[['weighted_f1_train']], color='olive', linewidth=2, label="weighted f1 on training set")
plt.plot(aa_train_val_value[['top_n_var']], aa_train_val_value[['weighted_f1_val']], color='red', linewidth=2, label="weighted f1 on validation set")
plt.axvline(x=72, linestyle='dashed', markersize=2, color='black')
#72      0.877659022808075       0.8232266306877136      0.8774092793464661      0.7482990026473999      0.8772946000099182      0.7254903
plt.ylim(0,1)
plt.legend()

# add axs
plt.title("AA All Drugs Model Training and Validation F1 score")
plt.xlabel("Features (sorted)")
plt.ylabel("Weighted F1 Score")
plt.savefig("plots/aa_all_drug_training_and_val_weighted_f1.png")
plt.close(fig)













