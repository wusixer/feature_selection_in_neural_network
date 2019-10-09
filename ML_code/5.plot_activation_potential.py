
import matplotlib
matplotlib.use('Agg') # set matplotlib to not use the Xwindows backend to not plot interactively
import numpy as np
import pandas
import matplotlib.pyplot as plt

aa_rank=pandas.read_csv("aa.ci_plus_average.final.txt", delimiter='\t',header=0 )
# turn interactive plotting off
plt.ioff()
# plot overall
fig = plt.figure()
plt.plot(aa_rank[['rank']], aa_rank[['action_potential']])
# add axs
plt.title("AA All drugs Model")
plt.xlabel("Features (sorted)")
plt.ylabel("Activation Potential")
plt.savefig("../plots/aa_all_drug_action_potential_graph.png")
plt.close(fig)

# plot the one zoomed in to find the activation potential drop, looking from the above graph, choose 250
fig = plt.figure()
plt.plot(aa_rank[['rank']][:250], aa_rank[['action_potential']][:250])
# add axs
plt.title("AA All drugs Model Top ranked 250 Features")
plt.xlabel("Features (sorted)")
plt.ylabel("Activation Potential")
plt.savefig("../plots/aa_all_drug_action_potential_graph_zoomed_in.png")
plt.close(fig)

