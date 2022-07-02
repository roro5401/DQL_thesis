import torch
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt

"""
This file helps in the state analysis.
"""

## create state to be analyzed and matrix to store outputs
output = np.zeros((12, 65))
state = [1 for i in range(0, 64)]

## load in the policy net and analyze the states
policy_net = torch.load("Models/64/c_t30_final_model")
for i in range(0, 12):
    s_i = [i+1]
    s_i.extend(state)
    out = policy_net.forward(s_i)
    output[i,:] = out.detach().numpy()

## plot the heatmap
ax = sns.heatmap(data=output, linewidth=0.5, yticklabels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set(xlabel="Scheduled PM", ylabel="Month")
plt.show()




