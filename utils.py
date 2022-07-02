import torch
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

"""
This file contains two functions to plot the moving average during training times. No further explanation is provided
as they are rather self-explanatory.
"""
## PLOTTING ##

def plot(values, moving_avg_period, is_ipython, benchmark = None, no_pm = None):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.plot(values, label = "Agent")
    plt.plot(get_moving_average(moving_avg_period, values), label = "Moving Average")
    if benchmark is not None:
        plt.axhline(y=benchmark, label = "p-ARP", color = "purple")
    if no_pm is not None:
        plt.axhline(y=no_pm, label = "No PM", color = "grey")
    plt.legend()
    plt.pause(0.001)
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype = torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size = period, step = 1).mean(dim=1).flatten(start_dim = 0)
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

