import math
import random
import numpy

import torch

import policies
from data_generator import create_dataset

"""
This class manages the environment for the simulations. All things regarding the wind turbines and interacting
with them can be found in here. 

1. __init__(): initiliazes the environment
2. reset(): resets the environment
3. sort_data(): sorts wind turbines based on age (descending)
4. close(): ends the episode and returns the total costs
5. num_actions_available(): returns the number of available actions
6. take_action(): maintains the farm based on a specified number of PMs, CMs are always done
7. breakdowns(): checks which wind turbines break down at the start of the current period
8. preventice_maintenance(): used in take_action() to do the PMs
9. corrective_maintenance(): used in take_action() to do the CMs
10. get_state(): returns the current state of the environment
11. get_expected_state(): returns the estimated expected state at the start of next period, i.e. after these month's 
    maintenance is finished
12. c_pm(): returns the total costs for a specified number of PMs
13. c_cm(): returns the total costs of the required CMs
14. set_seed(): sets the seed used to generate wind turbine lifetimes
15. pm_ages(): returns the ages at which PM is performed
"""

class EnvManager():
    """
    Initializes the environment.

    @param device: reference to the GPU on which you want to run your model
    @param farm_size: number of wind turbines on the farm
    @param alpha: scale parameter for weibull distribution
    @param beta: shape parameter for the weibull distribution
    @param periods: number of periods in one year
    @param max_steps: number of steps taken per simulation
    @param c_transport: the set-up costs, incurred for every CM and only once regardless of how many PMs
    """
    def __init__(self, device, farm_size, alpha, beta, periods, max_steps, c_transport):
        self.c_transport = c_transport
        self.max_steps = max_steps
        self.t = 1
        self.N = periods
        self.cost = 0
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.size = farm_size
        self.seed = 0
        self.reset()
        self.data = None

    """
    Resets the environment, setting costs to 0, t to 1, and generating new unique data for the wind turbine lifetimes.
    """
    def reset(self):
        self.t = 1
        self.cost = 0

        # generate new data according to a different seed to keep the experiences diverse
        self.seed += 1
        self.data = create_dataset(self.alpha, self.beta, self.size, self.max_steps+1, self.seed)

    """
    Sorts the data of wind turbines by age, descending order.
    """
    def sort_data(self):
        self.data = self.data.sort_values(by='age', ascending=False)

    """
    Closes the environment, i.e. returning total costs.
    """
    def close(self):
        return self.cost

    """
    Returns the number of actions available. This is size + 1 because we can schedule maintenance up to size including 0.
    """
    def num_actions_available(self):
        return self.size + 1

    """
    Takes the planned action, i.e. the schedules PMs.
    @param pm: tje number of PMs to be performed in this period.
    @return c_t: returns the total costs from this action, tensor data type is used for training the DQN later on.
    """
    def take_action(self, pm):
        c_pm = self.preventive_maintenance(pm)
        c_cm = self.corrective_maintenance()
        c_t = c_cm + c_pm
        return torch.tensor([c_t], device=self.device)

    """
    This function breaks down the wind turbines that are supposed to break down in the current period.
    """
    def breakdowns(self):
        self.t += 1
        for id, row in self.data.iterrows():
            #print(self.t, self.data.at[id, 'next_failure'])
            if self.data.at[id, 'next_failure'] == self.t:
                self.data.at[id, 'age'] = 0
                #print("breakdown")

    """
    Performs the preventive maintenance, i.e. alters the wind turbine ages. It sets the age to 1 for the maintained
    turbines, sets their next failure to self.t + next life time, and removes the first element from the lifetime 
    vector. Note that PM is performed on the oldest wind turbines as they yield the most benefit.
    @param pm: number of PMs to be performed.
    """
    def preventive_maintenance(self, pm):
        i = 0
        self.sort_data()
        for id, row in self.data.iterrows():
            if i < pm:
                lt = self.data.at[id, 'lifetimes']
                self.data.at[id, 'next_failure'] = self.t + lt[0]
                self.data.at[id, 'lifetimes'] = lt[1:]
                self.data.at[id, 'age'] = 1
            elif self.data.at[id, 'age'] != 0:
                self.data.at[id, 'age'] += 1
        c_pm = self.c_pm(pm)
        self.cost += -c_pm
        self.sort_data()
        return c_pm

    """
    Performs the corrective maintenance, i.e. alters the wind turbine ages. It sets the age to 1 for the maintained
    turbines, sets their next failure to self.t + next life time, and removes the first element from the lifetime 
    vector. Note that we do not need to specify an input parameter as the number of CMs can simply be retreived by
    looking at the broken turbines.
    """
    def corrective_maintenance(self):
        self.sort_data()
        cm = self.broken_turbines()
        #print("cms, ", cm)
        #print(self.data.tail(cm))
        if cm > 0:
            for id, row in self.data.tail(n=cm).iterrows():
                lt = self.data.at[id, 'lifetimes']
                self.data.at[id, 'next_failure'] = self.t + lt[0]
                self.data.at[id, 'lifetimes'] = lt[1:]
                self.data.at[id, 'age'] = 1
        c_cm = self.c_cm(cm)
        self.cost += -c_cm
        return c_cm

    """
    Returns the current state of the system. The state is in the form [period, wind turbine lifetimes].
    """
    def get_state(self):
        self.sort_data()
        s_t = [self.t % self.N + 1]
        s_t.extend(self.data['age'].tolist())
        return s_t

    """
    Returns the expected state at the start of the next period. The state is in the form [period, wind turbine lifetimes].
    """
    def get_expected_state(self, pm):
        s_t1 = [(self.t + 1) % self.N]
        self.sort_data()
        ages = self.data['age'].tolist()
        for i in range(0, len(ages)):
            if i < pm:
                ages[i] = 1
            else:
                ages[i] += 1
        return s_t1.extend(ages)

    """
    Calculates the cost for a given number of PMs.
    """
    def c_pm(self, pm):
        if pm == 0: return 0
        t = self.get_state()[0]
        c_pm = 10 + 2 * math.cos(2 * math.pi * t / self.N - 2 * math.pi / self.N)
        c_t = self.c_transport + pm * c_pm
        return -c_t

    """
    Calculates the cost for a given number of CMs.
    """
    def c_cm(self, cm):
        if cm == 0: return 0
        t = self.get_state()[0]
        c_cm = 50 + 10 * math.cos(2 * math.pi * t / self.N - 2 * math.pi / self.N)
        c_t = cm * self.c_transport + cm * c_cm
        return -c_t

    """
    Returns the number of broken turbines, i.e. turbines with an age of 0.
    """
    def broken_turbines(self):
        cm = 0
        for i in range(len(self.data)):
            if self.data.loc[i, 'age'] == 0:
                cm += 1
        #print("CMs: ", cm)
        return cm

    """
    Set the seed for the data generator.
    """
    def set_seed(self, seed):
        self.seed = seed

    """
    Return the ages of the wind turbines on which PM was performed.
    """
    def pm_ages(self, pm):
        ages = []
        self.sort_data()
        if pm != 0:
            for i, row in self.data.head(pm).iterrows():
                ages.append(self.data.loc[i, 'age'])
        return ages
