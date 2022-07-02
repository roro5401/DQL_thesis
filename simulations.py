from data_generator import create_dataset
from environment import EnvManager
import random
import policies
import torch
import numpy as np
import DQN as dqn
import xlsxwriter

"""
This class provides a simulator used to compare policies. The following functions are available:

1. __init__(): initializes a simulator
2. simulation(): runs one simulation
3. compare_policies(): compares policies under equal circumstances.
"""
class simulator():

    """
    Initiliazes an instance os the simulator class.

    @param device: device on which the simulations are run.
    @param farm_size: number of wind turbines considered in the simulation.
    @param steps_per_run: duration in months per simulation.
    @param periods_per_year: number of periods per year.
    @param alpha: scale parameter of the Weibull distribution.
    @param beta: shape parameter of the Weibull distribution.
    @param c_transport: set-up costs for the environment.
    """
    def __init__(self, device, farm_size, steps_per_run, periods_per_year, alpha, beta, c_transport):
        self.em = EnvManager(device=device, farm_size=farm_size, alpha=alpha, beta=beta,
                             periods=periods_per_year, max_steps=steps_per_run, c_transport=c_transport)

    """
    Runs a simulation for the given policy.
    
    @param policy: policy for selecting actions.
    @param np_seed: numpy seed for generating data set.
    @return total costs, actions of the policy.
    """
    def simulation(self, policy, np_seed=0):
        actions = []
        self.em.set_seed(np_seed)
        self.em.reset()

        scheduled_pm = torch.tensor([0])
        expected_state = self.em.get_expected_state(scheduled_pm.item())

        for time in range(0, self.em.max_steps):
            self.em.breakdowns()
            actions.append(Action(scheduled_pm, self.em.broken_turbines(), self.em.get_state()[1:scheduled_pm+1], self.em.get_state()[0]+1))
            reward = self.em.take_action(scheduled_pm)
            current_state = self.em.get_state()
            expected_state = current_state
            expected_state[0] = (current_state[0] + 1) % 12
            scheduled_pm = policy.policy(expected_state)

        return self.em.close(), actions

    """
    Compares policies for a given number of simulations. 
    
    @param policies: list of policies that have a policy function which takes in a state and outputs an action.
    Additionally, the policies need to have a name property. Add an if statement and list for each policy you want
    to compare.
    @param seeds: seeds for generating data, needed for comparing the policies under equal data sets.
    @return list of the runs of the given policies.
    """
    def compare_policies(self, policies: list, seeds : list):
        no_pm_runs = []
        arp_runs = []
        dqn_runs = []
        mbrp_runs = []
        for seed in seeds:
            for policy in policies:
                cost, actions = self.simulation(policy, seed)
                if policy.name == "DQN": dqn_runs.append(Run("DQN", actions, cost, seed))
                elif policy.name == "No PM": no_pm_runs.append(Run(policy.name, actions, cost, seed))
                elif policy.name == "ARP": arp_runs.append(Run(policy.name, actions, cost, seed))
                elif policy.name == "MBRP": mbrp_runs.append(Run(policy.name, actions, cost, seed))
        return no_pm_runs, arp_runs, dqn_runs, mbrp_runs

"""
This class is used to store an action.
"""
class Action():

    """
    Initiliazes an instance of the action class. The following data is saved:

    @param pm: number of preventive maintenances performed.
    @param cm: number of corrective maintenances performed.
    @param ages: list of ages of the preventively maintained wind turbines
    """
    def __init__(self, pm, cm, ages, timestep):
        self.pm = pm
        self.cm = cm
        self.ages = ages
        self.timestep = timestep

"""
This class is used to store an entire simulation run. Some functions are available to acquire some statistics of 
the run. 
"""
class Run():

    """
    Initializes an instance of the Run() class.

    @param policy: name of the policy that did the run.
    @param actions: list of actions that took place during the run.
    @param costs: total cost at the end of the run.
    @param seed: numpy seed under which the data was generated.
    """
    def __init__(self, policy, actions, cost, seed):
        self.policy = policy
        self.actions = actions
        self.cost = cost
        self.seed = seed

    """
    Returns the ages at which PM was performed.
    """
    def average_pm_age(self):
        age = []
        for action in self.actions:
            age.extend(action.ages)
        return age

    """
    Returns the number of PMs per period.
    """
    def pm_per_period(self):
        pm = []
        for action in self.actions:
            pm.append(action.pm)
        return pm

    """
    Returns the number of PMs per period, excluding zeroes.
    """
    def average_pms_at_once(self):
        pm = []
        for action in self.actions:
            if action.pm != 0: pm.append(action.pm)
        return pm

    """
    Prints some statistics for the run.
    """
    def print_all_stats(self):
        pm_at_once = self.average_pms_at_once()
        pm_per_period = self.average_pms_at_once()
        pm_age = self.average_pm_age()
        print("Average PM age: ", np.mean(pm_age), " Variance: ", np.var(pm_age))
        print("Average PM per period: ", np.mean(pm_per_period), " Variance: ", np.var(pm_per_period))
        print("Average PM when PM: ", np.mean(pm_at_once), " Variance: ", np.var(pm_at_once))

"""
This function is used to summarize and store results in an excel file. Two files are created, one with PMs and their
respective age. It is in the following format: [period 1, age of PM1, age of PM2, ..., age of PM 32] where ages are 
only specified when a PM is performed for that number. The other file stores the statistics regarding CMs in the 
following form [period, CMs in that period]. Both files contain all actions from all simulations after each other.
"""
def summarize_results(results : list, pm_file_name : str, cm_file_name : str):
    pm_ages = []
    cm = []
    for run in results:
        for action in run.actions:
            if action.pm > 0:
                state = [action.timestep]
                state.extend(action.ages)
                pm_ages.append(state)
            else:
                pm_ages.append([action.timestep, 0])
            cm.append([action.timestep, action.cm])

    with xlsxwriter.Workbook(pm_file_name) as workbook:
        worksheet = workbook.add_worksheet()
        for row_num, data in enumerate(pm_ages):
            worksheet.write_row(row_num, 0, data)

    with xlsxwriter.Workbook(cm_file_name) as workbook:
        worksheet = workbook.add_worksheet()
        for row_num, data in enumerate(cm):
            worksheet.write_row(row_num, 0, data)



## define policies and problem setting
M = 10000
cma = [M, M, M, M, M, M, M, 44, M, M, M, M]
cma_dqn = [72, 72, 72, 72, 72, 72, 72, 44, 72, 72, 72, 72]
cma_mbrp = [M, M, M, M, M, M, 42, M, M, M, M, M]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arp = policies.ARP(cma)
mbrp = policies.ARP(cma_mbrp, name = "MBRP")
no_pm = policies.ARP([M for i in range(0, 12)], name = "No PM")
steps_per_run = 360
farm_size =64
agent = dqn.Agent(strategy = None, num_actions = None, critical_maintenance_ages = cma_dqn, device = device, policy_path ="Models/64/c_t50_final_model_2")

## define simulator and compare policies
sim = simulator(device=device,
                farm_size=farm_size,
                steps_per_run=steps_per_run,
                periods_per_year=12,
                alpha=36,
                beta=2,
                c_transport=50)

no_pm_results, arp_results, dqn_results, mbrp_results = sim.compare_policies([agent], seeds = [i for i in range(0, 100)])
summarize_results(no_pm_results, pm_file_name="No PM - PM.xlsx", cm_file_name="No PM - CM.xlsx")
summarize_results(arp_results, pm_file_name="ARP - PM.xlsx", cm_file_name="ARP - CM.xlsx")
summarize_results(dqn_results, pm_file_name="DQN - PM.xlsx", cm_file_name="DQN - CM.xlsx")
summarize_results(mbrp_results, pm_file_name="MBRP - PM.xlsx", cm_file_name="MBRP - CM.xlsx")

## Uncomment to print some states


# total_costs = 0
# n = 0
# no_pm_costs = []
# pm_ages = []
# cm = []
# for run in no_pm_results:
#     total_costs += run.cost/(farm_size*steps_per_run)
#     no_pm_costs.append(run.cost/(farm_size*steps_per_run))
#     n += 1
#     for action in run.actions:
#         if action.pm > 0:
#             pm_ages.append(action.ages)
#         cm.append(action.cm)
# print(pm_ages)
# res = [['first', 'rest']]
# res.extend(pm_ages)
# print(res)
#
# with xlsxwriter.Workbook('stats.xlsx') as workbook:
#     worksheet = workbook.add_worksheet("No PM")
#     for row_num, data in enumerate(res):
#         worksheet.write_row(row_num, 0, data)
#
# with xlsxwriter.Workbook('cm_stats.xlsx') as workbook:
#     worksheet = workbook.add_worksheet("No PM")
#     for row_num, data in enumerate(cm):
#         worksheet.write_row(row_num, 0, data)
#
# #print("No PM: ", total_costs/n, " (", np.var(no_pm_costs), ")")
#
# total_costs = 0
# n = 0
# arp_costs = []
# pm_ages = []
# cm = []
# for run in arp_results:
#     total_costs += run.cost/(farm_size*steps_per_run)
#     arp_costs.append(run.cost/(farm_size*steps_per_run))
#     n += 1
#     for action in run.actions:
#         if action.pm > 0:
#             pm_ages.append(action.ages)
#         cm.append(action.cm)
#
# #print("pARP: ", total_costs/n, " (", np.var(arp_costs), ")")
#
total_costs = 0
n = 0
dqn_costs = []
pm_ages = []
cm = []
for run in dqn_results:
    total_costs += run.cost/(farm_size*steps_per_run)
    n += 1
    dqn_costs.append(run.cost/(farm_size*steps_per_run))
    for action in run.actions:
        if action.pm > 0:
            pm_ages.append(action.ages)
        cm.append(action.cm)
print("DQN: ", total_costs/n, " (", np.var(dqn_costs), ")")


print("Variance of DQN costs: ")




