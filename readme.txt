This Readme outlines the use of each file in the directory. The following files are included:

data_generator.py: includes a function to generate a dataset as described in section 3 of the corresponding paper.
DQN.py: includes all functionality needed for building the DQN as described in section 4 of the original paper. This includes: Deep Q Network, Experiences, 
Agent, Replay Memory,  Epsilon Greedy Strategy, Q-value functions, and some utility functions for training.
environment.py: includes all necessary functions for the environment. The environment is described in section 4.3.1.
main.py: includes the training loop for the DQL agent.
policies.py: includes policies as presented by Schouten et. al (2022), which take in a state of a large wind farm and apply the policies to each wind
turbine individually. More details can be found in section 4.2 of the corresponding paper.
replication.py: contains the linear formulations of the policies in the aformentioned file.
simulations.py: includes all functionality to run simulations and compare the performance of multiple policies under equal circumstances.
state analysis.py: includes a few lines of code to generate a heat map from the outputted q-value approximations of the DQN.
utils.py: includes a few plotting functions.

Note that, if you would like to load in a DQN in state analysis.py or simulations.py, make sure that the architecture of the DQN is implemented in the DQN file, otherwise the weight
dictionary will not load appropriately. All models that were generated for the corresponding paper are saved under the Models folder. Some problem instances include multiple models as
the some hyperparameters were changed due to insatisfactory results from the original hyperparameters.
