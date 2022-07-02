import matplotlib
import torch
import torch.nn.functional as F
import torch.optim as optim
import DQN as dqn
import utils
from environment import EnvManager
import time

"""Main file to train the agent."""

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## initialize hyperparameters and problem setting
    batch_size = 256
    gamma = 0.999
    eps_start = 1
    eps_end = 0.001
    eps_decay = 0.001
    target_update = 5
    memory_size = 200000
    lr = 0.001
    num_episodes = 10000
    steps_per_episode = 360
    wind_turbines = 64
    N = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = EnvManager(device=device,
                    farm_size=wind_turbines,
                    alpha=36,
                    beta=2,
                    periods=N,
                    max_steps=steps_per_episode,
                    c_transport=50)

    ## creates agent and injects critical maintenance ages (cma) from p-ARP
    strategy = dqn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    print(strategy.get_exploration_rate(2000))

    cma = [72, 72, 72, 72, 72, 72, 72, 44, 72, 72, 72, 72]
    agent = dqn.Agent(strategy, em.num_actions_available(), cma, device)
    memory = dqn.ReplayMemory(memory_size)

    ## create policy and target net
    policy_net = dqn.DQN(wind_turbines + 1, wind_turbines + 1).to(device)
    target_net = dqn.DQN(wind_turbines + 1, wind_turbines + 1).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    ## intialize optimizer and decay rate
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
    decayRate = 0.86
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    ## start the training
    episode_costs = []
    seeds = [i for i in range(num_episodes)]
    start_time = time.time()
    for episode in range(num_episodes):
        print("Current Episode: ", episode)
        print(strategy.get_exploration_rate(episode))
        em.set_seed(seeds[episode])
        em.reset()

        scheduled_pm = torch.tensor([0])
        expected_state = em.get_expected_state(scheduled_pm.item())

        # check if target network needs to be updated
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # runs one training episode and updates the DQN
        for timestep in range(steps_per_episode):
            em.breakdowns()
            reward = em.take_action(scheduled_pm.item())
            current_state = em.get_state()
            if expected_state is not None and scheduled_pm.item() != 0: memory.push((expected_state, scheduled_pm.item(), current_state, reward))
            expected_state = current_state
            expected_state[0] = current_state[0]+1 % N
            scheduled_pm = agent.training_policy(expected_state, policy_net)

            # update the DQN at the end of each step
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = dqn.extract_tensors(experiences)

                current_q_values = dqn.QValues.get_current(policy_net=policy_net, states = states, actions=actions)
                next_q_values = dqn.QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ## decay learning rate, store and plot results
        scheduler.step()
        episode_costs.append(em.close())
        agent.update_step()
        utils.plot(episode_costs, 100, is_ipython, benchmark=1.112*wind_turbines*360, no_pm=1.777*360*wind_turbines)
        print(episode_costs[-1])

        ## saves the model if the exploration rate is below 0.03, used as aconvergence criterion
        if strategy.get_exploration_rate(episode) < 0.03:
            print("Model converged after ", time.time() - start_time)
            torch.save(policy_net, "Models/64/c_t50_final_model_2")
            break



