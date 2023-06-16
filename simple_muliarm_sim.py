"""
This Python script implements a Multi-Armed Bandit problem using the epsilon-greedy algorithm with hints. 

The code is structured as follows:

1. Class `MultiArmBandit`: This class represents the multi-armed bandit. It includes:
    - `__init__`: Initializes the bandit with `k` arms, each with a reward value (`q_star`) randomly assigned between 0 and 1. 
    It also accepts a standard deviation parameter (`stdev`) for the Gaussian distribution of rewards.
    - `pull`: Simulates pulling an arm of the bandit, which returns a reward. The reward follows a Gaussian distribution 
    centered at the true reward value (`q_star`) for the chosen arm, with a provided standard deviation (`stdev`).
    - `hint`: Provides a hint about the best action (arm with the maximum reward) with a 50% chance. Otherwise, 
    it randomly selects an action.

2. Function `epsilon_greedy_with_hints`: This function implements the epsilon-greedy strategy with an added hint mechanism.
It accepts the following parameters:
    - `bandit`: An instance of the `MultiArmBandit` class.
    - `epsilon`: The probability with which a random action is selected, representing the exploration rate.
    - `hint_prob`: The probability with which the agent uses a hint (if available) to select an action.
    - `num_steps`: The number of actions (arm pulls) to be performed.
   The function tracks and returns the actions taken, rewards obtained, and hints provided during the interaction with the bandit.

3. Example Usage: This part of the script demonstrates how to create a `MultiArmBandit` instance and 
how to apply the `epsilon_greedy_with_hints` function. It shows the usage both with and without the use of hints and 
prints the average rewards obtained, the hints provided, and the actions taken.

Please note, the random nature of this problem implies that the outputs will vary between different runs of the script.
"""

import random
import numpy as np

class MultiArmBandit:
    def __init__(self, k, q_star=None, stdev=0):
        self.k = k
        self.q_star = [random.uniform(0, 1) for _ in range(k)]
        self.stdev = stdev
        
    def pull(self, action):
        reward = random.gauss(self.q_star[action], self.stdev)
        return reward

    def hint(self):
        if random.random() < 0.5:
            return self.q_star.index(max(self.q_star))
        else:
            return random.randint(0, self.k-1)
    
def epsilon_greedy_with_hints(bandit, epsilon, hint_prob, num_steps):
    action_counts = [0 for _ in range(bandit.k)]
    q_estimates = [0 for _ in range(bandit.k)]
    rewards = []
    actions = []
    hints = []
    
    for i in range(num_steps):
        if random.uniform(0, 1) < hint_prob:
            action = bandit.hint()
        elif random.uniform(0, 1) < epsilon:
            action = random.randint(0, bandit.k-1)
        else:
            action = q_estimates.index(max(q_estimates))
        reward = bandit.pull(action)
        rewards.append(reward)
        actions.append(action)
        hints.append(bandit.hint())
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action])/action_counts[action]
        
    return (hints, actions, rewards)

# Example usage:
bandit = MultiArmBandit(3, q_star=[0.6, 0.1, 0.6])
printed_rewards = ["{:.1f}".format(_) for _ in bandit.q_star]
print(f"Bandit reward means: {printed_rewards}")
(hints, actions, rewards) = epsilon_greedy_with_hints(bandit, 0.1, 0.0, 20)
printed_eg_reward = "{:.1f}".format(sum(rewards)/len(rewards))
print("Average reward using epsilon greedy:", printed_eg_reward)
print(f"Hints: {hints}")
print(f"Actions: {actions}")
(hints, actions, rewards) = epsilon_greedy_with_hints(bandit, 0.0, 0.1, 20)
print("Average reward using greedy with hints:", "{:.1f}".format(sum(rewards)/len(rewards)))
#print(rewards)
print(f"Hints: {hints}")
print(f"Actions: {actions}")

