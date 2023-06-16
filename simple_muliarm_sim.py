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

