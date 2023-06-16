import numpy as np
from scipy.stats import gaussian_kde, entropy

class POMDP:
    def __init__(self, num_states, num_actions, num_observations):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.transition_probs = np.random.rand(num_states, num_states, num_actions)
        self.observation_probs = np.random.rand(num_states, num_observations)
        # Normalize the transition probabilities so they sum to 1
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)
        # Normalize the observation probabilities so they sum to 1
        self.observation_probs /= self.observation_probs.sum(axis=1, keepdims=True)
    def step(self, state, action):
        next_state = np.random.choice(self.num_states, p=self.transition_probs[state,:,action])
        observation = np.random.choice(self.num_observations, p=self.observation_probs[next_state])
        return next_state, observation

def data_simulator(pomdp, num_steps, initial_state, policy):
    states = np.zeros(num_steps, dtype=int)
    actions = np.zeros(num_steps, dtype=int)
    observations = np.zeros(num_steps, dtype=int)
    states[0] = initial_state
    for t in range(num_steps-1):
        actions[t] = policy(states[t])
        states[t+1], observations[t+1] = pomdp.step(states[t], actions[t])
    actions[-1] = policy(states[-1])
    return states, actions, observations

def kl_divergence(p, q, base=None):
    return entropy(p, q, base=base)

# Define two POMDPs and a random policy
pomdp1 = POMDP(2, 2, 2)
pomdp2 = POMDP(2, 2, 2)
policy = lambda state: np.random.choice(2)

# Simulate data from the POMDPs
num_steps = 1000
initial_state = 0
data1 = data_simulator(pomdp1, num_steps, initial_state, policy)
data2 = data_simulator(pomdp2, num_steps, initial_state, policy)

# Flatten the data and estimate the PDFs
data1_flat = np.hstack(data1)
data2_flat = np.hstack(data2)
pdf1 = gaussian_kde(data1_flat)
pdf2 = gaussian_kde(data2_flat)

# Evaluate the PDFs on a linear space
x = np.linspace(min(min(data1_flat), min(data2_flat)), max(max(data1_flat), max(data2_flat)), num=num_steps)
p = pdf1.evaluate(x)
q = pdf2.evaluate(x)

# Calculate the KLD
kld = kl_divergence(p, q)
print(f"The Kullback-Leibler Divergence between the two POMDP traces is {kld}")

