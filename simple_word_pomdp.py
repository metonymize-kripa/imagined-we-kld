"""
WordBasedPOMDPAgent: A simplified POMDP agent in a text-based environment.

This class represents a Partially Observable Markov Decision Process (POMDP) agent, 
which learns and operates in a text-based environment where states and observations are represented 
as sentences and actions are represented by a limited list of verbs. 

The agent keeps track of:

- Transition probability matrix (A): Defines the probabilities of transitioning from 
  one state to another given an action.

- Observation probability matrix (B): Defines the probabilities of observing a 
  certain state given the current state. In this case, the observations are the same as the states.

- Preference matrix (C): Defines the agent's preferences over observations. 

- Policy matrix (D): Defines the agent's policy, i.e., the probabilities of taking a certain 
  action given the current state and the previous action.

The agent can take the following actions:

- Update the A and B matrices based on the current state, action, and next state.

- Choose the next state based on the current state and action.

- Choose an action based on the current state and the policy matrix.

- Get an observation (which is the same as the state in this case) based on the current state.

Example:
    verbs = ["approach", "retreat", "nothing"]
    sentences = ["The cat is on the mat.", "The cat is under the table.", "The cat is sleeping.", "The cat is eating."]
    agent = WordBasedPOMDPAgent(verbs, sentences)
"""

import numpy as np

class WordBasedPOMDPAgent:
    def __init__(self, verbs, sentences, transition_matrix, observation_matrix, preference_matrix, policy_matrix):
        self.verbs = verbs  # List of possible actions
        self.sentences = sentences  # List of possible states
        self.transition_matrix = transition_matrix  # Transition probability matrix A
        self.observation_matrix = observation_matrix  # Observation probability matrix B
        self.preference_matrix = preference_matrix  # Preference matrix C
        self.policy_matrix = policy_matrix  # Policy matrix D
        self.current_state = np.random.choice(len(self.sentences))  # Initialize the agent at a random state

    def take_action(self):
        # Choose an action based on the policy matrix for the current state
        action = np.random.choice(self.verbs, p=self.policy_matrix[self.current_state])
        return action

    def update_state(self, action):
        # Update the state based on the transition probabilities and the chosen action
        action_index = self.verbs.index(action)
        self.current_state = np.random.choice(len(self.sentences), p=self.transition_matrix[self.current_state, :, action_index])

    def simulate(self, rounds=5):
        for i in range(rounds):
            action = self.take_action()
            print(f"Round {i+1}:")
            print(f"Starting state: {self.sentences[self.current_state]}")
            print(f"Action: {action}")
            self.update_state(action)
            print(f"New state: {self.sentences[self.current_state]}")
            print("------------------")

n_states = 4
n_actions = 3
verbs = ["approach", "retreat", "nothing"]
sentences = ["The cat is on the mat.", "The cat is under the table.", "The cat is sleeping.", "The cat is eating."]

A = np.full((n_states, n_states, n_actions), 1/n_states) #state transition based on actions
B = np.full((n_states, n_states), 1/n_states) #assume one to one mapping between observation and state for now
C = np.array([0.2, 0.3, 0.2, 0.3]) #assume preference for cat under the table and cat eating
D = np.zeros((n_states, n_actions))
D[0, 0] = 1  # approach when cat is on the mat
D[1, 1] = 1  # retreat when cat is under the table
D[2, 0] = 1  # approach when cat is sleeping
D[3, 1] = 1  # retreat when cat is eating

agent = WordBasedPOMDPAgent(verbs, sentences, A, B, C, D)

# Simulate 5 rounds of actions and transitions
agent.simulate()

