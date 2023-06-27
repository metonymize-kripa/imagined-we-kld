"""
The script defines three classes: WordBasedPOMDPAgent, LearningPOMDPAgent, and ExpandingPOMDPAgent, each of which extends the one before it.

WordBasedPOMDPAgent: This class models an agent operating in a Partially Observable Markov Decision Process (POMDP) environment. The agent has a set of states (sentences), actions (verbs), and uses matrices to store transition probabilities between states (transition_matrix), observation probabilities (observation_matrix), preferences for each state (preference_matrix), and a policy matrix (policy_matrix). The agent takes actions based on its policy and transitions between states based on its transition matrix.

LearningPOMDPAgent: This class extends WordBasedPOMDPAgent by introducing a learning factor. This agent can learn from its environment, updating its transition matrix according to the actual observed transitions.

ExpandingPOMDPAgent: This class extends LearningPOMDPAgent and allows the agent to expand its set of states and actions based on some pre-defined environment sentences and verbs. The agent has a chance to learn new verbs and sentences from its environment after each round of simulation.

Finally, an instance of ExpandingPOMDPAgent is created with some initial states, actions, transition matrix, observation matrix, preference matrix, and policy matrix. The agent is then run for a few rounds of simulation, during which it may take actions, transition between states, learn from the environment, and possibly expand its known states and actions.


"""
# Import the necessary libraries
import numpy as np, random

# The base class for the agent
class WordBasedPOMDPAgent:
    # Initialize the agent with the necessary parameters
    def __init__(self, verbs, sentences, transition_matrix, observation_matrix, preference_matrix, policy_matrix):
        self.verbs = verbs  # List of possible actions
        self.sentences = sentences  # List of possible states
        self.transition_matrix = transition_matrix  # Transition probability matrix A
        self.observation_matrix = observation_matrix  # Observation probability matrix B
        self.preference_matrix = preference_matrix  # Preference matrix C
        self.policy_matrix = policy_matrix  # Policy matrix D
        self.current_state = np.random.choice(len(self.sentences))  # Initialize the agent at a random state

    # Method for the agent to take an action
    def take_action(self):
        # Normalize the policy for the current state
        policy = self.policy_matrix[self.current_state]
        policy /= policy.sum()
        # Choose an action based on the policy matrix for the current state
        action = np.random.choice(self.verbs, p=self.policy_matrix[self.current_state])
        return action

    # Method for updating the agent's state
    def update_state(self, action):
        # Normalize the transition matrix
        self.transition_matrix /= self.transition_matrix.sum(axis=2, keepdims=True)
        # Update the state based on the transition probabilities and the chosen action
        action_index = self.verbs.index(action)
        self.current_state = np.random.choice(len(self.sentences), p=self.transition_matrix[self.current_state, :, action_index])

    # Method for simulating the agent's actions and transitions
    def simulate(self, rounds=5):
        for i in range(rounds):
            action = self.take_action()
            print(f"Round {i+1}:")
            print(f"Starting state: {self.sentences[self.current_state]}")
            print(f"Action: {action}")
            self.update_state(action)
            print(f"New state: {self.sentences[self.current_state]}")
            print("------------------")

# An extension of the base class which includes learning
class LearningPOMDPAgent(WordBasedPOMDPAgent):
    # Initialize the learning agent
    def __init__(self, *args, learning_rate=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate  # The rate at which the agent learns from new observations

    # Overwrite the update_state method to include learning
    def update_state(self, action):
        # Update the state based on the transition probabilities and the chosen action
        action_index = self.verbs.index(action)
        # Normalize the transition probabilities for the current state-action pair
        transition_probs = self.transition_matrix[self.current_state, :, action_index]
        transition_probs /= transition_probs.sum()

        new_state = np.random.choice(len(self.sentences), p=self.transition_matrix[self.current_state, :, action_index])

        # Update the transition probabilities based on the observed transition
        self.transition_matrix[self.current_state, new_state, action_index] += self.learning_rate
        self.transition_matrix[self.current_state, :, action_index] /= self.transition_matrix[self.current_state, :, action_index].sum()

        self.current_state = new_state

    # Method for updating the policy based on the current beliefs about the state transition probabilities
    def update_policy(self):
        # Update the policy based on the current beliefs about the state transition probabilities
        self.policy_matrix = np.argmax(self.transition_matrix.sum(axis=1), axis=1)

# An extension of the learning agent class which includes expansion
class ExpandingPOMDPAgent(LearningPOMDPAgent):
    # Initialize the expanding agent
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_verbs = ["jump", "observe", "run", "hide"]  # Potential new verbs/actions
        self.env_sentences = ["The cat is playing.", "The cat is climbing the tree.", "The cat is chasing a mouse."]  # Potential new states

    # Method for the agent to learn from the environment
    def learn_from_environment(self):
        if random.uniform(0, 1) < 1/3:  # 1 in 3 chance of learning from the environment
            # Add a new verb to the list of verbs and extend the transition and policy matrices accordingly
            new_verb = random.choice(self.env_verbs)
            self.verbs.append(new_verb)
            self.transition_matrix = np.dstack([self.transition_matrix, np.full((self.transition_matrix.shape[0], self.transition_matrix.shape[1]), 1/self.transition_matrix.shape[0])])
            self.policy_matrix = np.hstack([self.policy_matrix, np.full((self.policy_matrix.shape[0], 1), 1/self.policy_matrix.shape[1])])

            # Add a new sentence to the list of sentences and extend the transition, observation and preference matrices accordingly
            new_sentence = random.choice(self.env_sentences)
            self.sentences.append(new_sentence)
            self.transition_matrix = np.concatenate([self.transition_matrix, np.full((1, self.transition_matrix.shape[1], self.transition_matrix.shape[2]), 1/self.transition_matrix.shape[1])])
            self.transition_matrix = np.concatenate([self.transition_matrix, np.full((self.transition_matrix.shape[0], 1, self.transition_matrix.shape[2]), 1/self.transition_matrix.shape[0])], axis=1)
            self.observation_matrix = np.vstack([self.observation_matrix, np.full((1, self.observation_matrix.shape[1]), 1/self.observation_matrix.shape[1])])
            self.preference_matrix = np.append(self.preference_matrix, 0)  # Neutral preference for the new state
            self.policy_matrix = np.vstack([self.policy_matrix, np.full((1, self.policy_matrix.shape[1]), 1/self.policy_matrix.shape[1])])

    # Overwrite the simulate method to include learning from the environment
    def simulate(self, rounds=5):
        for i in range(rounds):
            action = self.take_action()
            print(f"Round {i+1}:")
            print(f"Starting state: {self.sentences[self.current_state]}")
            print(f"Action: {action}")
            self.update_state(action)
            print(f"New state: {self.sentences[self.current_state]}")
            print("------------------")
            self.learn_from_environment()  # Learn from the environment after each round

# Instantiate an ExpandingPOMDPAgent and simulate it
n_states = 4
n_actions = 3
A = np.full((n_states, n_states, n_actions), 1/n_states)
B = np.full((n_states, n_states), 1/n_states)
C = np.array([0.2, 0.3, 0.2, 0.3])
D = np.zeros((n_states, n_actions))
D[0, 0] = 1  # approach when cat is on the mat
D[1, 1] = 1  # retreat when cat is under the table
D[2, 0] = 1  # approach when cat is in the garden
D[3, 2] = 1  # wait when cat is on the roof
agent = ExpandingPOMDPAgent(["approach", "retreat", "wait"], ["The cat is on the mat.", "The cat is under the table.", "The cat is in the garden.", "The cat is on the roof."], A, B, C, D, learning_rate=0.1)
agent.simulate(10)

