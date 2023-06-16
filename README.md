# imagined-we-kld
Experimenting with ways to measure effects of changes in the mental models of agents, with shared beliefs and intentions, using KLD like measures

# README: Kullback-Leibler Divergence and Multi-Agent Interaction Scripts

This repository contains Python scripts designed to compute the Kullback-Leibler (KL) Divergence between different probability distributions and to simulate multi-agent interactions.

The repository is divided into three major sections: KL Divergence, Multi-Agent Simulations, and Multi-Armed Bandit.

## Section 1: KL Divergence

The scripts in this section are used for computing the KL Divergence between different types of data distributions.

### 1.1. basic_kld.py
This script calculates the KL Divergence between two Gaussian distributions using their means and standard deviations. More details in the script comments.

### 1.2. gen_kld.py
This script demonstrates how to estimate the KL Divergence between two simulated Gaussian data distributions with different standard deviations. The entire process from data simulation to KL divergence calculation is described step by step.

### 1.3. gen_multivariate_kld.py
This script calculates the KL Divergence between two 4-dimensional Gaussian distributions. It includes the generation of data samples and the computation of the KL Divergence. Please be aware of the WARNING regarding the number of samples (n).

### 1.4. kld-pomdp.py
This script simulates Partially Observable Markov Decision Processes (POMDPs) and calculates the KL Divergence between two such processes.

## Section 2: Multi-Agent Simulations

The scripts in this section are used to simulate interactions among multiple agents in different scenarios.

### 2.1. imagined_we_pseudocode.py
This script describes a multi-agent system where each agent updates its belief, goal, and intention variables based on the actions and observations of a random subset of other agents in the environment.

### 2.2. three_agents.py
This script simulates a scenario where three agents are trying to coordinate their actions to move an object from one place to another.

### 2.3. simple_stag_hunt.py
This script simulates a game of Stag Hunt in a one-dimensional environment, with multiple hunters, rabbits, and stags. Each hunter chooses what to hunt based on their proximity to the nearest prey and their last payoff.

## Section 3: Multi-Armed Bandit

The scripts in this section deal with the Multi-Armed Bandit problem, a well-known problem in reinforcement learning.

### 3.1. multi_arm_bandit_simple_game.py
This script simulates a simple three-armed bandit game. Each arm provides a random reward when pulled, and the player has five tries to pull an arm and collect the reward.

### 3.2. simple_muliarm_sim.py
This script simulates a Multi-Armed Bandit problem using the epsilon-greedy algorithm with hints. It tracks and returns the actions taken, rewards obtained, and hints provided during the interaction with the bandit. 

Please refer to the comments within each script for a detailed explanation of the code. The scripts use the numpy and scipy libraries for numerical and statistical operations.

