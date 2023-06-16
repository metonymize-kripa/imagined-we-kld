"""
This Python script implements a simple simulation of a multi-armed bandit game. The bandit has three arms, named A, B, and C. 
Each arm, when pulled, provides a random reward between 0 and 1. The player is given five tries to pull an arm and collect the reward. 
The game continues, asking the player for an arm to pull until all tries are exhausted.

Key components of the script:

1. 'num_arms': Specifies the number of arms on the bandit.
2. 'arm_names': List of names for the arms.
3. 'rewards': Randomly generated list of rewards for each arm.
4. 'max_tries': The number of times the player can pull an arm.
5. 'play_game()': Function that conducts one round of the game. It prompts the player to choose an arm, 
checks if the input is valid, 
then reveals the reward for the chosen arm.
6. 'main()': The main driver function. It starts the game, manages the score across multiple rounds, 
and calculates and prints the average score at the end of the game. It also reveals the hidden rewards of each bandit arm at the end of the game.

Please note that the reward for each arm remains constant for the entire game, but is unknown to the player.
"""

import random

# Define the number of arms
num_arms = 3
arm_names = ["A", "B", "C"]

# Define the rewards for each arm
rewards = [random.uniform(0, 1) for _ in range(num_arms)]

# Define the number of tries
max_tries = 5

# Define the function to play the game
def play_game():
    print("Choose an arm to pull!")
    selected_arm = input("Enter the letter of the arm you want to pull (A, B, or C): ").upper()
    while selected_arm not in arm_names:
        print("Invalid input. Please enter A, B, or C.")
        selected_arm = input("Enter the letter of the arm you want to pull (A, B, or C): ").upper()
    selected_index = arm_names.index(selected_arm)
    print("You selected arm", selected_arm)
    reward = rewards[selected_index]
    print("You earned a reward of", "{:.1f}".format(reward), "points!\n")
    return reward

# Define the main function
def main():
    print("Multi-Armed Bandit Game")
    print("Welcome to the Multi-Armed Bandit Game!")
    total_score = 0
    print("You have", max_tries, "tries to earn the maximum reward.")
    for i in range(max_tries):
        print("Try", i+1, "of", max_tries)
        reward = play_game()
        total_score += reward
        #print("Your total score is", "{:.1f}".format(total_score))
    print("Game Over!")
    print("Your average score is", "{:.1f}".format(total_score/max_tries))
    printed_rewards = [f"{arm_names[i]}: {reward:.1f}" for i, reward in enumerate(rewards)]
    print("Hidden Bandit:", printed_rewards)



# Run the main function
if __name__ == '__main__':
    main()

