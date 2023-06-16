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

