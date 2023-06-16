"""
This program simulates a game of Stag Hunt with multiple hunters, rabbits, and stags in a one-dimensional environment. 
Each hunter, rabbit, and stag are initialized at a random location sampled from a Gaussian distribution.

The game consists of several rounds, and in each round, the following steps occur:

1. Each hunter decides what to hunt (either stags or rabbits) based on the proximity of the nearest rabbit or stag. 
   If the payoff from the last round was 0 or negative, the hunter will choose to hunt the nearest prey.

2. Each hunter attempts to hunt according to their chosen strategy. 
   If a hunter chose to hunt a rabbit and the nearest rabbit is within a unit distance, the hunt is successful and the rabbit is "removed" by moving it to a distant location.
   If a hunter chose to hunt a stag and all hunters chose to hunt stags, and the nearest stag is within a unit distance, the hunt has a 10% chance of being successful. 
   If successful, the stag is "removed" by moving it to a distant location. 
   The payoff for hunting a rabbit is 0.5 and for hunting a stag is 1.

3. After the hunting phase, each entity updates its location. 
   The location of rabbits is updated by adding a Gaussian random value multiplied by 0.1. 
   The location of stags is updated by adding a Gaussian random value. 
   The location of hunters who chose to hunt rabbits is updated by moving 0.5 units towards the location of the nearest rabbit. 
   The location of hunters who chose to hunt stags is updated by moving 0.25 units towards the location of the nearest stag. 

4. Finally, the program prints the status of each hunter, including their current strategy, location, and last payoff.

This process repeats for a defined number of rounds. 
Over time, the hunters learn the optimal strategy based on the received payoffs, and adapt their hunting strategy accordingly.
"""

import random
import math

class Hunter:
    def __init__(self, name):
        self.name = name
        self.strategy = random.choice(['Stag', 'Rabbit'])
        self.payoff = 0
        self.last_payoff = 0
        self.location = random.gauss(0, 1)
        self.target_location = None

    def decide(self, stags, rabbits):
        if self.last_payoff <= 0:
            nearest_stag = min(stags, key=lambda s: math.fabs(self.location - s.location))
            nearest_rabbit = min(rabbits, key=lambda r: math.fabs(self.location - r.location))

            if math.fabs(self.location - nearest_stag.location) < math.fabs(self.location - nearest_rabbit.location):
                self.strategy = 'Stag'
            else:
                self.strategy = 'Rabbit'

        return self.strategy

    def hunt(self, hunters, stags, rabbits):
        if self.strategy == 'Rabbit':
            nearest_rabbit = min(rabbits, key=lambda r: math.fabs(self.location - r.location))
            self.target_location = nearest_rabbit.location
            if math.fabs(self.location - nearest_rabbit.location) < 1:
                self.payoff = 0.5
            else:
                self.payoff = 0
        elif self.strategy == 'Stag':
            nearest_stag = min(stags, key=lambda s: math.fabs(self.location - s.location))
            self.target_location = nearest_stag.location
            if all(h.strategy == 'Stag' for h in hunters) and math.fabs(self.location - nearest_stag.location) < 1:
                self.payoff = 1 if random.random() < 0.1 else 0
            else:
                self.payoff = 0

    def update_location(self):
        if self.target_location is not None:
            # Update location by moving towards the target
            self.location += (self.target_location - self.location) * (0.5 if self.strategy == 'Rabbit' else 0.25)

    def update(self):
        self.last_payoff = self.payoff
        self.payoff = 0

    def __str__(self):
        return f'{self.name}, located at {self.location:0.1f} hunted a {self.strategy} located at {self.target_location},  and received a payoff of {self.payoff}'

class Rabbit:
    def __init__(self):
        self.location = random.gauss(0, 1)

    def update_location(self):
        self.location += random.gauss(0, 1) * 0.1


class Stag:
    def __init__(self):
        self.location = random.gauss(0, 1)

    def update_location(self):
        self.location += random.gauss(0, 1) * 1


hunters = [Hunter('Hunter1'), Hunter('Hunter2'), Hunter('Hunter3')]
rabbits = [Rabbit() for _ in range(3)]
stags = [Stag() for _ in range(1)]

num_rounds = 5
for round in range(num_rounds):
    for hunter in hunters:
        hunter.decide(stags, rabbits)

    for hunter in hunters:
        hunter.hunt(hunters, stags, rabbits)

    print(f'Round {round + 1}')
    for hunter in hunters:
        print(hunter)

    for hunter in hunters:
        hunter.update()
    for rabbit in rabbits:
        rabbit.update_location()
    for stag in stags:
        stag.update_location()
    for hunter in hunters:
        hunter.update_location()
    print('\n')
