"""
Let's consider a scenario where three agents - 
let's call them A, B, and C - are working together to move a large object. 
The goal is to coordinate their actions to move the object from one location to another.

In the context of active inference, each agent would have a model of the world that includes 
their own state, the state of the other agents, and the state of the environment 
(including the object to be moved). They would use this model to predict the 
actions required to achieve the goal and anticipate the actions of the other agents.

Agent A: This agent might predict that they need to push the object from one side. 
They would also predict that Agents B and C will push from the other sides. 
If Agent B or C does something unexpected (like stopping or pushing in a different direction), 
Agent A would update their model and adjust their actions to minimize the prediction error.

Agent B: Similarly, this agent would predict their own actions and those of Agents A and C. 
They would adjust their actions based on the actions of the other agents to maintain 
coordination and achieve the goal.

Agent C: This agent would also have a model of the world and predict the actions of Agents A 
and B. They would adjust their actions based on the actions of the other agents.

In this scenario, the "Imagined We" model would be the shared goal of moving the object and 
the coordinated actions required to achieve this goal. Each agent would be continuously 
updating their model of the world and predicting the actions of the other agents to 
minimize surprise and maintain coordination.

This is a simplified example, but it illustrates how active inference and the "Imagined We" 
model can be applied to a multi-agent scenario. The agents are continuously predicting, 
acting, and learning to achieve a shared goal.

SIMPLE SCENARIO:
We will now implement this scenario in Python. We will start with a simple scenario where
the agents are trying to move an object from one location to another. The agents will
have a model of the world that includes their own state, the state of the object, and
the goal location. They will use this model to predict the actions required to achieve
the goal. They do not know what the other agents are doing, so they will not be able to
anticipate their actions. They will simply act based on their own intention and observation
and adjust their actions based on the outcome. Let's make this concrete with agents that have 
two actions, push/pull. The effect of these actions are dependent on their location wrt the 
object. The closer they are the stronger the force. The agents will have an intention to move
the object in a particular direction. The strength of the intention will be proportional to
the distance from the object. The agents will act based on their intention and the outcome
will be determined by the strength of the intention and the distance from the object. If the
action is successful, the agent will increase the strength of the intention. If not, they
will decrease the strength. 
The agents will continue to act and update their intention until the object reaches the goal
location. The intention will be a vector with a direction and a strength. 
The strength will be proportional to the distance from the object. 
The direction will be either towards the object or away from it.
The action will be push/pull based on the direction of the intention.
The outcome will be determined by the strength of the intention and the distance from the
object. If the action is successful, the agent will increase the strength of the intention.
If not, they will decrease the strength and change the direction of the intention.
The agents will continue to act and update their intention until the object reaches the goal location. 

OBSERVING OTHER AGENTS:
In the previous scenario, we assumed that agents did not have a model of the other agents.
In this scenario, we will assume that each agent can observe the other agents. This
allows them to modify their intention and action towards the ultimate goal. The agents will
use learning to adjust their own actions in response to the actions of the other agents.

MODELING OTHER AGENT INTENTIONS:
In the previous example, we assumed that the agents did not have a model of what other agents
were intending. In this scenario, we will assume that the agents can infer the intentions of
the other agents based on their actions and the location of the object. This allows them to
converge towards a shared intention of moving the object as one joint agent with a shared goal.

IMAGINED WE MODEL:
In this scenario, one of the agents is randomly selected to be the "leader". It is the one that
will have the goal location and will be responsible for coordinating the actions of the other. 
However, none of the agents know which one is the leader. They will have to infer this based on
the actions of the other agents. This is similar to the "Imagined We" model, where the agents
are continuously predicting the actions of the other agents to infer the shared goal and
coordinate their actions to achieve it.

The setup below allows agents and the simulation to be configured with different parameters
setting up the scenario from one of the four above.
"""

import random

class Agent:
    def __init__(self, name, location):
        self.name = name
        self.location = location
        self.intention = {'strength': 1.0, 'direction': random.choice([-1, 1])}  # Initialize intention

    def action(self, obj_location):
        # The agent's action is determined by the direction of their intention
        # and their relative position to the object
        return self.intention['direction'] if self.location < obj_location else -self.intention['direction']

        return self.intention['direction']

    def update_intention(self, success):
        # If the action was successful, increase the strength and keep the same direction
        # If not, decrease the strength and change the direction
        if success:
            self.intention['strength'] *= 1.1
        else:
            self.intention['strength'] *= 0.9

    def learn(self, obj_location):
        # Change the direction based on the relative location of the object
        self.intention['direction'] = 1 if obj_location > self.location else -1

class Object:
    def __init__(self, location):
        self.location = location

def simulate(agents, obj, goal, steps):
    for step in range(steps):
        print(f"Step {step+1}")
        forces = []
        for agent in agents:
            action = agent.action(obj.location)
            # The force is proportional to the strength of the agent's intention
            # and inversely proportional to the distance
            # The direction of the force is the product of the direction of the intention
            # and whether the agent location is greater or less than the object location
            force = agent.intention['strength'] * action / (abs(agent.location - obj.location) + 0.001)
            forces.append(force)
            print(f"Agent {agent.name} at location {agent.location} exerts force {force}")

        # The object moves in the direction of the net force
        old_location = obj.location
        # The object moves one unit per step, irrespective of the force
        if sum(forces) > 0:
            obj.location += 1 
        elif sum(forces) < 0:
            obj.location -= 1
        print(f"Object moved to location {obj.location}\n")

        # Update each agent's intention based on whether the object moved closer to the goal
        for agent in agents:
            success = abs(goal - old_location) > abs(goal - obj.location)
            agent.update_intention(success)
            agent.learn(obj.location)

        # If the object has reached the goal, end the simulation
        if obj.location == goal:
            print(f"Goal reached in {step+1} steps!")
            break

# Initialize agents and object
agents = [Agent('A', 1), Agent('B', -1), Agent('C', 0)]
#agents = [Agent('A', 1), Agent('B', -1)]
#agents = [Agent('A', 0)]
obj = Object(0)

# Run the simulation for a maximum of 100 steps with a goal at location 5
simulate(agents, obj, 5, 100)
