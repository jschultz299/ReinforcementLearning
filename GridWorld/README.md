# GridWorld

The main file is run.m.

The goal of the project is to have the agent explore the GridWorld environment and to learn how best to reach the green space. The agent is rewarded when it reaches the green space, and penalized when it reaches the red space as well as penalized slightly for each action it performs.

The agent has the option of four actions:
- Move left
- Move right
- Move up
- Move down

## Section a) Demonstrate One Episode with Random Actions

The first section of the code renders the GridWorld environment and demonstrate one episode with random actions.

In this first example, the agent randomly selects an action until it reaches a goal space. The states of the environment spaces will be updated as the agent progresses.

<br>

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/GridWorld/Images/Demo%20One%20Episode.gif" width=40%>

<br>

## Section b) Value Iteration

The next section of the code runs many iterations of the agent performing actions according to a basic policy, updating the environment space Q-values as it progresses.

Below you can see the comparison between the first 100 and 1000 iterations.

<br>

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/GridWorld/Images/Random%20Value%20Iteration.png" width=70%>

<br>

## Section c) Epsilon Greedy Q-Learning

The final section of the code runs many iterations of the agent performing actions according to the current policy, while implementing Epsilong Greedy Q-learning. The value of epsilon can be manipulated to vary the agent's incentive to explore the space vs exploit the policy.

Below you can see the comparison between the first 1000 and 5000 iterations with epsilon = 0.2.

<br>

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/GridWorld/Images/Q-Learning.png" width = 70%>

<br>

By comparing the results of section b and section c, we notice that by training with Epsilon Greedy Q-learning, the agent has learned to avoid the bottom sections of the environment.
