# GridWorld

The main file is run.m.

## Section a) Demonstrate One Episode with Random Actions

The first section of the code will render the GridWorld environment and demonstrate one episode with random actions.

The agent has the option of four actions:
- Move left
- Move right
- Move up
- Move down

In this first example, the agent randomly selects an action until it reaches a goal space. The states of the environment spaces will be updated as the agent progresses.

<br>

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/GridWorld/Images/Demo%20One%20Episode.gif" width=60%>

<br>

## Section b) 

The next section of the code runs many iterations of the agent performing actions according to a basic policy, updating the environment space Q-values as it progresses.

Below you can see the comparison between the first 100 and 1000 iterations.

<br>

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/GridWorld/Images/Random%20Value%20Iteration.png" width=60%>
