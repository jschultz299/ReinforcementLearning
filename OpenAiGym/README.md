# OpenAiGym

There are 3 main projects within this folder, all of which leverage OpenAi [Gym](https://gym.openai.com) Environments.

## 1) Pendulum

This project uses the [CartPole](https://gym.openai.com/envs/CartPole-v1/) environment. The goal of the project is the train the agent to learn how to balance the pendulum upright for an extended period of time. The agent is rewarded if the pendulum remains upright within a certain range of joint angles.

To run the program:

```bash
python Pendulum.py
```

Below you can see a demonstration of the agent performing random actions for 5 episodes.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/Pendulum/Demo%20Environment.gif" width=50%>

<br>

Training a PPO model with 20,000 timesteps

```bash
model.learn(total_timesteps=20000, callback=eval_callback)
```

is sufficient for the agent to learn how to balance the pendulum, shown below.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/Pendulum/Evaluate%20Model.gif" width = 50%>

<br>

## 2) Breakout

This project uses the [Breakout](https://gym.openai.com/envs/Breakout-v0/) environment. The goal of the project is to play the Atari game Breakout. The agent is rewarded for each brick it breaks, and penalized for losing lives.

To run the program:

```bash
python Breakout.py
```
Below you can see a demonstration of the agent playing Breakout with random actions.



## Acknowledgments
I want to thank Nicholas Renotte for his [Reinforcement Learning in 3 Hours](https://www.youtube.com/watch?v=Mut_u40Sqz4) course on YouTube. You can check out his code for these projects as well [here](https://github.com/nicknochnack/ReinforcementLearningCourse).
