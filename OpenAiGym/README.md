# OpenAiGym

There are 3 main projects within this folder, all of which leverage OpenAi [Gym](https://gym.openai.com) Environments.

## 1) Pendulum

This project uses the [CartPole](https://gym.openai.com/envs/CartPole-v1/) environment. The goal of the project is the train the agent to learn how to balance the pendulum upright for an extended period of time. The agent is rewarded if the pendulum remains upright within a certain range of joint angles.

To run the program:

```bash
python Pendulum.py
```

Below you can see a demonstration of the agent performing random actions for 5 episodes.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/Pendulum/Demo%20Environment.gif" width=40%>

<br>

## Acknowledgments
I want to thank Nicholas Renotte for his [Reinforcement Learning in 3 Hours](https://www.youtube.com/watch?v=Mut_u40Sqz4) course on YouTube. You can check out his code for these projects as well [here](https://github.com/nicknochnack/ReinforcementLearningCourse)
