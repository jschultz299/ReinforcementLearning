# OpenAiGym

There are 4 main projects within this folder, all of which leverage OpenAi [Gym](https://gym.openai.com) Environments.

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
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)
```

is sufficient for the agent to learn how to balance the pendulum, shown below.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/Pendulum/Evaluate%20Model.gif" width = 50%>

Check out the tensorboard logs for the PPO model trained with 20k timesteps [here](https://github.com/jschultz299/ReinforcementLearning/tree/main/OpenAiGym/Images/Pendulum/Tensorboard%20Logs).

## 2) Breakout

This project uses the [Breakout](https://gym.openai.com/envs/Breakout-v0/) environment. The goal of the project is to play the Atari game Breakout. The agent is rewarded for each brick it breaks, and penalized for losing lives.

To run the program:

```bash
python Breakout.py
```
Below you can see a demonstration of the agent playing Breakout with random actions.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/Breakout/Demo%20Environment.gif" width = 50%>

Training an A2C model with 2 million timesteps

```bash
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)
```
results in an average reward of approximately 23 bricks broken per game, shown below.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/Breakout/Evaluate%20Model.gif" width = 40%>

Training for significantly longer might improve performance.

Check out the logs for an A2C model trained with 100k timesteps [here](https://github.com/jschultz299/ReinforcementLearning/tree/main/OpenAiGym/Images/Breakout/Tensorboard_Logs).

## 3) Self Driving

This project uses the [CarRacing](https://gym.openai.com/envs/CarRacing-v0/) environment. The goal of this project is for the agent (the car) to drive along the track for as long as possible. The agent receives rewards for remaining on the track, and is penalized for leaving the track as well as penalzed slightly for each timestep. The actions the agent may take are the direction to turn the wheels as well as acceleration and braking. All of the actions are in the continuous space. The track is considered solved if the agent receives a total score of 900.

To run the program:

```bash
python SelfDriving.py
```
Below you can see a demonstration of the agent driving along the track with random inputs.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/SelfDriving/Demo%20Environment.gif" width = 50%>

Training a PPO model with just 10,000 timesteps

```bash
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10000)
```
results in an agent with poor performance, shown below.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/SelfDriving/Evaluate%2010k%20Model.gif" width = 50%>

This model achieved an average score of -40.6.

Check out the tensorboard logs for the PPO model trained with 10k timesteps [here](https://github.com/jschultz299/ReinforcementLearning/tree/main/OpenAiGym/Images/SelfDriving/Tensorboard_Logs/10k_Model).

Training the model for more timesteps, this time 200,000

```bash
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=200000)
```
results in an agent with much better performance, shown below.

<img src="https://github.com/jschultz299/ReinforcementLearning/blob/main/OpenAiGym/Images/SelfDriving/Evaluate%20200k%20Model.gif" width = 50%>

This model achieved an average score of 436.2, and was even able to fully solve the track on occasion.

Check out the tensorboard logs for the PPO model trained with 10k timesteps [here](https://github.com/jschultz299/ReinforcementLearning/tree/main/OpenAiGym/Images/SelfDriving/Tensorboard_Logs/200k_Model).

Because the actions are continuous, the model has a hard time learning the appropriate actions to take. It is possible that training for longer might result in better performance. [NotAnyMike](https://github.com/NotAnyMike) tried adjusting the action space to only use discrete inputs, which simplified the problem somewhat. Check out his solution [here](https://notanymike.github.io/Solving-CarRacing/).

## Acknowledgments
I want to thank Nicholas Renotte for his [Reinforcement Learning in 3 Hours](https://www.youtube.com/watch?v=Mut_u40Sqz4) course on YouTube. You can check out his code for these projects as well [here](https://github.com/nicknochnack/ReinforcementLearningCourse).

## 4) Super Mario Bros

This project uses the [SuperMarioBros]([https://gym.openai.com/envs/CartPole-v1/](https://pypi.org/project/gym-super-mario-bros/)) environment wrapper. The goal of the project is the train the agent to learn how to play Super Mario Bros. The agent is rewarded if Mario reaches the flag.

To run the program:

```bash
python SuperMarioBros.py
```

COMING SOON...
