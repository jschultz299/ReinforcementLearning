### Import Dependencies ###
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import time

### Setup the Environment ###
environment_name = "CartPole-v0"
env = gym.make(environment_name)

print('\n*** Lets test the model with some random values. ***')
input('Press ENTER to Continue...\n')

episodes = 5
for episode in range(1, episodes+1):
  state = env.reset()
  done = False
  score = 0

  while not done:
    env.render()
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    score+=reward
  print('Episode:{} Score:{}'.format(episode, score))
env.close()

# 0-push cart to left, 1-push cart to the right
env.action_space.sample()

# [cart position, cart velocity, pole angle, pole angular velocity]
env.observation_space.sample()

### Train the Model ###
#print('Now lets train the model')

## Make directories first!
log_path = os.path.join('Training', 'Logs')

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

### Adding a Callback to Training Stage ###
print('\n*** Now Lets train the model. ***\n')
input('Press ENTER to Continue...\n')

save_path = os.path.join('Training', 'Saved Models')

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            best_model_save_path=save_path,
                            verbose=1)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)

### Evaluation ###
print('\n*** Lets evaluate the model. ***')
input('Press ENTER to Continue...\n')

evaluate_policy(model, env, n_eval_episodes=5, render=True)
env.close()

### Viewing Logs in Tensorboard ###
print('\n*** Lets view the training logs in tensorboard. ***')
print('--- Switch to new terminal to view logs ---')
input('Press ENTER to Continue...\n')
# Start in separate terminal!
#training_log_path = os.path.join(log_path, 'PPO_2')
#tensorboard --logdir={traning_log_path}
# Then type localhost:6006 in browser window to view tensorboard output

# Test with new model
print('\n*** Lets test the model. ***')
input('Press ENTER to Continue...\n')

del model
model = PPO.load(os.path.join('Training', 'Saved Models', 'Pendulum_Model'), env=env)

episodes = 5
for episode in range(1, episodes+1):
  obs = env.reset()
  done = False
  score = 0

  while not done:
    env.render()
    action, _ = model.predict(obs) # Now using model here!
    obs, reward, done, info = env.step(action)
    score+=reward # Cutoff is 200
  print('Episode:{} Score:{}'.format(episode, score))
env.close()
