### Import Dependencies ###

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

### Test the Environment ###
print('\n*** Lets test out the environment. ***')
input('Press ENTER to continue...\n')

environment_name = 'CarRacing-v0'
env = gym.make(environment_name)

episodes = 2
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

### Train the Model ###
print('\n*** Lets train the model. ***')
input('Press ENTER to continue...\n')

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

log_path = os.path.join('Training', 'Logs')
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

#model.learn(total_timesteps=20000)
#
#ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_Model')
#model.save(ppo_path)
#
#del model
#model = PPO.load(ppo_path, env)

# Or instead, load pre-trained model
del model
ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_model')
#ppo_path = os.path.join('Training', 'Saved Models', 'PPO_428k_Driving_model')
#ppo_path = os.path.join('Training', 'Saved Models', 'PPO_2m_Driving_model')
model = PPO.load(ppo_path, env)

### Evaluate the Model ###
print('\n*** Lets evaluate the model. ***')
input('Press ENTER to continue...\n')

out = evaluate_policy(model, env, n_eval_episodes=5, render=True)
print('Average Reward: (' + str(out[0]) + ' +/- ' + str(out[1]) + ')')

env.close()

print('\n*** Lets view the evaluation metrics in tensorboard. ***')
print('--- Switch to other terminal to view metrics ---')
input('Press ENTER to continue...\n')
