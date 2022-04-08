### Import Dependencies ###

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

### Test Environment ###
print('\n*** Lets test the model with random inputs. ***')
input('Press ENTER to continue...\n')

environment_name = 'Breakout-v0'
env = gym.make(environment_name)
#env = gym.make(environment_name, render_mode='human')

episodes = 5
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
env = make_atari_env(environment_name, n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

log_path = os.path.join('Training', 'Logs')
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

#model.learn(total_timesteps=100000)
#
#a2c_path = os.path.join('Training', 'Saved Models, 'A2C_Breakout_Model')
#model.save(a2c_path)
#
#del model
#model = A2C.load(a2c_path, env)

# Or instead load pre-trained model
del model
#a2c_path = os.path.join('Training', 'Saved Models', 'A2C_300k_Breakout_Model')
a2c_path = os.path.join('Training', 'Saved Models', 'A2C_2M_Breakout_Model')
model = A2C.load(a2c_path, env)

### Evaluate the Model ###
print('\n*** Lets evaluate the model. ***')
input('Press ENTER to continue...\n')
env = make_atari_env(environment_name, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

out = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print('Average Reward: (' + str(out[0]) + ' +/- ' + str(out[1]) + ')')

print('\n*** Lets view the evaluation metrics in tensorboard. ***')
print('--- Switch to other terminal to view metrics ---')
input('Press ENTER to continue...\n')

env.close()

### Test the Model ###

# Doesn't really work for some reason...
#episodes = 5
#for episode in range(1, episodes+1):
#    obs = env.reset()
#    done = False
#    score = 0
#
#    while not done:
#        env.render()
#        action, _ = model.predict(obs) # Now using model here!
#        obs, reward, done, info = env.step(action)
#        score+=reward
#    print('Episode:{} Score:{}'.format(episode, score))
#env.close()
