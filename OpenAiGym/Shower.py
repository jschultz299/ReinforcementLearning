### Import Dependencies ###
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# Import helpers
import numpy as np
import random
import os

# Import stable baseline dependencies
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

### Types of Spaces ###

# --- Usage ---
# Discrete(3).sample()
# Box(0,1, shape=(3,3))
# Tuple(Discrete(3), Box(0,1, shape=(3,)), MultiBinary(4))).sample()
# Dict({'height':Discrete(2), "speed":Box(0,100, shape=(1,)), "color":MultiBinary(4))}).sample()
# MultiBinary(4).sample()
# MultiDiscrete([5,2,2,5]).sample()

### Building an Environment ###
# - Build an agent to give us the best shower
# - Random temperature
# - Optimal temperature is between 37 and 39 degrees
# - Agent does not know this but will need to learn it

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        #self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.observation_space = Box(low=0, high=100, shape=(1,))
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60
        
    def step(self, action):
        # Apply temp adjustment
        self.state += action-1
        
        # Decrease shower time
        self.shower_length -= 1
        
        # Calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1
            
        if self.shower_length <= 0:
            done = True
        else:
            done = False
            
        info = {}
        
        return self.state, reward, done, info
        
    def render(self):
        # Implement Visualization
        pass
        
    def reset(self):
        # Reset the environment
        self.state = np.array([38+random.randint(-3,3)]).astype(float)
        self.shower_length = 60
        
        return self.state

### Test Environment ###
env = ShowerEnv()

episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

### Train Model ###
log_path = os.path.join('Training', 'Logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

#model.learn(total_timesteps=50000)
#
#ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Shower_model')
#model.save(ppo_path)

# Or instead, load pre-trained model
del model
ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Shower_model')
model = PPO.load(ppo_path, env)

### Evaluate the Model ###
out = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print('Average Reward: (' + str(out[0]) + ' +/- ' + str(out[1]) + ')')

env.close()
