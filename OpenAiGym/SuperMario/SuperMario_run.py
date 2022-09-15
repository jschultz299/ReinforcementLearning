### Import Dependencies ###

# Import the game
import gym_super_mario_bros
# Import the Joypad Wrapper
from nes_py.wrappers import JoypadSpace
# Import the Simplified controls (only 7 commands)
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import Monitor

# Setup the game
environment_name = 'SuperMarioBros-v0'
env = gym_super_mario_bros.make(environment_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

### Pre-process ###
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib
from matplotlib import pyplot as plt

# Re-create the base environment
environment_name = 'SuperMarioBros-v0'
env = gym_super_mario_bros.make(environment_name)

env = Monitor(env, './video', force=True)

env = JoypadSpace(env, SIMPLE_MOVEMENT)
# Apply GrayScaling
env = GrayScaleObservation(env, keep_dim=True)
# Stack frames
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 8, channels_order='last')

# Import PPO for algorithms
from stable_baselines3 import PPO

### Test the Model ###
#print('\n*** Lets test the model. ***')
#input('Press ENTER to continue...\n')

# Load pre-trained model
model = PPO.load('./train/new_model_5000000')

import time
state = env.reset()
success = False
count = 0
while not success:
#    time.sleep(.01)
   count+=1
   action, _ = model.predict(state)
   state, reward, done, info = env.step(action)
   success = info[0].get("flag_get")
   env.render()

   if count % 1000 == 0:
      print("Iteration: ", count)

   if count > 100000:
      print("Flag not reached.")
      break

if success:
   print("Flag reached!!")

print("done.")