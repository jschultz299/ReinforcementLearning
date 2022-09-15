### Import Dependencies ###

# Import the game
import gym_super_mario_bros
# Import the Joypad Wrapper
from nes_py.wrappers import JoypadSpace
# Import the Simplified controls (only 7 commands)
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

### Test Environment ###
print('\n*** Lets test the model with random inputs. ***')
input('Press ENTER to continue...\n')

# Setup the game
environment_name = 'SuperMarioBros-v0'
env = gym_super_mario_bros.make(environment_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Run the game with random actions
done = True
frames = 3000
for step in range(frames):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close()

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
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# Apply GrayScaling
env = GrayScaleObservation(env, keep_dim=True)
# Stack frames
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# Visualize the stacked frames
#state = env.reset()
#for i in range(10):
#    state, reward, done, info = env.step([5])
#
#plt.figure(figsize=(10,8))
#for idx in range(state.shape[3]):
#    plt.subplot(1,4,idx+1)
#    plt.imshow(state[0][:,:,idx])
#plt.show()

### Train the Model ###
print('\n*** Lets train the model. ***')
input('Press ENTER to continue...\n')

# Import os for file path management
import os
# Import PPO for algorithms
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

# Callback for saving the model
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
        
CHECKPOINT_DIR = './train'
LOG_DIR = '.logs'

# Be careful with check_freq. Set higher to save space and save less frequently.
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Initialize Model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=.000001, n_steps=512)

# Train the model
# Note: Takes a long time on CPU. Will need to use GPU.

#model.learn(total_timesteps=1000, callback=callback)

# model.learn(total_timesteps=1000)
# model.save('final_model')

### Test the Model ###

# Load pre-trained model
#model = PPO.load('./train/best_model_10000')
#
#state = env.reset()
#while True:
#    action, _ = model.predict(state)
#    state, reward, done, info = env.step(action)
#    env.render()
