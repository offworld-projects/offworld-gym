from sshkeyboard import listen_keyboard

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.real.real_env import AlgorithmMode, LearningType

# to surpress the warning when running in real env
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 


print(
    """
    This example allows you to manually control the OffWorld Gym robot.
    Use the arrow keys ← ↑ → ↓ to issue the commands and [Esc] to exit
    You can monitor the robot via the overhead cameras at https://gym.offworld.ai/cameras
    """)

key_actions = {'up': 2, 'down': 3, 'left': 0, 'right': 1}
key_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

# create the envronment and establish connection
env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='Manual control',
               resume_experiment=False, channel_type=Channels.DEPTH_ONLY,
               learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN)

def press(key):
    if key not in key_actions.keys():
        print("Unknown action, please use arrows to navigate or Esc to exit")

    else:
        print("                    \r", key_symbols[key])
        state, reward, done, _ = env.step(key_actions[key])
        if done:
            state = env.reset()
    
    print ("  ← ↑ → ↓ 'esc'", end="\r")

print ("  ← ↑ → ↓ 'esc'", end="\r")
listen_keyboard(on_press=press, sequential=True)
