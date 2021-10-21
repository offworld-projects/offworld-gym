import os
import copy
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from pynput.keyboard import Key, Listener

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.real.real_env import AlgorithmMode, LearningType

# to surpress the warning when running in real env
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 

def on_press(key):
    if len(keys) > 0:
        keys.pop(0)
    if key not in key_mapping:
        print("Please press arrow keys to control or esc key to exit.")
    else:
        keys.append(key_mapping[key])  


def on_release(key):
        return False

if __name__ == "__main__":

    print(
    """
    This example allows you to manually control the OffWorld Gym robot.
    Use the arrow keys ← ↑ → ↓ to issue the commands and [Esc] to exit
    You can monitor the robot via the overhead cameras at https://gym.offworld.ai/cameras
    """)

    key_mapping = { Key.up: 2, Key.down: 3, Key.left: 0, Key.right: 1, Key.esc: -1}
    
    keys = [] # current key buffer
    
    # create the envronment and establish connection
    env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='keyboard_manual_control',
               resume_experiment=True, channel_type=Channels.DEPTH_ONLY,
               learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN)

    state = env.reset()
    while True:
        done = False
        step_count = 0
        while not done:
            # press key remotely
            input_finished = False # wait for input
            while not input_finished:
                with Listener(on_press = on_press,
                            on_release=on_release) as listener:
                            listener.join()
                if len(keys) > 0:
                    input_finished = True
                    if keys[0] == -1: break # early stop 

            action = keys[0]
            state, reward, done, _ = env.step(action)

        state = env.reset()

        
