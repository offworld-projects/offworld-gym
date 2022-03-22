#!/usr/bin/env python

# Copyright 2019 OffWorld Inc.
# Doing business as Off-World AI, Inc. in California.
# All rights reserved.
#
# Licensed under GNU General Public License v3.0 (the "License")
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law, any source code or other materials
# distributed under the License is distributed on an "AS IS" basis,
# without warranties or conditions of any kind, express or implied.

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.real.real_env import AlgorithmMode, LearningType

# create the envronment and establish connection
env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='Random agent',
               resume_experiment=True, channel_type=Channels.RGBD,
               learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN)

# initialize figure for drawing RGB and D inputs
# fig, (ax1, ax2) = plt.subplots(1, 2);
# plt.ion();
# plt.show();

# env.reset()

# send a command to the robot
while True:
    done = False
    while not done:
        state, reward, done, _ = env.step(env.action_space.sample())

        # # display the state
        # ax1.imshow(np.array(state[0, :, :, :3], dtype='int'));
        # ax2.imshow(np.array(state[0, :, :, 3]), cmap='gray');
        # plt.draw();
        # plt.pause(0.001);

        # print out action outcome
        print("Step reward:", reward)
        print("Episode has ended:", done)

    env.reset()
