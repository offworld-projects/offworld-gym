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

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.real.real_env import AlgorithmMode, LearningType

# create the envronment and establish connection
env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='Demo of a minimal example 01',
               resume_experiment=False, channel_type=Channels.RGBD,
               learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN)
env.metadata = {'render.modes': []}

# send a command to the robot
state, reward, done, _ = env.step(FourDiscreteMotionActions.FORWARD)

# parse the telemetry
print("Step reward:", reward)
print("Episode has ended:", done)

# plot the state
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
fig, (ax1, ax2) = plt.subplots(1, 2);
ax1.imshow(np.array(state[0, :, :, :3], dtype='int'));
ax2.imshow(np.array(state[0, :, :, 3]), cmap='gray');
plt.show();
