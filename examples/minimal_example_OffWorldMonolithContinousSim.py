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
import random
import numpy as np

# create the envronment
env = gym.make('OffWorldMonolithContinousSim-v0', channel_type=Channels.DEPTH_ONLY)
env.seed(123)

input("Press Enter to continue...\n")

for i in range(50):
    action = np.array([random.uniform(-0.7, 0.7), random.uniform(-2.4, 2.4)])
    obs = env.step(action)
    print(obs)