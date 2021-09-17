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
import logging
import offworld_gym
from offworld_gym.envs.common.channels import Channels
# additional imports
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from gym.wrappers import Monitor
from datetime import datetime
from pathlib import Path #check is_file 

logging.basicConfig(level=logging.DEBUG)

# CREATE THE ENVIRONMENT
# env = gym.make("OffWorldDockerMonolithDiscreteSim-v0",
#                channel_type=Channels.RGB_ONLY)
# env.seed(42)

# logging.info(
#     f"action space: {env.action_space} observation_space: {env.observation_space}")
# while True:
#     env.reset()
#     done = False
#     while not done:
#         sampled_action = env.action_space.sample()
#         # env.render()
#         obs, rew, done, info = env.step(sampled_action)

## SET UP DISPLAY ##########
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

###DEEP Q NETWORK    ############


class DQN(nn.Module):
    # def __init__(self, img_height, img_width, action_num):
    def __init__(self, img_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height *
                             img_width*3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=4)  # 4 actions

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


### EXPERIENCE CLASS ####
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

e = Experience({}, [], "a", "v")
e

### REPLAY MEMORY ######


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

## EPSILON GREEDY STRATEGY ###################


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

# REINFORMCEMENT LEARNING AGENT


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                # exploit
                return policy_net(state).argmax(dim=1).to(self.device)

## ENVIRONMENT MANAGER ##################


class OffWorldEnvManager():
    def __init__(self, device):
        self.device = device
        #SIM WORLD - comment out if running real world
        self.env = gym.make(
            "OffWorldDockerMonolithDiscreteSim-v0", channel_type=Channels.RGB_ONLY)
        #REAL WORLD - comment out if running sim world
        # self.env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='Random agent demo',
        #        resume_experiment=False, channel_type=Channels.RGBD,
        #        learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN)
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    #def render(self, mode='human'):  # how you render the screen of the camera 'array'
    def render(self, mode='array'):
        return self.env.render(mode)

    # returns action space itself ie array of available actions eg if 3 actions, it is an array of 3
    def num_actions_available(self):
        return self.env.action_space.n  # number of 4 actions available
        # Check what action space it is initialised with

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('array').transpose(
            (2, 0, 1))  # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # Strip off top and bottom
        top = int(screen_height * 0.001)
        bottom = int(screen_height * 0.999)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage(), T.Resize((160, 120)), T.Grayscale(
                num_output_channels=3), T.ToTensor()
        ])

        # add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(self.device)


#Image processing and log methods
def imsave(imagearray):
    im = Image.fromarray(imagearray)
    im.save(figure_file_name('file_'))

def figure_file_name(file_prefix): #save figures
    # return './figures/' + file_prefix + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'
    return '/root/robot/offworld-gym/figures/' + file_prefix + '.png'

def log(*args): #log and save everything rather than print or render to screen. Print replaced with log.
    with open('pythonbreakout.log', 'a') as f:
        print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "), *args, file=f)

# EXAMPLE OF NON-PROCESSED SCREEN
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# em = OffWorldEnvManager(device)
# em.reset()
# screen = em.render(mode='array')

# plt.figure()
# plt.imshow(screen)
# #plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
# plt.title('Non-processed screen example')
# # plt.show()
# plt.savefig(figure_file_name('fig1'))

# logging.info("Line 268")

# # EXAMPLE OF PROCESSED SCREEN
# screen = em.get_processed_screen()

# logging.info("First post!")

# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
# plt.title('Processed screen example')
# # plt.show()
# plt.savefig(figure_file_name('fig2'))

# ## EXAMPLE OF STARTING STATE ###
# screen = em.get_state()

# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
# plt.title('Starting state example')
# # plt.show()
# plt.savefig(figure_file_name('fig3'))

# # EXAMPLE OF NON-STARTING STATE
# for i in range(5):
#     em.take_action(torch.tensor([1]))
# screen = em.get_state()

# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
# plt.title('Non-starting state example')
# # plt.show()
# plt.savefig(figure_file_name('fig4'))

# ##EXAMPLE OF END STATE ###
# em.done = True
# screen = em.get_state()

# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
# plt.title('Ending state example')
# # plt.show()
# plt.savefig(figure_file_name('fig5'))
# em.close()

##UTILITY FUNCTIONS ###
# Plotting


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    plt.savefig(figure_file_name('fig6'))
    print("Episode", len(values), "\n",
          moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython:
        display.clear_output(wait=True)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

plot(np.random.rand(300), 100)

### TENSOR PROCESSING ##############

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

e1 = Experience(1,1,1,1)
e2 = Experience(2,2,2,2)
e3 = Experience(3,3,3,3)

experiences = [e1,e2,e3]
experiences

batch = Experience(*zip(*experiences))
batch

##Q-Value Calculator #######


class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(
            non_final_states).max(dim=1)[0].detach()
        return values


##  MAIN PROGRAM     #################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device.type, device)

batch_size = 32  # 512 is an ideal batch size. 32 is best to start off with.
gamma = 0.999
eps_start = 0.9  # 1
eps_end = 0.01  # 0.01
eps_decay = 0.001   
target_update = 10   #saves model
memory_size = 100000  
lr = 0.001  
num_episodes = 400  # run for more episodes for better results

em = OffWorldEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

#if file exists, load model otherwise continue to policy_net 
#model and policy net = same thing. Policy network is the main model.
# model = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
# model.load_state_dict(torch.load("offworld.pt")) #should only run if file exists
# model.eval()

model_file = Path("offworld.pt")
if model_file.is_file():
    policy_net = DQN(em.get_screen_height(), em.get_screen_width())
    policy_net.load_state_dict(torch.load("offworld.pt")) #should only run if file exists
    policy_net.eval()
else: 
    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

# policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)



episode_durations = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            # print(experiences)
            # print("Exiting on lin")
            # exit()
            states, actions, rewards, next_states = extract_tensors(
                experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), "offworld.pt")
em.close()

# GET MOVING AVERAGE
assert get_moving_average(100, episode_durations)[-1] > 15

