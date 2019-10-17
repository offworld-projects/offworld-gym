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

from offworld_gym import version

__version__     = version.__version__

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys, pdb, time, glob
import numpy as np
from datetime import datetime
import pickle

# configure tensorflow and keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels

import keras
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent, HumanDQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from rl.callbacks import ModelIntervalCheckpoint, TerminateTrainingOnFileExists, SaveDQNTrainingState

from utils import TB_RL, GetLogPath


# define paths
NAME              = 'real_offworld_monolith-{}'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
OFFWORLD_GYM_ROOT = os.environ['OFFWORLD_GYM_ROOT']
LOG_PATH          = '%s/logs/real' % OFFWORLD_GYM_ROOT
MODEL_PATH        = '%s/models/real' % OFFWORLD_GYM_ROOT
STATE_PATH        = './run/real_agent_state'

if not os.path.exists(LOG_PATH): os.makedirs(LOG_PATH)
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)


# create the envronment
env = gym.make('OffWorldMonolithRealEnv-v0', experiment_name='dqn_depth_first_experiment', resume_experiment=True, channel_type=Channels.DEPTH_ONLY)
nb_actions = env.action_space.n


# define network architecture
def convBNRELU(input1, filters=8, kernel_size = 5, strides = 1, id1=0, use_batch_norm=False, use_leaky_relu=False, leaky_epsilon=0.1):
    cname = 'conv%dx%d_%d' % (kernel_size, kernel_size, id1)
    bname = 'batch%d' % (id1 + 1) # hard-coded + 1 because the first layer takes batch0
    elu_name = 'elu_%d' % (id1 + 1)
    leaky_relu_name = 'leaky_relu_%d' % (id1 + 1)
    out = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, name = cname)(input1)
    
    if use_batch_norm == True:
        out = BatchNormalization(name=bname)(out)
    
    if use_leaky_relu:
        out = LeakyReLU(leaky_epsilon, name=leaky_relu_name)(out)
    else:
        out = Activation('relu')(out)
    
    return out

def create_network():
    input_image_size = env.observation_space.shape[1:]
    img_input = Input(shape=input_image_size, name='img_input')
        
    x = img_input
    for i in range(2):
        x = convBNRELU(x, filters=4, kernel_size=5, strides=2, id1=i, use_batch_norm=False, use_leaky_relu=True)
        x = MaxPooling2D((2, 2))(x)
    x = convBNRELU(x, filters=1, kernel_size=5, strides=1, id1=9, use_batch_norm=False, use_leaky_relu=True)
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
        
    output = Dense(nb_actions)(x)
    model = Model(inputs=[img_input], outputs=output)
    print(model.summary())
        
    return model

class RosbotProcessor(Processor):
    
    def process_observation(self, observation):
        '''
        observations are [image(240, 320, 3), config(4,)]
        '''

        return observation

    def process_state_batch(self, batch):
        imgs_batch = []
        for exp in batch:
            imgs = []
            configs = []
            for state in exp:
                imgs.append(np.expand_dims(state[0], 0))
                configs.append(np.expand_dims(100, 0))
            imgs_batch.append(np.concatenate(imgs, -1))
        imgs_batch = np.concatenate(imgs_batch, 0)

        return imgs_batch


def train(hitl=False):
    
    # waiting for the ROS messages to clear
    time.sleep(5)

    # check whether to resume training
    print("\n====================================================")

    if os.path.exists("%s" % STATE_PATH):
        print("State from the previous run detected. Do you wish to resume learning from the latest available snapshot? (y/n)")
        while True:
            choice = input().lower()
            if choice == 'y':
                last_episode = np.max([int(os.path.basename(s).replace("episode-", "")) for s in glob.glob("%s/episode-*" % STATE_PATH)])
                LAST_STATE_PATH = "%s/episode-%d" % (STATE_PATH, last_episode)
                print("Resuming training from %s" % LAST_STATE_PATH) 
                resume_training = True
                break
            elif choice == 'n':
                print("Please remove or move %s and restart this script." % STATE_PATH)
                exit()
            else:
                print("Please answer 'y' or 'n'")

    else:
        print("Nothing to resume. Training a new agent.")
        os.makedirs(STATE_PATH)
        resume_training = False

    print("====================================================\n")

    # agent parameters
    memory_size = 25000
    window_length = 1
    total_nb_steps = 1000000
    exploration_anneal_nb_steps = 80000
    max_eps = 0.8
    min_eps = 0.1
    learning_warmup_nb_steps = 50
    target_model_update = 1e-2
    learning_rate = 1e-3

    # callback parameters
    model_checkpoint_interval = 5000 # steps
    verbose_level = 2                # 1 for step interval, 2 for episode interval
    log_interval = 200               # steps
    save_state_interval = 20          # episodes

    processor = RosbotProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', max_eps, min_eps, 0.0, exploration_anneal_nb_steps)

    # create or load model and memory
    if resume_training:
        model = load_model("%s/model.h5" % LAST_STATE_PATH)
        (memory, memory.actions, memory.rewards, memory.terminals, memory.observations) = pickle.load(open("%s/memory.pkl" % LAST_STATE_PATH, "rb"))
    else:
        model = create_network()
        memory = SequentialMemory(limit=memory_size, window_length=window_length)    

    Agent = None
    if hitl:
        Agent = HumanDQNAgent
    else:
        Agent = DQNAgent    
    # create the agent
    dqn = Agent(processor=processor, model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=learning_warmup_nb_steps,
                   target_model_update=target_model_update, policy=policy)
    dqn.compile(Adam(lr=learning_rate), metrics=['mae'])

    # model snapshot and tensorboard callbacks
    if resume_training:
        callback_tb = pickle.load(open("%s/tb_callback.pkl" % STATE_PATH, "rb"))
        (episode_nr, step_nr) = pickle.load(open("%s/parameters.pkl" % LAST_STATE_PATH, "rb"))
    else:    
        loggerpath, _ = GetLogPath(path=LOG_PATH, developerTestingFlag=False)
        callback_tb = TB_RL(None, loggerpath)
        tbfile = open("%s/tb_callback.pkl" % STATE_PATH, "wb")
        pickle.dump(callback_tb, tbfile)
        tbfile.close()
        episode_nr = 0
        step_nr = 0

    # other callbacks
    callback_poisonpill = TerminateTrainingOnFileExists(dqn, '/tmp/killrlreal')
    callback_modelinterval = ModelIntervalCheckpoint('%s/dqn_%s_step_{step:02d}.h5f' % (MODEL_PATH, NAME), model_checkpoint_interval, verbose=1) 
    callback_save_state = SaveDQNTrainingState(save_state_interval, STATE_PATH, memory, dqn, snapshot_limit=40)
    cbs = [callback_modelinterval, callback_tb, callback_save_state, callback_poisonpill]

    # train the agent
    dqn.fit(env, callbacks=cbs, action_repetition=1, nb_steps=total_nb_steps, visualize=False, 
                verbose=verbose_level, log_interval=log_interval, resume_episode_nr=episode_nr, resume_step_nr=step_nr)


if __name__ == "__main__":        
    train(True)
