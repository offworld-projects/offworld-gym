import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys, pdb
import numpy as np
from datetime import datetime

# configure tensorflow and keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))

import gym
import offworld_gym

import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from rl.callbacks import ModelIntervalCheckpoint

from utils import TB_RL, GetLogPath

# paths
OFFWORLD_GYM_ROOT = os.environ['OFFWORLD_GYM_ROOT']
LOG_PATH = '%s/logs/sim' % OFFWORLD_GYM_ROOT
MODEL_PATH = '%s/models/sim' % OFFWORLD_GYM_ROOT
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
NAME = 'dqn_offworld_monolith-{}'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

# Get the environment and extract the number of actions.
env = gym.make('OffWorldMonolithSimEnv-v0')
env.seed(123)
nb_actions = env.action_space.n


USE_BN = False
USE_LEAKY_RELU = True

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
    config_input = Input(shape=(1,), name='config_input')
        
    x = img_input
    for i in range(2):
        x = convBNRELU(x, filters=4, kernel_size=5, strides=2, id1=i, use_batch_norm=USE_BN, use_leaky_relu=USE_LEAKY_RELU)
        x = MaxPooling2D((2, 2))(x)

    x = convBNRELU(x, filters=1, kernel_size=5, strides=1, id1=9, use_batch_norm=USE_BN, use_leaky_relu=USE_LEAKY_RELU)
    x = Flatten()(x)
        
    x = keras.layers.concatenate([x, config_input])

    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
        
    output = Dense(nb_actions)(x)

    model = Model(inputs=[img_input, config_input], outputs=output)
    print(model.summary())
        
    return model


class RosbotProcessor(Processor):
    
    def process_observation(self, observation):
        '''
        observations are [image(240, 320, 3), config(4,)]
        '''
        return observation

    def process_state_batch(self, batch):
        '''
        stitch together images and configs
        '''
        imgs_batch = []
        configs_batch = []
        for exp in batch:
            imgs = []
            configs = []
            for state in exp:
                imgs.append(np.expand_dims(state[0], 0))
                #configs.append(np.expand_dims(state[1], 0)) # TODO: Change to real config if any
                configs.append(np.expand_dims(100, 0))
            imgs_batch.append(np.concatenate(imgs, -1))
            configs_batch.append(np.concatenate(configs, 0))
        imgs_batch = np.concatenate(imgs_batch, 0)
        configs_batch = np.concatenate(configs_batch, 0)
        return [imgs_batch, configs_batch]

def train():
    memory_size = 50000
    window_length = 1
    total_nb_steps = 1000000
    exploration_anneal_nb_steps = 40000
    max_eps = 0.8
    min_eps = 0.1
    learning_warmup_nb_steps = 50
    target_model_update = 1e-2
    learning_rate = 1e-3
    load_weights = False
    model_checkpoint_interval = 5000
    verbose_level = 2 # 1 == step interval, 2 == episode interval
    log_interval = 200

    model = create_network()
    memory = SequentialMemory(limit=memory_size, window_length=window_length)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', max_eps, min_eps, 0.0, exploration_anneal_nb_steps)
    processor = RosbotProcessor()
    dqn = DQNAgent(processor=processor, model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=learning_warmup_nb_steps,
                   target_model_update=target_model_update, policy=policy)
    dqn.compile(Adam(lr=learning_rate), metrics=['mae'])

    if load_weights:
        dqn.load_weights('dqn_{}_weights.h5f'.format(NAME))
        
    # model snapshot and tensorboard callbacks
    loggerpath, _ = GetLogPath(path=LOG_PATH, developerTestingFlag=False)
    cbs = [ModelIntervalCheckpoint('%s/dqn_%s_step_{step:02d}.h5f' % (MODEL_PATH, NAME), model_checkpoint_interval, verbose=1),
           TB_RL(None, loggerpath)]

    # with shovelbot gym-gazebo, it's important to do some action repetition 
    # because otherwise the wrong observation may be taken for an action
    action_repetition = 1
    dqn.fit(env, callbacks=cbs, action_repetition=action_repetition, nb_steps=total_nb_steps, visualize=False, 
            verbose=verbose_level, log_interval=log_interval)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_final_weights.h5f'.format(NAME), overwrite=True)
    pdb.set_trace()
        
    # Finally, evaluate our algorithm for 5 episodes.
    #dqn.test(env, nb_episodes=5, visualize=True)


def test():
    model = create_network()
    memory = SequentialMemory(limit=50000, window_length=1)
    processor = ShovelbotProcessor()
        
    dqn = DQNAgent(processor=processor, model=model, nb_actions=nb_actions, memory=memory, policy=None)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.load_weights('dqn_{}_final_weights.h5f'.format(NAME))

    # with shovelbot gym-gazebo, it's important to do some action repetition 
    # because otherwise the wrong observation may be taken for an action
    action_repetition = 1
    dqn.test(env, callbacks=[], nb_episodes=50, action_repetition=action_repetition, visualize=True)


if __name__ == "__main__":        
    train()
    #test()

