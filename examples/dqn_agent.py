import os
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))

import sys, pdb
import numpy as np
from datetime import datetime

import offworld_gym
import gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions

import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from rl.callbacks import ModelIntervalCheckpoint

# Paths (without trailing slashes)
#from keras_rl_utils import TB_RL
#from offworld_utils import GetLogPath

# --------------------------------- make env ---------------------------------
#OFFWORLD_ROOT = os.environ['OFFWORLD_ROOT']
#LOG_PATH = '%s/logs/research-baselines/gazebo-pacman/dqn_shovelbot_vizdoom' % OFFWORLD_ROOT
#MODEL_PATH = '%s/Models/research-baselines/gazebo-pacman/dqn_shovelbot_vizdoom' % OFFWORLD_ROOT
OFFWORLD_ROOT = '.'
LOG_PATH = '.'
MODEL_PATH = '.'
ENV_NAME = 'OffWorldMonolithSimEnv-v0'
NAME = 'dqn_offworld_monolith-{}'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

env.seed(123)
nb_actions = env.action_space.n

USE_BN = False
USE_LEAKY_RELU = True

def convBNRELU(input1, filters=8, kernel_size = 5, strides = 1, id1=0, use_batch_norm=False, use_leaky_relu=False, leaky_epsilon=0.1):
    cname = 'conv%dx%d_%d'%(kernel_size, kernel_size, id1)
    bname = 'batch%d'%(id1+1) # hard-coded + 1 because the first layer takes batch0
    elu_name = 'elu_%d'%(id1+1)
    leaky_relu_name = 'leaky_relu_%d'%(id1+1)
    out = Conv2D(filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            name = cname)(input1)
    
    if use_batch_norm == True:
        out = BatchNormalization(name=bname)(out)
    
    if use_leaky_relu:
        out = LeakyReLU(leaky_epsilon, name=leaky_relu_name)(out)
    else:
        out = Activation('relu')(out)
    return out

# Next, we build the model.
def create_network():
        inputImgSize = env.observation_space.shape
        inputImgSize = inputImgSize[1:]
        IMG_H, IMG_W, IMG_C = inputImgSize
        img_input = Input(shape=inputImgSize, name='img_input')
        config_input = Input(shape=(1,), name='config_input') # health
        x = img_input

        for i in range(2):
                x = convBNRELU(x, filters = 4,
                        kernel_size = 5,
                        strides = 2,
                        id1 = i,
                        use_batch_norm = USE_BN,
                        use_leaky_relu = USE_LEAKY_RELU)

                x = MaxPooling2D((2, 2))(x)

        # ---------------------- conv flatten ----------------------
        x = convBNRELU(x, filters = 1,
                        kernel_size = 5,
                        strides = 1,
                        id1 = 9,
                        use_batch_norm = USE_BN,
                        use_leaky_relu = USE_LEAKY_RELU)

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

class ShovelbotProcessor(Processor):
    def process_observation(self, observation):
        '''
        shovelbot observations are [image(240, 320, 3), config(4,)]
        '''
        # squeezes the useless dimension out
        #observation[0] = np.squeeze(observation[0], axis=0)
        #observation[1] = np.squeeze(observation[1], axis=0)
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
                #configs.append(np.expand_dims(state[1], 0))
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
        action_repetition = 2
        load_weights = False
        example_img_path = OFFWORLD_ROOT + '/../../research-baselines/gazebo-pacman/assets/shovelbot_example_img.npy'
        model_checkpoint_interval = 5000
        verbose_level = 2 # 1 == step interval, 2 == episode interval
        log_interval = 200

        model = create_network()
        memory = SequentialMemory(limit=memory_size, window_length=window_length)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', max_eps, min_eps, 0.0, exploration_anneal_nb_steps)
        processor = ShovelbotProcessor()
        dqn = DQNAgent(processor=processor, model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=learning_warmup_nb_steps,
                       target_model_update=target_model_update, policy=policy)
        dqn.compile(Adam(lr=learning_rate), metrics=['mae'])

        if load_weights:
            dqn.load_weights('dqn_{}_weights.h5f'.format(NAME))
        
        # load example input
        img_input_ex = np.load(example_img_path)
        if img_input_ex.shape[0] != 1: img_input_ex = np.expand_dims(img_input_ex, axis=0)
        example_input = {'config_input': np.zeros(4), 'img_input': img_input_ex}

        # with shovelbot gym-gazebo, it's important to do some action repetition 
        # because otherwise the wrong observation may be taken for an action
        #loggerpath, _ = GetLogPath(path=LOG_PATH, developerTestingFlag=False)
        #cbs = [ModelIntervalCheckpoint('%s/dqn_extra_linear_%s_step_{step:02d}.h5f' % (MODEL_PATH, NAME), model_checkpoint_interval, verbose=1, total_steps=0), TB_RL(None, loggerpath)]
        #cbs = [ModelIntervalCheckpoint('%s/dqn_extra_linear_%s_step_{step:02d}.h5f' % (MODEL_PATH, NAME), model_checkpoint_interval, verbose=1, total_steps=0)]
        cbs = []
        dqn.fit(env, callbacks=cbs, action_repetition=action_repetition, nb_steps=total_nb_steps, visualize=False, 
                verbose=verbose_level, log_interval=log_interval)

        # After training is done, we save the final weights.
        dqn.save_weights('dqn_{}_weights.h5f'.format(NAME), overwrite=True)

        print("Reward actions: {}".format(env.reward_actions))
        print("Done flag actions: {}".format(env.done_actions))
        pdb.set_trace()
        # Finally, evaluate our algorithm for 5 episodes.
        #dqn.test(env, nb_episodes=5, visualize=True)


def test():
        example_img_path = OFFWORLD_ROOT + '/../../research-baselines/gazebo-pacman/assets/example_runway_shovelbot.npy'

        model = create_network()

        memory = SequentialMemory(limit=50000, window_length=1)
        processor = ShovelbotProcessor()
        
        dqn = DQNAgent(processor=processor, model=model, nb_actions=nb_actions, memory=memory, policy=None)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        dqn.load_weights('dqn_{}_weights.h5f'.format(NAME))

        # load example input
        img_input_ex = np.load(example_img_path)
        if img_input_ex.shape[0] != 1: img_input_ex = np.expand_dims(img_input_ex, axis=0)
        example_input = {'config_input:0': np.zeros((4, 1)), 'img_input:0': img_input_ex}

        # with shovelbot gym-gazebo, it's important to do some action repetition 
        # because otherwise the wrong observation may be taken for an action
        #loggerpath, _ = GetLogPath(path=LOG_PATH, developerTestingFlag=False)
        #cbs = [TB_RL(example_input, loggerpath)]
        cbs = []
        action_repetition = 2

        # evaluate the algorithm for 50 episodes.
        dqn.test(env, callbacks = cbs, nb_episodes=50, action_repetition=action_repetition, visualize=True)

if __name__ == "__main__":
        
    train()
    #test()

