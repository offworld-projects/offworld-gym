import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys, pdb
import numpy as np
from datetime import datetime
import pickle

# configure tensorflow and keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels

import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, BatchNormalization, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from rl.callbacks import ModelIntervalCheckpoint, Callback

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
env = gym.make('OffWorldMonolithSimEnv-v0', channel_type=Channels.DEPTH_ONLY)
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
        
    #img_input = Input(shape=input_image_size, name='img_input')
    #config_input = Input(shape=(1,), name='config_input')
    img_input = Input(shape=(240, 320, 4), name='img_input')
    #config_input = Input(shape=(1, ), name='config_input')
        
    x = img_input
    #x = Permute((2, 3, 1))(x)
    for i in range(2):
        x = convBNRELU(x, filters=4, kernel_size=5, strides=2, id1=i, use_batch_norm=USE_BN, use_leaky_relu=USE_LEAKY_RELU)
        x = MaxPooling2D((2, 2))(x)

    x = convBNRELU(x, filters=1, kernel_size=5, strides=1, id1=9, use_batch_norm=USE_BN, use_leaky_relu=USE_LEAKY_RELU)
    x = Flatten()(x)
        
    #x = keras.layers.concatenate([x, config_input])

    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
        
    output = Dense(nb_actions)(x)

    #model = Model(inputs=[img_input, config_input], outputs=output)
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
        #return [imgs_batch, configs_batch]
        return imgs_batch

class TerminateOnInterrupt(Callback):

    def __init__(self, agent):
        self.agent = agent

    def on_action_begin(self, action, logs):
        if os.path.exists('/tmp/killrlsim'):
            print("STOPPING THE LEARNING PROCESS")
            self.agent.interrupt = True
    

def train():
    resume_training = False

    memory_size = 25000
    window_length = 4
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
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', max_eps, min_eps, 0.0, exploration_anneal_nb_steps)
    processor = RosbotProcessor()
    
    # create or load memory
    if resume_training:
        (memory, memory.actions, memory.rewards, memory.terminals, memory.observations) = pickle.load(open("running_sim_dqn_memory.pkl", "rb"))
    else:
        memory = SequentialMemory(limit=memory_size, window_length=window_length)

    # create the agent
    dqn = DQNAgent(processor=processor, model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=learning_warmup_nb_steps,
                   target_model_update=target_model_update, policy=policy)
    dqn.compile(Adam(lr=learning_rate), metrics=['mae'])

    # load weights
    if resume_training:
        dqn.load_weights("running_sim_dqn_weights.h5f")
        episode_nr = pickle.load(open("running_sim_episode_nr.pkl", "rb"))
    else:
        episode_nr = 0

    # model snapshot and tensorboard callbacks
    if resume_training:
        callback_tb = pickle.load(open("running_sim_tb_callback.pkl", "rb"))
    else:    
        loggerpath, _ = GetLogPath(path=LOG_PATH, developerTestingFlag=False)
        callback_tb = TB_RL(None, loggerpath)
        tbfile = open("running_sim_tb_callback.pkl", "wb")
        pickle.dump(callback_tb, tbfile)
        tbfile.close()
    
    callback_modelinterval = ModelIntervalCheckpoint('%s/dqn_%s_step_{step:02d}.h5f' % (MODEL_PATH, NAME), model_checkpoint_interval, verbose=1) 
    cbs = [callback_modelinterval, callback_tb, TerminateOnInterrupt(dqn)]

    # with shovelbot gym-gazebo, it's important to do some action repetition 
    # because otherwise the wrong observation may be taken for an action
    action_repetition = 1
    dqn.fit(env, callbacks=cbs, action_repetition=action_repetition, nb_steps=total_nb_steps, visualize=False, 
                verbose=verbose_level, log_interval=log_interval, resume_episode_nr=episode_nr)

    pdb.set_trace()

    # Save agent state to be able to resume later
    print("Saving the state of the agent... please wait")
    
    memdump = (memory, memory.actions, memory.rewards, memory.terminals, memory.observations)
    memfile = open("running_sim_dqn_memory.pkl", "wb")
    pickle.dump(memdump, memfile)
    memfile.close()

    dqn.save_weights("running_sim_dqn_weights.h5f", overwrite=True)
    
    episode_nr = dqn.episodes_completed
    episodefile = open("running_sim_episode_nr.pkl", "wb")
    pickle.dump(episode_nr, episodefile)
    episodefile.close()
    
    print("State of the agent is saved.")


    # After training is done, we save the final weights.
    #dqn.save_weights('dqn_{}_final_weights.h5f'.format(NAME), overwrite=True)
    #pdb.set_trace()
        
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

