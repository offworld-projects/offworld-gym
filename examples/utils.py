__author__      = "Toby Buckley"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__status__      = "Development"

import random
import numpy as np
import cv2
import rosbag
import tf
import rospy
import math, time, os, sys, csv, errno, re
import re
from math import sqrt
import scipy
import scipy.misc

from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
import skimage
import skimage.transform
from skimage.measure import block_reduce
import tensorflow
from keras import backend as K
backend = K.backend()
import pdb
import pickle
import warnings
import imageio
from collections import defaultdict

from tensorboard.backend.event_processing import event_multiplexer as event_multiplexer
import tensorboard.backend.event_processing.event_accumulator as event_accumulator

import signal
import matplotlib.pyplot as plt

from std_msgs.msg import Float32, Float32MultiArray, Bool

from rl.callbacks import ModelIntervalCheckpoint, FileLogger, TrainIntervalLogger, Callback
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess, RandomProcess 
from rl.policy import Policy, BoltzmannQPolicy
from rl.util import clone_model
from rl.core import Processor

from keras.layers import Lambda


# GLOBAL VARS
IMG_H = 240
IMG_W = 320
IMG_C = 3
NUM_CONFIGS = 6
NUM_CTRLS = 2

# Control downsampling: how many scalar data do we keep for each run/tag
# combination?
SIZE_GUIDANCE = {'scalars': 1000}
NON_ALPHABETIC = re.compile('[^A-Za-z0-9_]')


def GetLogPath(path=None, developerTestingFlag=True):
        '''
        Boosted from ur-interface: ur-interface/URBasic/dataLogging.py
        Setup a path where log files will be stored
        Path format .\[path]\YY-mm-dd\HH-MM-SS\
        '''
        if path is None:
            path = os.path.abspath('rr_log')
        else:
            path = path

        logDir = path
        if developerTestingFlag:
            directory = path
        else:
            directory =  os.path.join(path, time.strftime("%Y-%m-%d", time.localtime()), time.strftime("%H-%M-%S", time.localtime()))
        if not os.path.exists(directory):
            os.makedirs(directory)
        #print(directory + logDir)
        print("Log dir: {}".format(directory))
        return directory, logDir


# boosted from here: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
import matplotlib.cm
def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tensorflow.reduce_min(value) if vmin is None else vmin
    vmax = tensorflow.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tensorflow.squeeze(value)

    # quantize
    indices = tensorflow.to_int32(tensorflow.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = cm(np.arange(256))[:, :3]
    colors = tensorflow.constant(colors, dtype=tensorflow.float32)
    value = tensorflow.gather(colors, indices)
    #pdb.set_trace()
    return value

# boosted from here: https://gist.github.com/kukuruza/03731dc494603ceab0c5
def put_kernels_on_grid (kernel, rgb=False, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tensorflow.reduce_min(kernel)
  x_max = tensorflow.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tensorflow.pad(kernel, tensorflow.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]
  # put NumKernels to the 1st dimension
  x = tensorflow.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tensorflow.reshape(x, tensorflow.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tensorflow.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tensorflow.reshape(x, tensorflow.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tensorflow.transpose(x, (2, 1, 3, 0))

  # to tensorflow.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tensorflow.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x


class TB_convs(Callback):
    '''
    callback for TF tensorboard with keras
    tensorboard convolution layers + histograms
    '''
    def __init__(self, example_input, log_dir='./logs'):
        '''
        example_input = numpy array which will be fed through the network in order to
                        create activations for each layer
        '''
        super(TB_convs, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TB_convs callback only works with the TensorFlow backend.')
        self.log_dir = log_dir
        self.merged = None
        self.example_input = example_input

    def original_save_images(self, weight):
        mapped_weight_name = weight.name.replace(':', '_')
        w_img = tensorflow.squeeze(weight)
        shape = K.int_shape(w_img)
        if len(shape) == 2:  # dense layer kernel case
            if shape[0] > shape[1]:
                w_img = tensorflow.transpose(w_img)
                shape = K.int_shape(w_img)
            w_img = tensorflow.reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 4:  # convnet case
            if K.image_data_format() == 'channels_last':
                # switch to channels_first to display
                # every kernel as a separate image
                w_img = tensorflow.transpose(w_img, perm=[2, 0, 1])
                shape = K.int_shape(w_img)
            w_img = tensorflow.reshape(w_img, [shape[0],
                                        shape[1],
                                        shape[2],
                                        1])
        elif len(shape) == 1:  # bias case
            w_img = tensorflow.reshape(w_img, [1,
                                        shape[0],
                                        1,
                                        1])
        else:
            # not possible to handle 3D convnets etc.
            return

        shape = K.int_shape(w_img)
        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
        tensorflow.summary.image(mapped_weight_name, w_img)

    def gen_feed_dict(self):
        feed_dict = {}
        if len(self.model.model.inputs) > 1 or isinstance(self.example_input, dict):
            feed_dict = {inp.name: self.example_input[inp.name] for inp in self.model.model.inputs if self.example_input is not None}
        else:
            inp = self.model.model.inputs[0]
            if self.example_input is not None:
                feed_dict = {inp.name: self.example_input}
        return feed_dict

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.merged is None:
            for layer in model.layers:
                if self.example_input is not None and hasattr(layer, 'output'):
                    out = layer.output #colorize(layer.output, vmin=None, vmax=None, cmap='jet')
                    output_shape = K.int_shape(out)
                    if len(output_shape) == 4:
                        out = tensorflow.transpose(out, perm=[3, 1, 2, 0])
                        out = tensorflow.map_fn(lambda img: colorize(img, cmap='jet'), out)
                        tensorflow.summary.image('{}_out'.format(layer.name.replace(':', '_')), out, max_outputs=100)
                for weight in layer.weights:
                    # order: [h, w, c, numFilters]
                    mapped_weight_name = weight.name.replace(':', '_')
                    tensorflow.summary.histogram(mapped_weight_name, weight)
                    #pdb.set_trace()
                    # --------- ripped from https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L663 -------------
                    w_img = weight # want to keep channel=1 dimension #tensorflow.squeeze(weight)
                    shape = K.int_shape(w_img)
                    if len(shape) == 4:  # convnet case
                        ## expects [Y, X, NumChannels, NumKernels]
                        w_img = put_kernels_on_grid (weight)
                        if K.image_data_format() == 'channels_last':
                            # switch to channels_first to display
                            # every kernel as a separate image
                            w_img = tensorflow.transpose(w_img, perm=[3, 1, 2, 0])
                    else:
                        # not possible to handle 3D convnets etc.
                        continue
                    if w_img is None:
                        continue
                    shape = K.int_shape(w_img)
                    print("Weight shape: ", shape)
                    assert len(shape) == 4 and shape[-1] in [1, 3, 4] # greyscale, rgb, rgba
                    out = w_img #colorize(w_img, vmin=None, vmax=None, cmap='jet')
                    out = tensorflow.map_fn(lambda img: colorize(img, cmap='jet'), out)
                    tensorflow.summary.image(mapped_weight_name, out, max_outputs=100)
        self.merged = tensorflow.summary.merge_all()
        self.writer = tensorflow.summary.FileWriter(self.log_dir, self.sess.graph)
        #write initial values
        feed_dict = self.gen_feed_dict()
            
        #pdb.set_trace()
        result = self.sess.run([self.merged], feed_dict=feed_dict)
        summary_str = result[0]
        self.writer.add_summary(summary_str, 0)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # get images
        feed_dict = self.gen_feed_dict()
        result = self.sess.run([self.merged], feed_dict=feed_dict)
        summary_str = result[0]
        self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tensorflow.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

def process_depth_msg(depth_msg):
    '''converts a depth image into numpy float32 array'''
    cv_image = CvBridge().imgmsg_to_cv2(depth_msg, "32FC1")
    img = np.asarray(cv_image, dtype=np.float32)
    img = np.nan_to_num(img) # inf -> big value, nan -> zero
    img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
    return img


def process_img_msg(img_msg):
    ''' Unpacks ROS image, converts it to cv2, then to numpy.
    numpy image is currently downsampled by (2, 2, 1)'''

    img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
    # img = np.asarray(img)

    #TODO: make final decision on downsampling
    # img = block_reduce(img, block_size=(2, 2, 1), func=np.mean)

    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = scipy.misc.imresize(img, (IMG_H, IMG_W, IMG_C))

    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    return img


def corresponding_msg(bag_generator, sync_t):

    t = rospy.Time(0)
    between = []
    msg = None
    while (t < sync_t):
        try:
            top, msg, t = next(bag_generator)
            between.append(msg)
        except StopIteration:
            raise StopIteration

    return (msg, bag_generator, between)

def process_odom_msg(odom_msg):
    '''Unpacks ROS odom, transforms quaternion and packs state information into
    numpy array'''

    quaternion = (
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
    )
    #not completely necessary, but allows for intuitive reward design
    euler = tf.transformations.euler_from_quaternion(quaternion)

    #state: xy \dot{x}\dot{y} yaw \dot{yaw}
    state = [odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.twist.twist.linear.x,
            euler[2],
            odom_msg.twist.twist.angular.z]

    config = np.asarray([state])
    return config

def process_pose(pose):
    pose_vector = pose.pose_vector
    pose_vector = np.asarray([pose_vector])
    return pose_vector

def process_joint_msg(joint_msg):
    '''Unpacks ROS Twist, packs only active controls'''
    ctrl = joint_msg.position[:]
    ctrl = np.asarray([ctrl])
    return ctrl

def process_ctrl_msg(ctrl_msg):
    '''Unpacks ROS Twist, packs only active controls'''
    ctrl = [ctrl_msg.linear.x,
            ctrl_msg.angular.z]
    ctrl = np.asarray([ctrl])
    return ctrl

def process_model_msg(model_states_msg):
    '''Unpacks Model States, outputs target model locations.'''
    model_names = model_states_msg.name
    model_poses = model_states_msg.pose
    target_names = ['pebble', 'Goal']

    locations = []
    for i, model_name in enumerate(model_names):
        for target_name in target_names:
            if target_name in model_name:
                curr_position = model_poses[i].position
                locations.append([curr_position.x,
                                  curr_position.y,
                                  curr_position.z])

    return locations

def find_target(subject_odom, target_options):
    '''Find most appropriate target with current odometry value. Current
    approach is minimum XY-distance target'''

    min_distance = float('inf')
    for target in target_options:
        curr_distance = sqrt((subject_odom[0] - target[0])**2 +
                             (subject_odom[1] - target[1])**2)
        if curr_distance < min_distance:
            min_distance = curr_distance
            best_target = target

    best_target = np.asarray([best_target])
    return best_target

def in_FOV(m_x, m_y, o_x, o_y, o_yaw):
    '''Returns whether or not (m_x, m_y) is in 60 degree FOV of state (o_x, o_y,
    o_yaw)'''
    x = m_x - o_x
    y = m_y - o_y
    # rx = math.cos(-o_yaw) * x - math.sin(-o_yaw) * y
    # ry = math.sin(-o_yaw) * x + math.cos(-o_yaw) * y

    # angle = math.atan2(ry, rx)
    angle = math.atan2(y, x)

    return abs(o_yaw - angle) <= math.pi/8

                # curr_odom = np.asarray([[float(data['o_x']),
                                            # float(data['o_y']),
                                            # float(data['o_x_dot']),
                                            # float(data['o_y_dot']),
                                            # float(data['o_yaw']),
                                            # float(data['o_yaw_dot'])]])
                # curr_ctrl = np.asarray([[float(data['c_x'])/self.max_xctrl,
                                        # float(data['c_z'])/self.max_yawctrl]])

#TODO: this is currently very experimental, formalize later
def task_mode(odom, ctrl):
    """
    Task Modes:
        1. Drive towards pebble
        2. Find goal
        3. Drive towards goal
        4. Release pebble
    """
    curr_mode = [0.0, 0.0, 0.0, 0.0]
    if ctrl[0] < 0:
        curr_mode[3] = 1.0
    else:
        if abs(ctrl[1]) > .33:
            curr_mode[1] = 1.0
        else:
            if odom[2] < 0:
                curr_mode[0] = 1.0
            else:
                curr_mode[2] = 1.0
    return curr_mode



# ----------------------------------------------------------------------------------------

def test_exploration_policy(policy):
    class null_agent():
        def __init__(self):
            self.step = 0
            self.training = True
    max_steps = policy.nb_steps
    policy.agent = null_agent()
    x = np.array(range(max_steps))
    y = np.zeros(max_steps)
    for i in range(max_steps):
        policy.agent.step = i
        y[i] = policy.get_current_value()
    plt.plot(x, y)
    plt.show()

def rectMask(in_tensor, h, w, rect, name=""):
    v1, v2, v3, v4 = rect
    mask = createRectMask(h, w, v1, v2, v3, v4).flatten() # output is 1's where to mask
    mask = 1.0 - mask # keep the non-rectangle parts
    mask = K.variable(value=mask)
    
    out_tensor = Lambda(lambda x: x * mask, output_shape=(h*w,), name='rect_mask'+str(name))(in_tensor) #  keras.layers.Multiply()([in_tensor, mask])
    return out_tensor

def actionToXY(predicted, targetImgSize):
    # predicted is same order as keras
    # targetImgSize is order: h, w (same as keras)
    out = np.array( np.unravel_index(predicted.squeeze().argmax(), targetImgSize[0:2]), dtype='float32') # get x, y coords of max value
    outFlat = (out + 0.5) / targetImgSize[0:2] # get to center of cell, convert to ratio
    return outFlat #order: h, w

def actionToXYList(predicted, targetImgSize, num_peaks=15):
    predicted = predicted.squeeze()
    h, w = targetImgSize[0:2]
    if not np.array_equal(predicted.shape, [h, w] ):
        predicted = np.reshape(predicted, [h, w])
    return getPeaks(predicted, indices=True, num_peaks=num_peaks)

#def getImgSizes():
#    inputImgSize = (240, 320, 1) # for depth input
#    targetImgSize = (50, 50, 1)
#
#    return inputImgSize, targetImgSize, criticImgSize
def getChiselerInputs(data):
    curr_state = np.asarray([[float(data['x']),
                            float(data['y']),
                            float(data['z']),
                            float(data['r']),
                            float(data['p']),
                            float(data['yaw']),
                            ]])
    curr_action = np.asarray([[float(data['f_dAng'])]])
    return curr_state, curr_action

def actionIdxToXY(predictedIdx, targetImgSize):
    out = np.array( np.unravel_index(predictedIdx, targetImgSize[0:2]), dtype='float32')
    outFlat = (out + 0.5) / targetImgSize[0:2] # get to center of cell, convert to ratio
    return outFlat #order: h, w

def getPeaks(image, threshold=0.15, minPixelDist=3, indices=False, num_peaks=15):
    # REMINDER: this fxn switches the x-y order
    h, w = image.shape
    coordinates = peak_local_max(image, min_distance=minPixelDist, threshold_rel=threshold, num_peaks=num_peaks)
    if False:
        plt.imshow(image)
        plt.hold(True)
        plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.') # blue dots, and the coord sys is all setup correct
        plt.show()
        #pdb.set_trace()
    x_cam = (coordinates[:, 1:2] + 0.5) / w # switch order to: w, h
    y_cam = (coordinates[:, 0:1] + 0.5) / h # must be ratio so that we can upscale it to camera image
    if indices:
        return x_cam, y_cam, coordinates[:, 1:2], coordinates[:, 0:1]
    else:
        return x_cam, y_cam

def do_kdtree(targets,queries):
    # for each point in 'queries', find the closest point in 'targets', and return the index
    mytree = scipy.spatial.cKDTree(targets)
    dist, indexes = mytree.query(queries)
    return indexes

def generate_fake_experiences(nb_samples, env, callbacks):
    # generate fake experiences based off dataset that can be entered in memory in the following form:
    #self.memory.append(self.recent_observation, self.recent_action, reward, terminal, training=True)
    pass

class TB_RL(TB_convs):
    def __init__(self, *args, **kwargs):
        super(TB_RL, self).__init__(*args, **kwargs)
        self.totalSteps = 0

    def on_step_end(self, step, logs={}):
        # for use with keras-rl callbacks
        self.info = logs['info']
        self.wasFault = False

        if 'fault' in logs:
            self.wasFault = np.any(logs['fault'])
        
        if not self.wasFault:
            self.totalSteps += 1

    def on_episode_end(self, epoch, logs=None):
        # have to call on epoch end because that's what TB_convs has instead of episodes
        super(TB_RL, self).on_epoch_end(epoch, logs=logs)

class TB_RL_custom_rewards(TB_RL):
    """ Custom scalar rewards 
    insert a list of keywords 
    """
    def __init__(self, custom_rewards, *args, **kwargs):
        assert isinstance(custom_rewards, list)
        super(TB_RL_custom_rewards, self).__init__(*args, **kwargs)
        self.custom_rewards = custom_rewards
        self.r_dict = {}
        self.reset()
        self.episodeStep = 0

    def reset(self):
        for key in self.custom_rewards:
            self.r_dict[key] = 0
        self.episodeStep = 0

    def on_step_end(self, step, logs={}):
        # for use with keras-rl callbacks
        super(TB_RL_custom_rewards, self).on_step_end(step, logs=logs)
        
        if not self.wasFault:
            for key in self.r_dict: 
                if key in self.info: self.r_dict[key] += np.sum(self.info[key])
            self.episodeStep += 1

    def on_episode_end(self, epoch, logs=None):
        if not self.episodeStep == 0:
            #for key in self.r_dict: 
            #    logs[key] = self.r_dict[key] / float(self.episodeStep)
            super(TB_RL_custom_rewards, self).on_episode_end(epoch, logs=logs)
        #self.reset()

class TB_RL_chiseler(TB_RL_custom_rewards):
    def __init__(self, *args, **kwargs):
        super(TB_RL_chiseler, self).__init__(['successful', 'mass'], *args, **kwargs)
        
    def on_step_end(self, step, logs={}):
        super(TB_RL_chiseler, self).on_step_end(step, logs=logs)
        if not self.episodeStep == 0:
            reward = np.sum(logs['reward'])
            logs = {'reward': reward}
            for key in self.r_dict: 
                logs[key] = self.r_dict[key] / float(self.episodeStep)
            super(TB_RL_chiseler, self).on_episode_end(self.totalSteps, logs=logs)
        #self.reset()
    
    def on_episode_end(self, epoch, logs=None):
        self.reset()

class PublishQValue(Policy):
    def __init__(self, parent_policy, *args, **kargs):
        self.parent_policy = parent_policy
        self.pub = rospy.Publisher("/chiseler/maxQ", Float32, queue_size=5)

    @property
    def metrics_names(self):
        return self.parent_policy.metrics_names

    @property
    def metrics(self):
        return self.parent_policy.metrics

    def select_action(self, **kwargs):
        q_values = kwargs['q_values']
        pdb.set_trace()

        maxQValue = np.max(q_values)
        self.pub.publish(Float32(maxQValue))
        return self.parent_policy.select_action(**kwargs)

    def get_config(self):
        return self.parent_policy.get_config()

