from __future__ import print_function 
from offworld_gym import version

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = version.__version__
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

# std
from threading import Thread, Lock
import json 
from socket import *
from tlslite.api import *
from offworld_gym import logger
from tlslite import HTTPTLSConnection
import socketio
import time
from datetime import datetime

# offworld gym
from offworld_gym.envs.real.config import settings
from offworld_gym.envs.common.oops import Singleton
from offworld_gym.envs.real.core.request import *
from offworld_gym.envs.common.exception.gym_exception import GymException

DEBUG = True

class SecuredBridge(metaclass=Singleton):
    """ Secured websocket-based communication over HTTPS
    This securely sends/recieves content to/from the OffWorld Gym server
    """
    settings_dict = settings.config["application"]["dev"]

    def __init__(self):
        self.server_ip = self.settings_dict["gym_server"]["server_ip"]
        self.secured_port = self.settings_dict["gym_server"]["secured_port"]
        self.last_message = None
        self.sio_client = socketio.Client()
        self.last_message = None
        self.sio_client.on('connect', self.connect_handler)
        self.sio_client.on('TM_SAMPLES', self.tm_samples_handler)
        self._last_sample_time = None
        self._last_heartbeat_value = None
        self._last_done_value = None
        self._last_next_state_value = None
        self._last_reward_value = None
        self._is_reset = None

    def tm_samples_handler(self, message):
        msg_dict = json.loads(message)
        if 'samples' in msg_dict:
            samples = msg_dict['samples']
            _time_read = False
            for sample in samples:
                if 'time' in sample and not _time_read:
                    self._last_sample_time = datetime.strptime(sample['time'], '%Y-%m-%dT%H:%M:%S.%f')
                    _time_read = True
                # parse the value in the sample to get done, reward, next_state, heartbeat values
                try:
                    if sample is not None and 'value' in sample and isinstance(sample['value'], str):
                        val_dict = json.loads(sample['value'])
                        if isinstance(val_dict, dict):
                            if 'done' in val_dict:
                                if DEBUG: logger.debug("done received")
                                self._last_done_value = bool(val_dict['done'])
                            if 'reward' in val_dict:
                                if DEBUG: logger.debug("reward received")
                                self._last_reward_value = bool(val_dict['reward'])
                            if 'heartbeat' in val_dict:
                                if DEBUG: logger.debug("heart beat received")
                                self._last_heartbeat_value = bool(val_dict['heartbeat'])
                            if 'next_state' in val_dict:
                                if DEBUG: logger.debug("next state received")
                                self._last_next_state_value = bool(val_dict['next_state']) 
                            if 'reset' in val_dict:
                                if DEBUG: logger.debug("reset done flag received")
                                self._last_next_state_value = bool(val_dict['reset']) 
                except json.decoder.JSONDecodeError as e:
                    pass
            if DEBUG: logger.debug("Samples arrived.")

    def get_last_heartbeat(self):
        """Return the last heartbeat value
        """
        return self._last_heartbeat_value

    def connect_handler(self):
        """Perform action on connect
        """
        logger.info("Connected with the mission server id: {}".format(str(self.sio_client.eio.sid)))

    def recv_handler(self, message):
        """Perform action on message receive
        """
        self.last_message = message

    def initiate(self):
        """Initiate the mission connection
        """
        self.sio_client.connect('http://{}:{}'.format(self.server_ip, int(self.secured_port)))
        
    def perform_action(self, action_type, channel_type):
        """Perform an action on the robot
        """
        
        self._last_done_value = None
        self._last_reward_value = None

        set_action_type_request = {
            'op_index' : 1,
            'arg_name' : 'env_action_type',
            'arg_value' : action_type.value
        }
        set_arg_op_request = OperationRequest(OperationRequest.SET_ARG_OP_MISSION_OP, **set_action_type_request)
        self.sio_client.emit('MISSION_CONTROL', set_arg_op_request.to_json())

        set_channel_type_request = {
            'op_index' : 1,
            'arg_name' : 'env_channel_type',
            'arg_value' : channel_type.value
        }
        set_arg_op_request = OperationRequest(OperationRequest.SET_ARG_OP_MISSION_OP, **set_channel_type_request)
        self.sio_client.emit('MISSION_CONTROL', set_arg_op_request.to_json())

        mission_start_request = OperationRequest(OperationRequest.START_MISSION_OP)
        self.sio_client.emit('MISSION_CONTROL', mission_start_request.to_json())

        #collect samples
        if DEBUG: logger.debug("Waiting for telemetry from the server.")
        while self._last_done_value is None:
            time.sleep(0.5)

        while self._last_reward_value is None:
            time.sleep(0.5)

        state = self._last_done_value
        logger.info("Telemetry recieved.")
        return self._last_next_state_value, self._last_reward_value, self._last_done_value
    
    def perform_reset(self):
        """Requests server to reset the environment
        """
        self._is_reset = None
        args = ['ENV_RESET']
        kwargs = { 
            'private_op' : True
        }
        reset_request = OperationRequest(OperationRequest.RUN_OP_MISSION, *args, **kwargs)
        self.sio_client.emit("MISSION_CONTROL", reset_request.to_json()) 
        if DEBUG: logger.debug("Waiting for reset done from the server.")
        while self._is_reset is None:
            time.sleep(0.5)
        logger.info('Environment reset done.')
