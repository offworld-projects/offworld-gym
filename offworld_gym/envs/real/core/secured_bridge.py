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

# offworld gym
from offworld_gym.envs.real.config import settings
from offworld_gym.envs.common.oops import Singleton
from offworld_gym.envs.real.core.request import *
from offworld_gym.envs.common.exception.gym_exception import GymException

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
        self.sio_client.on('message', self.recv_handler)
        self.sio_client.on('TM_SAMPLE', self.tm_sample_handler)
        #self.sio_client.on('OP_RESPONSE', self.op_response_handle)
        self.mission_request = MissionRequest()
        self.tm_sample = None

    def op_response_handle(self, message):
        msg_dict = json.loads(message)
        response_dict = json.loads(msg_dict['response'])

    def tm_sample_handler(self, message):
        msg_dict = json.loads(message)
        sample_dict = json.loads(msg_dict['sample'])
        self.tm_sample = sample_dict['value']

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
        self.mission_request.add_action_channel(action_type, channel_type)
        self.sio_client.emit('MISSION_CONTROL', self.mission_request.to_json())

    def get_rgb_state(self):
        return self.read_telemetry(TelemetryRequest.ENV_STATE_RGB)

    def get_depth_state(self):
        return self.read_telemetry(TelemetryRequest.ENV_STATE_DEPTH)

    def get_rgbd_state(self):
        return self.read_telemetry(TelemetryRequest.ENV_STATE_RGBD)
    
    def get_reward(self):
        return self.read_telemetry(TelemetryRequest.ENV_REWARD)
    
    def get_heart_beat(self):
        return self.read_telemetry(TelemetryRequest.ROSBOT_HEARTBEAT)
    
    def read_telemetry(self, telemetry_name):
        tr = TelemetryRequest()
        tr.set_telemetry_name(telemetry_name)
        self.sio_client.emit('TM_SAMPLE_REQUEST', tr.to_json())

        timeout = time.time() + 60*1
        while self.tm_sample is None: 
            if time.time() > timeout:
                raise GymException("Request timed out.")
            else:
                time.sleep(0.01)
        tmp = self.tm_sample 
        self.tm_sample = None
        return tmp
