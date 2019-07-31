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
from offworld_gym import logger
import numpy as np
import requests
import os 

# offworld gym
from offworld_gym.envs.real.config import settings
from offworld_gym.envs.common.oops import Singleton
from offworld_gym.envs.real.core.request import *
from offworld_gym.envs.common.exception.gym_exception import GymException
from offworld_gym.envs.common.channels import Channels

DEBUG = False

class SecuredBridge(metaclass=Singleton):
    """ Secured rest-based communication over HTTPS
    This securely sends/recieves content to/from the OffWorld Gym server
    """
    settings_dict = settings.config["application"]["dev"]
    _TELEMETRY_WAIT_TIME = 10

    def __init__(self):
        self.server_ip = self.settings_dict["gym_server"]["server_ip"]
        self.secured_port = self.settings_dict["gym_server"]["secured_port"]
        self._action_counter = 0
        self._certificate = False #os.path.join(os.path.dirname(os.path.realpath(__file__)), "../certs/gym_monolith/certificate.pem") #TODO find out why doesn't the certificate work, unverified certs can cause mitm attack

    def get_last_heartbeat(self):
        """Return the last heartbeat value
        """
        req = HeartBeatRequest()
        api_endpoint = "https://{}:{}/{}".format(self.server_ip, self.secured_port, HeartBeatRequest.URI)
        response = requests.post(url = api_endpoint, json = req.to_dict(), verify=self._certificate) 
        response_json = json.loads(response.text)
        if not DEBUG: logger.debug("Heartbeat  : {}".format(response_json['heartbeat']))
        return response_json['heartbeat']
        
    def perform_action(self, action_type, channel_type):
        """Perform an action on the robot
        """
        self._action_counter += 1
        if not DEBUG: logger.debug("Start executing action {}, count : {}.".format(action_type.name, str(self._action_counter)))
        
        req = ActionRequest(action_type=action_type, channel_type=channel_type)
        api_endpoint = "https://{}:{}/{}".format(self.server_ip, self.secured_port, ActionRequest.URI)

        response = requests.post(url = api_endpoint, json = req.to_dict(), verify=self._certificate) 
        response_json = json.loads(response.text)

        reward = int(response_json['reward'])
        state = json.loads(response_json['state'])
        done = bool(response_json['done'])

        state = np.asarray(state)
        state = np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2]))

        if not DEBUG: logger.debug("Reward  : {}".format(str(reward)))
        if not DEBUG: logger.debug("Is done : {}".format(str(done)))
        logger.info("Action execution complete. Telemetry recieved.")
        
        return state, reward, done
    
    def perform_reset(self, channel_type=Channels.DEPTH_ONLY):
        """Requests server to reset the environment
        """
        if DEBUG: logger.debug("Waiting for reset done from the server.")       
        
        req = ResetRequest(channel_type=channel_type)
        api_endpoint = "https://{}:{}/{}".format(self.server_ip, self.secured_port, ResetRequest.URI)
        response = requests.post(url = api_endpoint, json = req.to_dict(), verify=self._certificate) 
        response_json = json.loads(response.text)

        state = json.loads(response_json['state'])

        state = np.asarray(state)
        state = np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2]))
        logger.info('Environment reset done.')
        
        return state
