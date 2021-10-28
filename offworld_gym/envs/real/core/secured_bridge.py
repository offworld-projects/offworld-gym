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

# std 
from offworld_gym import logger
import numpy as np
import requests
import os 
import time
from http import HTTPStatus

# offworld gym
from offworld_gym.envs.real.config import settings
from offworld_gym.envs.common.oops import Singleton
from offworld_gym.envs.real.core.request import *
from offworld_gym.envs.common.exception.gym_exception import GymException
from offworld_gym.envs.common.channels import Channels

class SecuredBridge(metaclass=Singleton):
    """Secured rest-based communication over HTTPS.

    This securely sends/recieves content to/from the OffWorld Gym server.
    """
    settings_dict = settings.config["application"]["dev"]
    _TELEMETRY_WAIT_TIME = 10

    def __init__(self):
        self._server_ip = self.settings_dict["gym_server"]["server_ip"]
        self._secured_port = self.settings_dict["gym_server"]["secured_port"]
        self._action_counter = 0
        self._certificate = False #os.path.join(os.path.dirname(os.path.realpath(__file__)), "../certs/gym_monolith/certificate.pem") #TODO find out why doesn't the certificate work, unverified certs can cause mitm attack
        
    def _initiate_communication(self):
        """Validate api token, get web token for next request.

        Validates the api-token of an user and checks if user has access to the environment.
        """

        token_var = self.settings_dict["user"]["api_token"]
        if not token_var in os.environ:
            raise ValueError("Please update OFFWORLD_GYM_ACCESS_TOKEN environment variable with api-token.")

        if os.environ[token_var] is None or os.environ[token_var] == '':
            raise ValueError("Api-token is null or empty.")
        
        client_version = __version__
        req = TokenRequest(os.environ[token_var], client_version)
        api_endpoint = "https://{}:{}/{}".format(self._server_ip, self._secured_port, TokenRequest.URI)
        response = None
        try:
            response = requests.post(url=api_endpoint, json=req.to_dict(), verify=self._certificate)
            if response.status_code != HTTPStatus.BAD_REQUEST and response.status_code != HTTPStatus.INTERNAL_SERVER_ERROR:
                response_json = json.loads(response.text)
            else:
                raise Exception(json.loads(response.text)['errors'][0])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            raise GymException(f"A request error occurred:\n{err}")
        except Exception as err:
            if response is not None and response.status_code == HTTPStatus.NOT_FOUND:
                raise GymException("The robot is not available. The environment is possibly under MAINTENANCE.")
            elif response is not None and response.status_code == HTTPStatus.UNAUTHORIZED:
                raise GymException("Unauthorized. Most likely your time slot has ended. Please try again.")
            else:
                raise GymException(f"A server error has occurred. Please contact the support team: gym.beta@offworld.ai. {str(err)}")
        logger.debug("Web Token  : {}".format(response_json['web_token']))
        return response_json['web_token']

    def perform_handshake(self, experiment_name, resume_experiment, learning_type, algorithm_mode, environment_name):
        """Perform handshake with the gym server.

        To perform a handshake: initiate communication with the server, 
        get the robot's heartbeat, send experiment details to the server.

        Args:
            experiment_name: String value as the experiment name.
            resume_experiment: Boolean value to indicate if existing experiment is to be resumed.
            learning_type: String value indicating whether type is end2end, humandemos or sim2real.

        Returns:
            A string value with the heartbeat status.
            A boolean value to indicate whether experiment was registered or not.
            A string containing message from the server.
        """
    
        # Initiate communication by sharing the api-token
        self._web_token = self._initiate_communication()

        # Get the heartbeat of the robot
        # Share the experiment details with the server
        req = SetUpRequest(self._web_token, experiment_name, resume_experiment, learning_type, algorithm_mode, environment_name)
        api_endpoint = "https://{}:{}/{}".format(self._server_ip, self._secured_port, SetUpRequest.URI)

        set_up_response = None
        try:
            set_up_response = requests.post(url=api_endpoint, json=req.to_dict(), verify=self._certificate)
            set_up_response_json = json.loads(set_up_response.text)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            raise GymException(f"A request error occurred:\n{err}")
        except Exception as err:
            if set_up_response is not None and set_up_response.status_code == HTTPStatus.NOT_FOUND:
                raise GymException("The robot is not available. The environment is possibly under MAINTENANCE.")
            elif set_up_response is not None and set_up_response.status_code == HTTPStatus.UNAUTHORIZED:
                raise GymException("Unauthorized. Most likely your time slot has ended. Please try again.")
            else:
                raise GymException(f"A server error has occurred. Please contact the support team: gym.beta@offworld.ai.\n{err}")
        logger.debug("Heartbeat  : {}".format(set_up_response_json['heartbeat']))
        self._web_token = set_up_response_json['web_token']

        return set_up_response_json['heartbeat'], set_up_response_json['registered'], set_up_response_json['message']
        
    def monolith_discrete_perform_action(self, action_type, channel_type, algorithm_mode):
        """Perform an action on the robot

        Args:
            action_type: FourDiscreteMotionActions type value with the action to execute.
            channel_type: Channels type value, determines observation's channel.
            algorithm_mode: Whether algorithm is being run in train or test modde.

        Returns:
            A numpy array as the observation.
            An integer value represeting reward from the environment.
            A boolean value that indicates whether episode is done or not.
        """
        start_time = time.time()
        self._action_counter += 1
        logger.debug("Start executing action {}, count : {}.".format(action_type.name, str(self._action_counter)))
        
        req = MonolithDiscreteActionRequest(self._web_token, action_type=action_type, channel_type=channel_type, algorithm_mode=algorithm_mode)
        api_endpoint = "https://{}:{}/{}".format(self._server_ip, self._secured_port, MonolithDiscreteActionRequest.URI)

        response = None
        try:
            response = requests.post(url=api_endpoint, json=req.to_dict(), verify=self._certificate)
            response_json = json.loads(response.text)
            reward = int(response_json['reward'])
            state = json.loads(response_json['state'])
            done = bool(response_json['done'])
            if response is not None and bool(response_json['is_success']) == False:
                raise GymException(str(response_json['message']))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            raise GymException(f"A request error occurred:\n{err}")
        except Exception as err:            
            if response is not None and response.status_code == HTTPStatus.NOT_FOUND:
                raise GymException("The robot is not available. The environment is possibly under MAINTENANCE.")
            elif response is not None and response.status_code == HTTPStatus.UNAUTHORIZED:
                raise GymException("Unauthorized. Most likely your time slot has ended. Please try again.")
            else:
                raise GymException(f"A server error has occurred. Please contact the support team: gym.beta@offworld.ai.\n{err}")

        if 'testing' in response_json:
            raise GymException(response_json["message"])

        state = np.asarray(state)
        state = np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2]))

        logger.debug("Reward  : {}".format(str(reward)))
        logger.debug("Is done : {}".format(str(done)))

        self._web_token = response_json['web_token']
        logger.debug("Action execution complete. Telemetry recieved. Total time to execute: {}.".format(str(time.time() - start_time)))
        
        return state, reward, done

    def monolith_continous_perform_action(self, action, channel_type, algorithm_mode):
        """Perform an action on the robot

        Args:
            action: Two dimensional action value with the action to execute.
            channel_type: Channels type value, determines observation's channel.
            algorithm_mode: Whether algorithm is being run in train or test modde.

        Returns:
            A numpy array as the observation.
            An integer value represeting reward from the environment.
            A boolean value that indicates whether episode is done or not.
        """
        start_time = time.time()
        self._action_counter += 1
        logger.debug("Start executing action {}, count : {}.".format(np.array2string(action), str(self._action_counter)))
        
        req = MonolithContinousActionRequest(self._web_token, action=action.tolist(), channel_type=channel_type, algorithm_mode=algorithm_mode)
        api_endpoint = "https://{}:{}/{}".format(self._server_ip, self._secured_port, MonolithContinousActionRequest.URI)

        response = None
        try:
            response = requests.post(url=api_endpoint, json=req.to_dict(), verify=self._certificate)
            response_json = json.loads(response.text)
            reward = int(response_json['reward'])
            state = json.loads(response_json['state'])
            done = bool(response_json['done'])
            if response is not None and bool(response_json['is_success']) == False:
                raise GymException(str(response_json['message']))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            raise GymException(f"A request error occurred:\n{err}")
        except Exception as err:            
            if response is not None and response.status_code == HTTPStatus.NOT_FOUND:
                raise GymException("The robot is not available. The environment is possibly under MAINTENANCE.")
            elif response is not None and response.status_code == HTTPStatus.UNAUTHORIZED:
                raise GymException("Unauthorized. Most likely your time slot has ended. Please try again.")
            else:
                raise GymException(f"A server error has occurred. Please contact the support team: gym.beta@offworld.ai.\n{err}")

        if 'testing' in response_json:
            raise GymException(response_json["message"])

        state = np.asarray(state)
        state = np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2]))

        logger.debug("Reward  : {}".format(str(reward)))
        logger.debug("Is done : {}".format(str(done)))

        self._web_token = response_json['web_token']
        logger.debug("Action execution complete. Telemetry recieved. Total time to execute: {}.".format(str(time.time() - start_time)))
        
        return state, reward, done
    
    def monolith_discrete_perform_reset(self, channel_type=Channels.DEPTH_ONLY):
        """Requests server to reset the environment.

        Args:
            channel_type: Channels type value, determines observation's channel.

        Returns:
            A numpy array as the observation.
        """
        logger.debug("Waiting for reset done from the server.")       
        
        req = MonolithDiscreteResetRequest(self._web_token, channel_type=channel_type)
        api_endpoint = "https://{}:{}/{}".format(self._server_ip, self._secured_port, MonolithDiscreteResetRequest.URI)

        response = None
        try:
            response = requests.post(url=api_endpoint, json=req.to_dict(), verify=self._certificate)
            response_json = json.loads(response.text)
            state = json.loads(response_json['state'])
            if response is not None and bool(response_json['is_success']) == False:
                raise GymException(str(response_json['message']))
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            raise GymException(f"A request error occurred:\n{err}")
            
        except Exception as err:
            if response is not None and response.status_code == HTTPStatus.NOT_FOUND:
                raise GymException("The robot is not available. The environment is possibly under MAINTENANCE.")
            elif response is not None and response.status_code == HTTPStatus.UNAUTHORIZED:
                raise GymException("Unauthorized. Most likely your time slot has ended. Please try again.")
            else:
                raise GymException(f"A server error has occurred. Please contact the support team: gym.beta@offworld.ai.\n{err}")

        state = np.asarray(state)
        state = np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2]))
        logger.debug('Environment reset done. The state shape is: '+ str(state.shape))

        self._web_token = response_json['web_token']
        
        return state
    
    def monolith_continous_perform_reset(self, channel_type=Channels.DEPTH_ONLY):
        """Requests server to reset the environment.

        Args:
            channel_type: Channels type value, determines observation's channel.

        Returns:
            A numpy array as the observation.
        """
        logger.debug("Waiting for reset done from the server.")       
        
        req = MonolithDiscreteResetRequest(self._web_token, channel_type=channel_type)
        api_endpoint = "https://{}:{}/{}".format(self._server_ip, self._secured_port, MonolithDiscreteResetRequest.URI)

        response = None
        try:
            response = requests.post(url=api_endpoint, json=req.to_dict(), verify=self._certificate)
            response_json = json.loads(response.text)
            state = json.loads(response_json['state'])
            if response is not None and bool(response_json['is_success']) == False:
                raise GymException(str(response_json['message']))
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            raise GymException(f"A request error occurred:\n{err}")
            
        except Exception as err:     
            if response is not None and response.status_code == HTTPStatus.NOT_FOUND:
                raise GymException("The robot is not available. The environment is possibly under MAINTENANCE.")
            elif response is not None and response.status_code == HTTPStatus.UNAUTHORIZED:
                raise GymException("Unauthorized. Most likely your time slot has ended. Please try again.")
            else:
                raise GymException(f"A server error has occurred. Please contact the support team: gym.beta@offworld.ai.\n{err}")

        state = np.asarray(state)
        state = np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2]))
        logger.debug('Environment reset done. The state shape is: '+ str(state.shape))

        self._web_token = response_json['web_token']
        
        return state
    
    def disconnect(self, channel_type, discrete=True):
        """Disconnect from the backend.
        """
        logger.debug("Disconnecting from the server.") 

        uri = DisconnectRequest.URI_DISCRETE if discrete else DisconnectRequest.URI_CONTINOUS
        req = DisconnectRequest(self._web_token, channel_type=channel_type)
        api_endpoint = "https://{}:{}/{}".format(self._server_ip, self._secured_port, uri)
        try:
            response = requests.post(url = api_endpoint, json = req.to_dict(), verify=self._certificate)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            raise GymException(f"A request error occurred:\n{err}")
        except Exception as err:
            raise GymException(
                f"A server error has occurred. Please contact the support team: gym.beta@offworld.ai.\n{err}")
