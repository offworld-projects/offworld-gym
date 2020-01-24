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

import json
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions

class TokenRequest:
    """Request model for getting a web token from the server

    Attributes:
        URI: String constant containing uri to access initiate api.
        api_token: String with the user's api token.
    """
    URI = "initiate"
    def __init__(self, api_token):
        self.api_token = api_token

    def to_json(self):
        return json.dumps(self.to_dict())
    
    def to_dict(self):
        return self.__dict__

class Request:
    """Base class for backend api requests

    Attributes:
        web_token: String with the server's web token.
    """
    def __init__(self, web_token):
        self.web_token = web_token

    def to_json(self):
        """Returns a JSON representation of the request object
        """
        return json.dumps(self.__dict__)
    
    def to_dict(self):
        """Returns the class properties encapsulated in a dictionary
        """
        return self.__dict__        

class ActionRequest(Request):
    """Environment action request model

    Attributes:
        URI: String constant containing uri to access action api.
        web_token: String with the server's web token.
        channel_type: Channels type indicating type of channel for observation.
        action_type: FourDiscreteMotionActions type value indicating type of action.
    """
    URI = "action"

    def __init__(self, web_token, action_type=FourDiscreteMotionActions.FORWARD, channel_type=Channels.DEPTH_ONLY):
        Request.__init__(self, web_token)
        self.action_type = action_type.value if isinstance(action_type, FourDiscreteMotionActions) else action_type
        self.channel_type = channel_type.value if isinstance(channel_type, Channels) else channel_type

class ResetRequest(Request):
    """Environment reset request model

    Attributes:
        URI: String constant containing uri to access reset api.
        web_token: String with the server's web token.
        channel_type: Channels type value indicating type of channel for observation.
        no_value: Boolean value to determine if action is to be taken during reset.
    """    
    URI = "reset"

    def __init__(self, web_token, channel_type=Channels.DEPTH_ONLY):
        Request.__init__(self, web_token)
        self.channel_type = channel_type.value if isinstance(channel_type, Channels) else channel_type
        self.no_action = False  

class SetUpRequest(Request):
    """Robot heartbeat request model

    Attributes:
        URI: String constant containing uri to access setup api.
        web_token: String with the server's web token.
        experiment_name: String containing the experiment name.
        resume_experiment: Boolean indicating whether existing experiment is to be resumed.
    """
    
    URI = "setup"
    STATUS_RUNNING = "STATUS_RUNNING"
    
    def __init__(self, web_token, experiment_name, resume_experiment, learning_type, algorithm_mode):
        Request.__init__(self, web_token)
        self.experiment_name = experiment_name
        self.resume_experiment = resume_experiment
        self.learning_type = learning_type
        self.algorithm_mode = algorithm_mode


class DisconnectRequest(Request):
    """Server Disconnect request

    Attributes:
        URI: String constant containing uri to access disconnet api.
        web_token: String with the server's web token.
    """

    URI = "disconnect"
