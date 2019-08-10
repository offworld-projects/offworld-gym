#!/usr/bin/env python
# Copyright offworld.ai 2019
from offworld_gym import version

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = version.__version__
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

import json
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions

class TokenRequest:
    """ Request model for getting a web token from the server
    """
    URI = "initiate"
    def __init__(self, api_token):
        self.api_token = api_token

    def to_json(self):
        return json.dumps(self.to_dict())
    
    def to_dict(self):
        return self.__dict__

class Request:
    """ Generitc environment request model
    """
    def __init__(self, web_token):
        self.web_token = web_token

    def to_json(self):
        return json.dumps(self.__dict__)
    
    def to_dict(self):
        return self.__dict__
        

class ActionRequest(Request):
    """ Environment action request model
    """
    URI = "action"

    def __init__(self, web_token, action_type=FourDiscreteMotionActions.FORWARD, channel_type=Channels.DEPTH_ONLY):
        Request.__init__(self, web_token)
        self.action_type = action_type.value if isinstance(action_type, FourDiscreteMotionActions) else action_type
        self.channel_type = channel_type.value if isinstance(channel_type, Channels) else channel_type


class ResetRequest(Request):
    """ Environment reset request model
    """    
    URI = "reset"

    def __init__(self, web_token, channel_type=Channels.DEPTH_ONLY):
        Request.__init__(self, web_token)
        self.channel_type = channel_type.value if isinstance(channel_type, Channels) else channel_type


class HeartBeatRequest(Request):
    """ Robot heartbeat request model
    """
    
    URI = "heartbeat"
    
    def __init__(self, web_token):
        Request.__init__(self, web_token)

