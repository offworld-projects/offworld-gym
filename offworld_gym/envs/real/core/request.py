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

class Request:

    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password

    def to_json(self):
        return json.dumps(self.__dict__)
    
    def to_dict(self):
        return self.__dict__
        

class ActionRequest(Request):

    URI = "action"

    def __init__(self, username=None, password=None, action_type=FourDiscreteMotionActions.FORWARD, channel_type=Channels.DEPTH_ONLY):
        Request.__init__(self, username, password)
        self.action_type = action_type.value if isinstance(action_type, FourDiscreteMotionActions) else action_type
        self.channel_type = channel_type.value if isinstance(channel_type, Channels) else channel_type


class ResetRequest(Request):
    
    URI = "reset"

    def __init__(self, username=None, password=None, channel_type=Channels.DEPTH_ONLY):
        Request.__init__(self, username, password)
        self.channel_type = channel_type.value if isinstance(channel_type, Channels) else channel_type


class HeartBeatRequest(Request):
    
    URI = "heartbeat"
    
    def __init__(self, username=None, password=None):
        Request.__init__(self, username, password)

