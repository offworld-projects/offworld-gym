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
    """Generic class to create requests for the offworld gym server
    """
    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password

    def to_json(self):
        """Returns a JSON representation of the request object
        """
        return json.dumps(self.__dict__)
    
    def to_dict(self):
        """Returns the class properties encapsulated in a dictionary
        """
        return self.__dict__
        

class ActionRequest(Request):
    """Create request for the action api
    """
    URI = "action"

    def __init__(self, username=None, password=None, action_type=FourDiscreteMotionActions.FORWARD, channel_type=Channels.DEPTH_ONLY):
        Request.__init__(self, username, password)
        self.action_type = action_type.value if isinstance(action_type, FourDiscreteMotionActions) else action_type
        self.channel_type = channel_type.value if isinstance(channel_type, Channels) else channel_type

class ResetRequest(Request):
    """Create request for the reset api
    """    
    URI = "reset"

    def __init__(self, username=None, password=None, channel_type=Channels.DEPTH_ONLY):
        Request.__init__(self, username, password)
        self.channel_type = channel_type.value if isinstance(channel_type, Channels) else channel_type
        self.no_action = False  

class HeartBeatRequest(Request):
    """Create request for the heartbeat api
    """    
    URI = "heartbeat"
    STATUS_RUNNING = "STATUS_RUNNING"
    
    def __init__(self, username=None, password=None):
        Request.__init__(self, username, password)
