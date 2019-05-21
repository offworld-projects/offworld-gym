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

from enum import Enum
import json

from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.common.channels import Channels

class MissionRequest:
    ENV_EXECUTE_RESULT = 'OPS:RESULT:ENV_EXECUTE'
    
    def __init__(self, op_name='START_MISSION', telemetry_name=ENV_EXECUTE_RESULT, *args, **kwargs):
        self.mission = ActionRequest(op_name, args, kwargs)
        self.telemetry_name = telemetry_name

    def add_action_channel(self, action_type, channel_type):
        self.mission.add_action_channel(action_type, channel_type)
    
    def to_json(self):
        json_dict = {}
        json_dict['mission'] = self.mission.__dict__
        json_dict['telemetry_name'] = self.telemetry_name
        return json.dumps(json_dict)

class ActionRequest:
    def __init__(self, op_name='START_MISSION', *args, **kwargs):
        self.op_name = op_name
        self.args = args
        self.kwargs = kwargs
    
    def add_action_channel(self, action_type, channel_type):
        assert isinstance(action_type, FourDiscreteMotionActions), "Action can only be of type: ActionType."
        assert isinstance(channel_type, Channels), "Channel can only be of type: Channels."
        self.kwargs = {'action_type': action_type.value, 'channel_type': channel_type.value}
    
class TelemetryRequest:
    ROSBOT_HEARTBEAT = "TM:S:ROSBOT_HEARTBEAT"
    ENV_STATE_DEPTH = "TM:S:ENV_STATE_DEPTH"
    ENV_STATE_RGB = "TM:S:ENV_STATE_RGB"
    ENV_STATE_RGBD = "TM:S:ENV_STATE_RGBD"

    def __init__(self, telemetry_name=ROSBOT_HEARTBEAT):
        self.telemetry_name = telemetry_name

    def set_telemetry_name(self, telemetry_name):
        assert telemetry_name is not None, "Telemetry name cannot be None."
        assert telemetry_name != '', "Telemetry name cannot be empty."
        self.telemetry_name = telemetry_name

    def to_json(self):
        return json.dumps(self.__dict__)