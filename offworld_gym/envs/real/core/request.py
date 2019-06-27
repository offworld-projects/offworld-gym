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

class OperationRequest:
    START_MISSION_OP = 'START_MISSION'
    SET_ARG_OP_MISSION_OP = 'SET_ARG_OP_MISSION'
    RUN_OP_MISSION = 'RUN_OP_MISSION'

    def __init__(self, op_name=START_MISSION_OP, *args, **kwargs):
        self.op_name = op_name
        self.args = args
        self.kwargs = kwargs
    
    def to_json(self):
        return json.dumps(self.__dict__)
    
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