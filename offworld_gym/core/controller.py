from offworld_gym import version

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = version.__version__
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

from offworld_gym import logger
from offworld_gym.core.secured_bridge import SecuredBridge
from offworld_gym.utils.oops import Singleton
import numpy as np
import zlib
import json

class Message(object):
    """ Generic message container
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class StateMessage:
    """Message for the current state request
    """
    def __init__(self, security_token):
        self.msg = Message(action='state', token=security_token)

class ActionMessage:
    """Message to trigger the action on the robot
    """
    def __init__(self, action, security_token):
        self.msg = Message(action='action', command=action, token=security_token)

class GymController(metaclass=Singleton):
    """Request controller for the OffWorld Gym library
    """
    def __init__(self):
        self.secured_bridge = SecuredBridge()   

    
    def get_state(self):
        """Send a request for the current state from server
        """
        request_msg = StateMessage(self.secured_bridge.get_security_token())
        response = self.secured_bridge.send_request(request_msg)
        state = self._post_process_state_response(response)
        return state
    
    def do_action(self, action):
        """Send a request to trigger action
        """        
        request_msg = ActionMessage(action, self.secured_bridge.get_security_token())
        response = self.secured_bridge.send_request(request_msg)
        next_state, reward = self._post_process_action_response(response)
        return next_state, reward

    def _post_process_state_response(self, response_msg):
        """Post process the response from the server to extract state info
        """
        assert(response_msg is not None)
        msg_dict = json.loads(response_msg)
        uncompressed_body = zlib.decompress(msg_dict["body"])
        state = np.fromstring(uncompressed_body)
        return state

    def _post_process_action_response(self, response_msg):
        """Post process the response from the server to extract next state and reward info
        """
        assert(response_msg is not None)
        msg_dict = json.loads(response_msg)
        uncompressed_body = zlib.decompress(msg_dict["body"]["next_state"])
        next_state = np.fromstring(uncompressed_body)
        reward = msg_dict["body"]["reward"]
        return next_state, reward

    


