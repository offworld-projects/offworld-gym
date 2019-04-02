from __future__ import print_function 
from offworld_gym import version

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = version.__version__
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

from offworld_gym.config import settings
from offworld_gym.utils.oops import Singleton
from threading import Thread, Lock
import json 

class SecuredBridge(metaclass=Singleton):
    """ Secured JSON-based communication over HTTPS
    This securely sends/recieves content to/from the OffWorld Gym server
    """
    def __init__(self):
        self.server_ip = settings.config["application"]["dev"]["gym_server"]["server_ip"]
        self.network_mutex = Lock()
        self.security_token = self._perform_handshake()

    def _perform_handshake(self):
        """Performs trusted handshake with the server to get the security token
        """        
        #TODO need to think about user authentication
        return 'yYs9kHDHE9wb3WUJXgKebKMDmCJyZFR3' # hard-coded security token for testing

    def get_security_token(self):
        """Return the security token
        """
        return self.security_token

    def send_request(self, object):
        self.network_mutex.acquire()
        try:
            message = json.dumps(object.__dict__) 
            pass
        finally:
            self.network_mutex.release()