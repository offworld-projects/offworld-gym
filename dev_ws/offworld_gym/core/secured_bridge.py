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
from offworld_gym.cert import x509
from threading import Thread, Lock
import json 
from socket import *
from tlslite.api import *
from offworld_gym import logger
from tlslite import HTTPTLSConnection

class SecuredBridge(metaclass=Singleton):
    """ Secured JSON-based communication over HTTPS
    This securely sends/recieves content to/from the OffWorld Gym server
    """
    settings_dict = settings.config["application"]["dev"]

    def __init__(self):
        self.server_ip = self.settings_dict["gym_server"]["server_ip"]
        self.secured_port = self.settings_dict["gym_server"]["secured_port"]
        self.network_mutex = Lock()
        self.server_socket = self.server_connection = None
        self._perform_handshake()
        self.authenticated, self.security_token = self._authenticate_authorize_user()

    def _perform_handshake(self):
        """Creates a secured HTTP connection with the OffWorld Gym Server        
        """
        self.https_client = HTTPTLSConnection(host=self.server_ip,
                                port=self.secured_port,
                                certChain=x509.cert_chain, 
                                privateKey=x509.private_key,
                                timeout=15)
        assert(self.https_client is not None)

    def _authenticate_authorize_user(self):
        """
        Performs authentication of the user and checks if the user is using the
        physical environment in their booked time slot. After authentication and
        authorization, an user is allowed access to the gym server. An access code
        is returned to the user and this access code must be included in every request.
        """        
        #TODO need to think about user authentication
        return True, 'yYs9kHDHE9wb3WUJXgKebKMDmCJyZFR3' # hard-coded security token for testing

    def get_security_token(self):
        """Return the security token
        """
        return self.security_token

    def send_request(self, object):
        if self.authenticated:
            self.network_mutex.acquire()
            try:
                message = json.dumps(object.__dict__) 
                pass
            finally:
                self.network_mutex.release()