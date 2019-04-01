from __future__ import print_function 

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = "0.0.1"
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

from offworld_gym.config import settings

class SecuredBridge:
    """ Security Handler
    This is the security handler and securely sends/recieves
    content to/from the OffWorld Gym server
    """

    def __init__(self):
        self.server_ip = settings.config["application"]["dev"]["gym_server"]["server_ip"]