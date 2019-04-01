"""
Custom logging for the OffWorld Gym client.
Perform terminal logging, file logging and network streaming of logs by default
To change the default behavion, edits the settings.yaml file
"""

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = "0.0.1"
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

import os
import socket
import logging 
import time
from offworld_gym.config import settings

logger = logging.getLogger('offworld_gym')
logger.setLevel(logging.DEBUG)

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Terminal Handler
terminal_handler = logging.StreamHandler()
terminal_handler.setLevel(logging.DEBUG)
terminal_formatter = logging.Formatter(LOG_FORMAT)
terminal_handler.setFormatter(terminal_formatter)
logger.addHandler(terminal_handler)

# File Stream Handler 
file_flag = settings.config["application"]["dev"]["log"]["disable_file_log_stream"]
if not file_flag:
    directory_name = settings.config["application"]["dev"]["log"]["log_folder"]
    log_directory = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."), directory_name)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_name = os.path.join(log_directory, "real_gym_{}.log".format(time.strftime("%Y-%m-%d_%H:%M:%S")))
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    
# Network Stream handler
network_flag = settings.config["application"]["dev"]["log"]["disable_network_log_stream"]
if not network_flag:
    server_ip = settings.config["application"]["dev"]["log"]["log_server_ip"]
    if server_ip is not None or server_ip != '':
        try:
            socket.inet_pton(socket.AF_INET, server_ip) # check if ip is valid
        finally:
            server_port =  80
            socket_handler = logging.handlers.SocketHandler(server_ip, server_port)
            logger.addHandler(socket_handler)

def debug(msg):
    logger.debug(msg)

def info(msg):
    logger.info(msg)

def warn(msg):
    logger.warn(msg)

def error(msg):
    logger.error(msg)

def critical(msg):
    logger.critical(msg)