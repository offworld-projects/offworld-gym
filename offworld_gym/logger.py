"""
Custom logging for the OffWorld Gym client.
Perform terminal logging, file logging and network streaming of logs by default
To change the default behavion, edits the settings.yaml file
"""
from offworld_gym import version

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = version.__version__
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

import os
import socket
import logging, logging.handlers
import time
from offworld_gym.envs.real.config import settings

logger = logging.getLogger('offworld_gym')
logger.setLevel(logging.DEBUG)

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Terminal Handler
terminal_handler = logging.StreamHandler()
terminal_handler.setLevel(logging.DEBUG)
terminal_formatter = logging.Formatter(LOG_FORMAT)
terminal_handler.setFormatter(terminal_formatter)
logger.addHandler(terminal_handler)

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