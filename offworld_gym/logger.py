#!/usr/bin/env python

# Copyright 2019 OffWorld Inc.
# Doing business as Off-World AI, Inc. in California.
# All rights reserved.
#
# Licensed under GNU General Public License v3.0 (the "License")
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law, any source code or other materials
# distributed under the License is distributed on an "AS IS" basis,
# without warranties or conditions of any kind, express or implied.

from offworld_gym import version

__version__     = version.__version__

import os
import socket
import logging, logging.handlers
import time
from offworld_gym.envs.real.config import settings

logger = logging.getLogger('offworld_gym')
if settings.config["application"]["dev"]["debug"]:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

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