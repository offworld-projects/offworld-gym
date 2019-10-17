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

from offworld_gym.envs.common.data_structures import UniqueDict

class Singleton(type):
    """
    A meta class to define another class as a singleton class
    """
    
    _instances = UniqueDict()
    
    def __call__(singleton_class, *args, **kwargs):
        singleton_class._instances[singleton_class] = super(Singleton, singleton_class).__call__(*args, **kwargs)
        return singleton_class._instances[singleton_class]