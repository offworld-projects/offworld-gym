from enum import Enum
class Channels(Enum):
    """Channel Types of the camera
    """
    DEPTH_ONLY = 1
    RGB_ONLY = 3
    RGBD = 4