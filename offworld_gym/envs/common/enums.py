
from enum import Enum

class AlgorithmMode(Enum):
    """Algorithm run mode
    """
    TRAIN = "train"
    TEST = "test"

class LearningType(Enum):
    """Type of learning
    """
    
    END_TO_END = "end2end"
    SIM_2_REAL = "sim2real"
    HUMAN_DEMOS = "humandemos"