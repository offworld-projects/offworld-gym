from offworld_gym.envs.real.core.secured_bridge import SecuredBridge
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.common.channels import Channels

mc = SecuredBridge()
mc.initiate()
#mc.get_heart_beat()
mc.perform_action(FourDiscreteMotionActions(2), Channels(1))
