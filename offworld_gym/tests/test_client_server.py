from offworld_gym.envs.real.core.secured_bridge import SecuredBridge
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.common.channels import Channels

mc = SecuredBridge()
mc.initiate()
import time
while mc.get_last_heartbeat() is None:
    time.sleep(1)
print("Connection established.")
state,reward,done = mc.perform_action(FourDiscreteMotionActions.FORWARD, Channels.DEPTH_ONLY)

mc.perform_reset()
