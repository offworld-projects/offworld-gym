from matplotlib import pyplot as plt
from offworld_gym.envs.real.core.secured_bridge import SecuredBridge
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.common.channels import Channels

mc = SecuredBridge()

print(mc.get_last_heartbeat())
print("Connection established.")
state,reward,done = mc.perform_action(FourDiscreteMotionActions.FORWARD, Channels.RGB_ONLY)
print(state.shape, reward, done)
plt.imshow(state[0])
plt.show()
state = mc.perform_reset(Channels.DEPTH_ONLY)
print(state.shape)