import offworld_gym
import gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
import pdb
from matplotlib import pyplot as plt

def test_env():
    env = gym.make('OffWorldMonolithDiscreteSim-v0', channel_type=Channels.RGB_ONLY)
    state, reward, done, info = env.step(FourDiscreteMotionActions.LEFT)
    assert state.shape == (1, 240, 320, 3)
    assert reward is not None
    assert not done
    assert not info
    plt.imshow(state[0])
    plt.show()
if __name__ == "__main__":
    test_env()