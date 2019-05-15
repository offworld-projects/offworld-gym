import offworld_gym
import gym
from offworld_gym.envs.common import FourDiscreteMotionActions
from offworld_gym.envs.common import Channels
import pdb
from matplotlib import pyplot as plt

def test_env():

    env = gym.make('RosbotMonolithSimEnv-v0', channel_type=Channels.RGB_ONLY)
    state, reward, done, info = env.step(FourDiscreteMotionActions.LEFT)
    assert state.shape == (1, 240, 320, 3)
    assert reward is not None
    assert not done
    assert not info
    plt.imshow(state[0])
    plt.show()
if __name__ == "__main__":
    test_env()
