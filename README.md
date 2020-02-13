# OffWorld Gym

The challenge that the community sets as a benchmark is usually the challenge that the community eventually solves. The ultimate challenge of reinforcement learning research is to train *real* agents to operate in the *real* environment, but until now there has not been a common real-world RL benchmark.  

OffWorld Gym is free to use, try it out at [gym.offworld.ai](https://gym.offworld.ai)

## Real-world Robotics Environment for Reinforcement Learning Research

We have created OffWorld Gym - a collection of real-world environments for reinforcement learning in robotics with free public remote access. Close integration into the existing ecosystem allows you to start using OffWorld Gym without any prior experience in robotics and removes the burden of managing a physical robotics system, abstracting it under a familiar API.

With this release we introduce our first prototype navigation task, where the robot has to reach the visual beacon over an uneven terrain using only the camera input.

When testing your next RL algorithm on Atari, why not also gauge its applicability to the real world!

Install the library, change your `gym.make('CartPole-v0')` to `gym.make('OffWorldMonolith-v0', ...)` and you are all set to run your RL algorithm on a **real robot**, for free!

![OffWorld Monolith environment](https://github.com/offworld-projects/offworld-gym/blob/develop/docs/images/offworld-gym-monolith-v2.png)  
Environment 1: OffWorld Monolith



## Getting access to OffWorld Gym Real
The main purpose of OffWorld Gym is to provide you with easy access to a physical robotic environment and allow you to train and test your algorithms on a real robotic system. To get access to the real robot, head to our web portal [gym.offworld.ai](https://gym.offworld.ai) and do the following:

  * Register as a user at [gym.offworld.ai](https://gym.offworld.ai).
  * [Book your experiment](https://gym.offworld.ai/book) using the OffWorld Gym resource management system.
  * Once you install the `offworld_gym` library, copy "OffWorld Gym Access Token" from your [Profile](https://gym.offworld.ai/account) page into `OFFWORLD_GYM_ACCESS_TOKEN` variable in your `offworld-gym/scripts/gymshell.sh` script.

The setup is complete! Now you can:

  * Read about our mission in the [About](https://gym.offworld.ai/about) section.
  * Browse the documentation, including the examples, at [gym.offworld.ai/docs](https://gym.offworld.ai/docs)
  * See the [Leaderboard](https://gym.offworld.ai/leaderboard), can your algorithm do better?
  * Run your experiments and monitor their performance under [My Experiments](https://gym.offworld.ai/myexperiments)

You can now install the `offworld_gym` library. Please follow the instructions in the [Installation](https://gym.offworld.ai/docs/installation.html) section of this documentation. Then proceed to the [Examples](https://gym.offworld.ai/docs/examples.html).



## Installation
There are two installation options: **real-only** (OS agnostic, no ROS dependencies) and **real-and-sim** (requires Ubuntu 16.04 and ROS Kinetic).  

Please check the [Installation](https://gym.offworld.ai/docs/installation.html) section of the documentation for the instructions.



## Examples
For a short example of how to connect to the real robot and interact with it please have a look at the [Minimal example in the Real environment](https://gym.offworld.ai/docs/examples.html).

We also provide examples where and agent achieves learning in both the real and the simulated environment. We use a slightly [modified version](https://github.com/offworld-projects/keras-rl/tree/offworld-gym) of [Keras-RL](https://github.com/keras-rl/keras-rl) library that allows us to make the training process resumable after an interruption. This is something that happens quite often when training in real. A set of `utils` allows you to visualize additional information on a TensorBoard. The `offworld_gym` library itself does not depend on these tools - you can ignore them, build on top of them or use them for inspiration. Keras-RL was our choice but you can use any other framework when developing your RL agents.

See the [Examples](https://gym.offworld.ai/docs/examples.html) section of the Docs for more details.
