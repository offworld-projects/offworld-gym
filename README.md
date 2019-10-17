# OffWorld Gym

The challenge that the community sets as a benchmark is usually the challenge that the community eventually solves. The ultimate challenge of reinforcement learning research is to train *real* agents to operate in the *real* environment, but until now there has not been a common real-world RL benchmark.

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
The installation was tested on: Ubuntu 16.04.6. Following these steps will prepare you for running both the Real and the Sim versions of OffWorld Gym. You will need to use **Python 3** with this environemt, Python 2 is not supported.

#### Pre-requisites
Please install the following components using the corresponding installation instructions.

  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)
  
For GPU support also install

  * [CUDA 10.0 Library](https://developer.nvidia.com/cuda-10.0-download-archive)
  * [cuDNN 7.0 Library](https://developer.nvidia.com/cudnn)



#### Setup
```
git clone https://github.com/offworld-projects/offworld-gym.git
cd offworld-gym/scripts
export OFFWORLD_GYM_ROOT=`pwd`/..
./install.sh
```

Each time you will be running OffWorld Gym, execute
```
source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
```
in each new terminal. Or add it  your ~/.bashrc by running
```
echo "source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh" >> ~/.bashrc
```

To test the Real environment:	

  * Register as a user at [gym.offworld.ai](https://gym.offworld.ai)
  * Copy "OffWorld Gym Access Token" from your [Profile](https://gym.offworld.ai/account) page into `OFFWORLD_GYM_ACCESS_TOKEN` variable in your `offworld_gym/scripts/gymshell.sh` script (this instruction will be repeated in the installation manual)
  * Follow the instructions in the [Examples](https://gym.offworld.ai/docs/examples.html) section of [documentation](https://gym.offworld.ai/docs)

To test Sim environment open two terminals, `source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh` in each of them, and run:  
```
terminal one: roslaunch gym_offworld_monolith env_bringup.launch  
terminal two: gzclient  
```


## Examples
In our examples we use a slightly [modified version](https://github.com/offworld-projects/keras-rl/tree/offworld-gym>) of [Keras-RL](https://github.com/keras-rl/keras-rl) library that allows us to make the training process resumable after an interruption. This is something that happens quite often when training in real. A set of `utils` allows you to visualize additional information on a TensorBoard. The `offworld_gym` library itself does not depend on these tools - you can ignore them, build on top of them or use them for inspiration. Keras-RL was our choice but you can use any other framework when developing your RL agents.

Please see the [Examples](https://gym.offworld.ai/docs/examples.html) section of the Docs.
