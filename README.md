# OffWorld Gym
The challenge that the community sets as a benchmark is usually the challenge that the community eventually solves. The ultimate challenge of reinforcement learning research is to train *real* agents to operate in the *real* environment, but until now there has not been a common real-world RL benchmark.

We have created OffWorld Gym - a collection of real-world environments for reinforcement learning in robotics with free public remote access. Close integration into existing ecosystem allows you to start using OffWorld Gym without any prior experience in robotics and takes away the burden of managing a physical robotics system, abstracting it under a familiar API.

With this release we introduce our first prototype navigation task, where the robot has to reach the visual beacon on an uneven terrain using only the camera input.

When you will be testing your next RL algorithm on Atari, why not also gauge it's applicability to the real world!



### Real-world robotics environment for Reinforcement Learning research

Install the library, change your `gym.make('CartPole-v0')` to `gym.make('OffWorldMonolith-v0')` and you are all set to run your RL algorithm on a **real robot**, for free!

![OffWorld Monolith environment](https://github.com/offworld-projects/offworld-gym/blob/develop/docs/images/offworld-gym-monolith-v1.png)  
Environment 1: OffWorld Monolith



### Getting access to OffWorld Gym Real
*(section about registration and resource manager)*



### Installation
The installation was tested on: Ubuntu 16.04.6. Following these steps will prepare you for running both the Real and the Sim versions of the OffWorld Gym. You will be able to use Python 3 with this environemt.

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

To prepare a terminal for running OffWorld Gym, run
```
source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
```
in each new terminal. Or add it  your ~/.bashrc by running
```
echo "source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh" >> ~/.bashrc
```

To test Real environment:	
	(add instructions here)

To test Sim environment open two terminals, `source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh` in each of them, and run:  

	1. `roslaunch gym_offworld_monolith env_bringup.launch`  
	2. `gzclient`  



### Examples
*(have one real and one simulated example)*
