Introduction
============

The challenge that the community sets as a benchmark is usually the challenge that the community eventually solves. The ultimate challenge of reinforcement learning research is to train *real* agents to operate in the *real* environment, but until now there has not been a common real-world RL benchmark.

We have created OffWorld Gym - a collection of real-world environments for reinforcement learning in robotics with free public remote access. Close integration into existing ecosystem allows you to start using OffWorld Gym without any prior experience in robotics and takes away the burden of managing a physical robotics system, abstracting it under a familiar API.

With this release we introduce our first prototype navigation task, where the robot has to reach the visual beacon on an uneven terrain using only the camera input.

When you will be testing your next RL algorithm on Atari, why not also gauge its applicability to the real world!



**OffWorld Gym: real-world robotics environment for Reinforcement Learning research**

Install the library, change ``gym.make('CartPole-v0')`` to ``gym.make('OffWorldMonolithRealEnv-v0')`` and you are all set to run your RL algorithm on a **real robot**, for free!

.. figure:: images/offworld-gym-monolith-v2.png

    Environment 1: OffWorld Monolith