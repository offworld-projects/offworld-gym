Examples
========

In our examples we use slightly `modified version <https://github.com/offworld-projects/keras-rl/tree/offworld-gym>`_ of `Keras-RL <https://github.com/keras-rl/keras-rl>`_ library that allows us to make training process resumable after an interruption (something that happens quite often when training in real) and a set of ``utils`` to visualize additional information on a TensorBoard. The ``offworld_gym`` library itself does not depend on these tools, you can ignore them, build on top of them or use for inspiration. Keras-RL was our choice, you can use any other framework when developing your RL agents.

Real
----

By now you have registered at `gym.offworld.ai <https://gym.offworld.ai>`_, booked you experiment time using the `resource manager <https://gym.offworld.ai/book>`_ and have copied "OffWorld Gym Access Token" from your `Profile <https://gym.offworld.ai/account>`_ page into ``OFFWORLD_GYM_ACCESS_TOKEN`` variable in your ``offworld-gym/scripts/gymshell.sh`` script.

Execute ``source offworld-gym/scripts/gymshell.sh`` to prepare you environment and head to `offworld-gym/examples`. Run `python3.6 dqn_agent_real.py` to start training a DQN agent on a real robot! Note that it will only work if you have booked the time with the resource manager and the time of running the experiment is the time you've booked.

.. code:: bash

    source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
    cd $OFFWORLD_GYM_ROOT/examples
    python3.6 dqn_agent_real.py

.. note::
   When initializing new environment you need to give a unique name for each new experiment.

   .. code:: python

      env = gym.make('OffWorldMonolithRealEnv-v0', experiment_name='My new experiment',
                     resume_experiment=False, channel_type=Channels.DEPTH_ONLY)

   Alternatively you have the option to resume one of the previous experiments

   .. code:: python
   
      env = gym.make('OffWorldMonolithRealEnv-v0', experiment_name='My resumable experiment',
                     resume_experiment=True, channel_type=Channels.DEPTH_ONLY)

You will see the commands your agent is sending, the actions the robot is executing, episode progress and rewards, and other useful information. To monitor the behavior of the robot, head to `My Experiments <https://gym.offworld.ai/myexperiments>`_ section of the web page. Here you can find all the experiments you have conducted, the learning curves and other stats.

Currently active experiment will have the ``RUNNING`` indicator next to it, together with the ``SEE THE CAMERAS`` link, that gives you access to two overhead cameras positioned inside of the environment:

.. figure:: images/my-experiments.png

    List of my experiments and the link to access the overhead cameras


Camera feed will be active during the whole duration of your time slot.

.. figure:: images/cameras.png

    Two overhead cameras to monitor robot behavior.

We wish you the best of luck with your algorithm design and hope to see you on the `Leaderboard <https://gym.offworld.ai/leaderboard>`_!


Sim
---
Make sure you have executed ``source scripts/gymshell.sh`` before running any OffWorld Gym programs. This script takes care of setting the environment variables.

Start training by going to ``examples`` and running ``python3.6 dqn_agent_sim.py``. This will initialize the environment and start the training process, you can have a peak by running ``gzclient`` in a separate terminal.

The ``SaveDQNTrainingState`` callback will store model and memory snapshots every 100 episodes in the ``sim_agent_state`` directory. In case your process dies you can just restart the python script, confirm that you wish to resume learning from the latest snapshot, and the learning will continue. Since we are storing the DQN replay buffer alongside the model, the script saves only 3 last snapshots by default to save some storage space. Feel free to change that parameter or set it to `None` if you would like to keep all snapshots. You can also stop training manually by calling ``touch /tmp/killrlsim`` or pressing Ctrl+C (sometimes fails, better user the ``touch`` method).

Calling ``pkill -f ros`` is a good way to get rid of runaway ROS processes that might still be running if the process was not cleanly terminated.

By default the script is saving TensorBoard log data under `logs`, you can see it by running ``tensorboard --logdir=logs`` and opening `http://localhost:6006 
<http://localhost:6006>`_ in your web browser. 

.. figure:: images/running-sim-experiments.png

    Running Sim experiments