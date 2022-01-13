# Reference: https://github.com/ros-teleop/teleop_twist_keyboard/blob/master/teleop_twist_keyboard.py
from __future__ import print_function
import sys, select, termios, tty
import threading

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.real.real_env import AlgorithmMode, LearningType

# to surpress the warning when running in real env
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 


msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .
For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >
t : up (+z)
b : down (-z)
anything else : stop
q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
CTRL-C to quit
"""

moveBindings = {
        'i':(1,0,0,0),
        'o':(1,0,0,-1),
        'j':(0,0,0,1),
        'l':(0,0,0,-1),
        'u':(1,0,0,1),
        ',':(-1,0,0,0),
        '.':(-1,0,0,1),
        'm':(-1,0,0,-1),
        'O':(1,-1,0,0),
        'I':(1,0,0,0),
        'J':(0,1,0,0),
        'L':(0,-1,0,0),
        'U':(1,1,0,0),
        '<':(-1,0,0,0),
        '>':(-1,-1,0,0),
        'M':(-1,1,0,0),
        't':(0,0,1,0),
        'b':(0,0,-1,0),
    }

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
    }

class PublishThread(threading.Thread):
    def __init__(self, rate):
        super(PublishThread, self).__init__()
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever
        # for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()


    def update(self, x, y, z, th, speed, turn):
        self.condition.acquire()
        self.x = x
        self.y = y
        self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0)
        self.join()

    def run(self):
        self.condition.acquire()

        # Copy state into twist message.
        linear_x = self.x * self.speed
        linear_y = self.y * self.speed
        linear_z = self.z * self.speed
        angular_x = 0
        angular_y = 0
        angular_z = self.th * self.turn

        self.condition.release()

        return linear_x, angular_z

def getKey(key_timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed, turn):
    return "Modified to:\tspeed %s\tturn %s " % (speed,turn)


if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    # create the envronment and establish connection
    # env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='Manual control',
    #             resume_experiment=False, channel_type=Channels.DEPTH_ONLY,
    #             learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN)
    env = gym.make("OffWorldDockerMonolithContinuousSim-v0", channel_type=Channels.DEPTH_ONLY)

    speed, turn, rate, key_timeout  = 0.1, 1.0, 0.0, 0.1

    x, y, z, th, status = 0, 0, 0, 0, 0

    pub_thread = PublishThread(rate)

    try:
        pub_thread.update(x, y, z, th, speed, turn)
        print(msg)
        print(vels(speed,turn))
        
        env.reset()

        # send a command to the robot
        while True:
            done = False
            while not done:
                # get the keys, convert to linear_x and angular_z
                key = getKey(key_timeout)
                if key in moveBindings.keys():
                    x = moveBindings[key][0]
                    y = moveBindings[key][1]
                    z = moveBindings[key][2]
                    th = moveBindings[key][3]
                    pub_thread.update(x, y, z, th, speed, turn)
                    linear_x, angular_z = pub_thread.run()
                    action = [linear_x, angular_z]
                    state, reward, done, _ = env.step(action)

                    # print out action outcome
                    print("currently:\tspeed %s\tturn %s " % (linear_x, angular_z))
                    print("Step reward:", reward)
                    print("Episode has ended:", done, "\n")
                elif key in speedBindings.keys():
                    speed = speed * speedBindings[key][0]
                    turn = turn * speedBindings[key][1]
                    pub_thread.update(x, y, z, th, speed, turn)
                    print(vels(speed,turn))
                else:
                    if key == '' and x == 0 and y == 0 and z == 0 and th == 0:
                        continue
                    if (key == '\x03'):
                        exit()
                        
            env.reset()

    except Exception as e:
        print(e)

    finally:
        pub_thread.stop()

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)




    

