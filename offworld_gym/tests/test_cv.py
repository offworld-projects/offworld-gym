import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import scipy
import rospy
from sensor_msgs.msg import Image
from offworld_gym.envs.gazebo.utils import ImageUtils

IMG_H = 240
IMG_W = 320

def test():
    rospy.init_node('test_cv')
    dimage_msg = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
    dimage_data = ImageUtils.process_depth_msg(dimage_msg)
    assert dimage_data.shape[1] == IMG_H
    assert dimage_data.shape[2] == IMG_W


    rgbimage_msg = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
    rgbimage_data = ImageUtils.process_img_msg(rgbimage_msg)
    assert rgbimage_data.shape[1] == IMG_H
    assert rgbimage_data.shape[2] == IMG_W

if __name__ == "__main__":
    test()
