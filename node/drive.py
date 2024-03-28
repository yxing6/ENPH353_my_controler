#! /usr/bin/env python3

import rospy
import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock


class Drive:

    def __init__(self):

        # set a simulation time
        self.sim_time = 0

        # Initialize a ros node
        rospy.init_node('image_subscriber_node', anonymous=True)

        # Subscribe to the image topic and clock
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_callback)
        # Create a publisher for cmd_vel and score_tracker
        self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.score_track_pub = rospy.Publisher("/score_tracker", String, queue_size=3)

        # Add a delay of 1 second before sending any messages
        rospy.sleep(1)
        # Set a rate to publish messages
        self.rate = rospy.Rate(100)  # 100 Hz

        # Create a bridge between ROS and OpenCV
        self.bridge = CvBridge()
        # driving control parameters
        self.Kp = 0.02  # Proportional gain
        self.error_threshold = 20  # drive with different linear speed wrt this error_theshold
        self.linear_val_max = 0.4  # drive fast when error is small
        self.linear_val_min = 0.1  # drive slow when error is small
        self.mid_x = 0.0  # center of the frame initialized to be 0, updated at each find_middle function call

        self.timer = None
        self.timer_not_inited = True
        self.start_not_sent = True
        self.end_not_sent = True

    def image_callback(self, data):

        # process the scribed image from camera in openCV 
        # convert image message to openCV image 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            rospy.logerr(e)
            return

        # Create Twist message and publish to cmd_vel
        twist_msg = Twist()
        speed = self.calculate_speed(cv_image)
        # print("speed: ", speed)
        twist_msg.linear.x = speed[0]
        twist_msg.angular.z = speed[1]
        self.cmd_vel_pub.publish(twist_msg)

    def calculate_speed(self, img):

        dim_x = img.shape[1]

        # detect_line calculates and modifies the target center
        self.find_middle(img)
        # angular error is the different between the center of the frame and targer center
        angular_error = dim_x / 2 - self.mid_x
        angular_vel = self.Kp * angular_error

        if abs(angular_error) <= self.error_threshold:
            linear_vel = self.linear_val_max
        else:
            linear_vel = self.linear_val_min

        return linear_vel, angular_vel

    def find_middle(self, img):

        # image processing: 
        # change the frame to grey scale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # not using gaussianBlur for now
        # blur it so the "road" far away become less easy to detect
        # kernel_size: Gaussian kernel size
        # sigma_x: Gaussian kernel standard deviation in X direction
        # sigma_y: Gaussian kernel standard deviation in Y direction
        kernel_size = 13
        sigma_x = 5
        sigma_y = 5
        blur_gray = cv.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x, sigma_y)  # gray scale the image

        # binary it
        # ret, binary = cv.threshold(blur_gray, 70, 255, cv.THRESH_BINARY)
        ret, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)

        cv.imshow("camera view", binary)
        cv.waitKey(3)

        last_row = binary[-1, :]
        # print(last_row)

        if np.any(last_row == 0):
            last_list = last_row.tolist()
            first_index = last_list.index(0)
            last_index = len(last_list) - 1 - last_list[::-1].index(0)
            new_mid = (first_index + last_index) / 2
            self.mid_x = new_mid

    def clock_callback(self, data):

        start_msg = 0
        stop_msg = -1
        string_message = '14,password,{0},NA'

        start_message = string_message.format(start_msg)
        stop_message = string_message.format(stop_msg)

        duration = 10
        sim_time = data.clock.secs

        if self.start_not_sent:
            print("I am going to send the first message to start! ")
            self.score_track_pub.publish(start_message)
            self.timer = sim_time
            self.start_not_sent = False

        if sim_time >= self.timer + duration:
            if self.end_not_sent:
                print("I am going to stop the timer")
                self.score_track_pub.publish(stop_message)
                self.end_not_sent = False


def main():
    try:
        image_subscriber = Drive()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
