#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Drive:

    def __init__(self):

        # Initialize a ros node
        rospy.init_node('image_subscriber_node', anonymous=True)

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)

        # Create a publisher for cmd_vel/
        self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)

        # Create a bridge between ROS and OpenCV
        self.bridge = CvBridge()

    def image_callback(self, data):

        # process the scribed image from camera in openCV 
        # convert image message to openCV image 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            rospy.logerr(e)
            return

        # Create Twist message and publish to cmd_vel
        twist_msg = self.tape_follow(cv_image)
        print("speed: ", twist_msg.linear.x)
        print("angular: ", twist_msg.angular.z)
        self.cmd_vel_pub.publish(twist_msg)

    def tape_follow(self, img):

        cropped_img = img[int(img.shape[0]/2):int(img.shape[0])]
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        x_len = len(cropped_img[0])
        y_len = len(cropped_img)

        # Define the kernel size for erosion and dilation
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply a black and white threshold
        (thresh, black_white_frame) = cv2.threshold(gray_frame, 83, 255, cv2.THRESH_BINARY)

        # Apply dilation
        dilation = cv2.dilate(black_white_frame, kernel, iterations=1)

        # Apply erosion
        erosion = cv2.erode(dilation, kernel, iterations=3)

        # Invert colours to correct findContours
        inverted_erosion = cv2.bitwise_not(erosion)

        left_boundary = int(x_len/2)
        right_boundary = int(x_len/2)

        for pixel_index in range(x_len):
            if inverted_erosion[y_len-1, pixel_index] > 0:
                left_boundary = pixel_index
                break
        
        for pixel_index in reversed(range(x_len)):
            if inverted_erosion[y_len-1, pixel_index] > 0:
                right_boundary = pixel_index
                break
        
        mid_of_boundaries = int((left_boundary+right_boundary)/2)

        contours, hierarchy = cv2.findContours(inverted_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            x_midpoint = int(len(inverted_erosion[0])/2)
            y_midpoint = int(len(inverted_erosion)/2)
            index = 0

            # If there is more than 1 contour, take the lowest one
            if len(contours) > 1:
                lowest_box = 0
                index = 0
                for i in range(len(contours)):
                    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contours[i])
                    box_y = rect_y + rect_h
                    if box_y > lowest_box:
                        lowest_box = box_y
                        index = i

            # Create a minimum area rectangle around the contour
            (rect_x, rect_y), (rect_w, rect_h), angle = cv2.minAreaRect(contours[index])

            # Calculate the horizontal distance from the center of the box to the center of the frame
            #averaged_x_midpoint = (mid_of_boundaries + rect_x) / 2
            averaged_x_midpoint = mid_of_boundaries
            error = averaged_x_midpoint - x_midpoint
            move = Twist()
            move.linear.x = 1 - (rect_y / (y_midpoint * 2))
            move.angular.z = -1 * error / x_midpoint

            gray_frame = cv2.circle(gray_frame, (int(rect_x), int(rect_y)), 16, 60, -1)
            gray_frame = cv2.circle(gray_frame, (int(mid_of_boundaries), int(y_midpoint)), 16, 20, -1)
            cv2.imshow("image", gray_frame)
            cv2.waitKey(3)

            return move
        else:
            return Twist()


def main():
    try:
        image_subscriber = Drive()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
