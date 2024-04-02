#! /usr/bin/env python3

import rospy
import cv2
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

        # Subscribe to image topic
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        # Publish to cmd_vel topic
        self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        # Subscribe to clock topic
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_callback)
        # Publish to score_tracker
        self.score_track_pub = rospy.Publisher("/score_tracker", String, queue_size=3)

        # Add a delay of 1 second before sending any messages
        rospy.sleep(1)
        # Set a rate to publish messages
        self.rate = rospy.Rate(100)  # 100 Hz

        # Create a bridge between ROS and OpenCV
        self.bridge = CvBridge()
        self.camera_image = None
        self.clue_board_detected = False
        self.clue_board_raw = None

        # read in the fizz gear for SIFT
        gear_path = "/home/fizzer/ros_ws/src/my_controller/launch/clue_board_top_left.png"
        self.gear_image = cv2.imread(gear_path)
        self.gear_grey = cv2.cvtColor(self.gear_image, cv2.COLOR_BGR2GRAY)

        # construct a SIFT object
        self.sift = cv2.SIFT_create()
        # detect the keypoint in the image,
        # with mask being None, so every part of the image is being searched
        self.keypoint = self.sift.detect(self.gear_grey, None)
        # print("the number of key points: ", len(keypoint))
        # cv2.imshow("name", self.gear_image)
        # cv2.waitKey(3)
        # draw the keypoint onto the image, show and save it
        # kp = cv2.drawKeypoints(self.gear_grey, self.keypoint, self.gear_grey, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("kp", kp)
        # cv2.waitKey(3)
        # cv2.imwrite('keypoints detected.jpg', kp)

        # calculate the descriptor for each key point
        self.kp_gear, self.des_gear = self.sift.compute(self.gear_grey, self.keypoint)

        # driving control parameters
        self.Kp = 0.02  # Proportional gain
        self.error_threshold = 20  # drive with different linear speed wrt this error_theshold
        self.linear_val_max = 0.25  # drive fast when error is small
        self.linear_val_min = 0.1  # drive slow when error is small
        self.mid_x = 0.0  # center of the frame initialized to be 0, updated at each find_middle function call

        self.timer = None
        # self.timer_not_inited = True
        self.start_not_sent = True
        self.end_not_sent = True

    def stop(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0
        twist_msg.angular.z = 0
        return twist_msg

    def drive_normal(self):
        return self.calculate_speed(self.camera_image)

    def image_callback(self, data):

        # process the scribed image from camera in openCV 
        # convert image message to openCV image 
        try:
            self.camera_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            cv2.imshow("Full camera view", self.camera_image)
            cv2.waitKey(3)
        except Exception as e:
            rospy.logerr(e)
            return

        self.clue_board_detection()
        if self.clue_board_detected:
            self.SIFT_image()

        if self.end_not_sent and not self.start_not_sent:
            # if end_not_sent is true and start_not_sent is false
            # Create Twist message and publish to cmd_vel
            twist_msg = self.drive_normal()
            self.cmd_vel_pub.publish(twist_msg)

        if not self.end_not_sent:
            # if end_not_sent is false
            # I stop driving
            # Create a Twist to stop the robot and publish to cmd_vel
            twist_msg = self.stop()
            self.cmd_vel_pub.publish(twist_msg)

    def calculate_speed(self, img):

        dim_x = img.shape[1]
        twist_msg = Twist()

        # detect_line calculates and modifies the target center
        self.find_middle(img)
        # angular error is the different between the center of the frame and target center
        angular_error = dim_x / 2 - self.mid_x
        twist_msg.angular.z = self.Kp * angular_error

        if abs(angular_error) <= self.error_threshold:
            twist_msg.linear.x = self.linear_val_max
        else:
            twist_msg.linear.x = self.linear_val_min

        return twist_msg

    def find_middle(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
        last_row = binary[-1, :]

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

        duration = 100
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

    def clue_board_detection(self):

        height, width, _ = self.camera_image.shape
        roi = self.camera_image[int(height/2.5):, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # cv2.imshow('Blue Square Detection', hsv)
        # cv2.waitKey(1)

        # Define range of blue color in HSV
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 220])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find square-like shapes
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 200 and h > 150 and 1.5 < w/h < 2.5:
                    print("width is: ", w)
                    print("height is: ", h)
                    img = cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.imshow('Blue Square Detection', img)
                    # cv2.waitKey(1)

                    self.clue_board_detected = True

                    # Display the result
                    self.clue_board_raw = roi[y:y + h, x:x + w]
                    cv2.imshow('Blue Square RAW', self.clue_board_raw)

    def SIFT_image(self):

        matching_points = 4

        clue_board_grey = cv2.cvtColor(self.clue_board_raw, cv2.COLOR_BGR2GRAY)
        kp_camera, des_camera = self.sift.detectAndCompute(clue_board_grey, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # return the best 2 matches
        matches = flann.knnMatch(self.des_gear, des_camera, k=2)

        # Need to draw only good matches, so create a mask
        homography_mask = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                homography_mask.append(m)

        # draw homography in the camera image
        query_pts = np.float32([self.kp_gear[m.queryIdx].pt for m in homography_mask]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_camera[m.trainIdx].pt for m in homography_mask]).reshape(-1, 1, 2)
        if len(query_pts) >= matching_points:
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

            # Perspective transform
            h, w = self.gear_image.shape[0], self.gear_image.shape[1]
            pts = np.float32([[0, 0], [0, 2*h], [2*w, 2*h], [2*w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography_img = cv2.polylines(self.clue_board_raw, [np.int32(dst)], True, (0, 0, 255), 4)
            cv2.imshow("Homography", homography_img)
            cv2.waitKey(1)

        self.clue_board_detected = False


def main():
    try:
        image_subscriber = Drive()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.detroyAllWindows()


if __name__ == '__main__':
    main()
