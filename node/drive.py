#! /usr/bin/env python3

import rospy
import cv2
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
import tensorflow as tf


class Drive:

    def __init__(self):

        # set a simulation time
        self.sim_time = 0
        self.driving_state = 0
        self.on_purple = 0

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
        self.blue_board_detected = False
        self.white_board_detected = False
        self.clue_board_detected = False
        self.blue_board = None
        self.white_board = None
        self.clue_detected = False
        # import the CNN text prediction model
        self.model = tf.keras.models.load_model("/home/fizzer/ros_ws/src/my_controller/node/character_prediction.h5")

        self.clue_type_dict = {
            'S': '1',
            'V': '2',
            'C': '3',
            'T': '4',
            'P': '5',
            'M': '6',
            'W': '7',
            'B': '8'
        }
        self.clue_type_str = None
        self.clue_type_id = None
        self.clue_value_str = None

        # driving control parameters
        self.Kp = 0.034  # Proportional gain
        self.error_threshold = 35  # drive with different linear speed wrt this error_theshold
        self.linear_val_max = 0.3  # drive fast when error is small
        self.linear_val_min = 0.1  # drive slow when error is small
        self.mid_x = 640  # center of the frame initialized to be 0, updated at each find_middle function call

        self.twist_msg = Twist()
        self.imgnum = 1483
        self.state_trans_start_time = 0

        self.timer = None
        self.start_not_sent = True
        self.end_not_sent = True

    def get_current_vel(self, twist):
        self.twist_msg.linear.x = round(twist.linear.x, 3)
        self.twist_msg.angular.z = round(twist.angular.z, 3)

    def stop(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0
        twist_msg.angular.z = 0
        return twist_msg

    def image_callback(self, data):

        # process the scribed image from camera in openCV 
        # convert image message to openCV image 
        try:
            self.camera_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            cv2.imshow("Full camera view", self.camera_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(e)
            return

        if not self.blue_board_detected and not self.white_board_detected:
            self.detect_blue_board()
        if self.blue_board_detected:
            self.detect_white_board()
        if self.white_board_detected:
            clue_type_x0, clue_type_y0 = 250, 37
            clue_value_x0, clue_value_y0 = 30, 260

            clue_type_prediction = self.parse_type(clue_type_x0, clue_type_y0)
            print("The predicted clue type letter is: ", clue_type_prediction)
            self.clue_type_str = ''.join(clue_type_prediction)
            self.clue_type_id = self.clue_type_dict.get(self.clue_type_str)

            clue_value_prediction = self.parse_value(clue_value_x0, clue_value_y0)
            print("clue_value_prediction: ", clue_value_prediction)
            self.clue_value_str = ''.join(clue_value_prediction)

            self.clue_detected = True
            self.blue_board_detected = False
            self.white_board_detected = False

        if self.end_not_sent and not self.start_not_sent:
            self.calculate_speed(self.camera_image)

            if not self.driving_state == 6:
                self.cmd_vel_pub.publish(self.twist_msg)

        if not self.end_not_sent:
            # if end_not_sent is false
            # I stop driving
            # Create a Twist to stop the robot and publish to cmd_vel
            # Create a Twist to stop the robot and publish to cmd_vel
            self.twist_msg.linear.x = 0
            self.twist_msg.angular.z = 0
            self.cmd_vel_pub.publish(self.twist_msg)

    def calculate_speed(self, img):

        dim_x = img.shape[1]
        dim_y = img.shape[0]

        hsv = cv2.cvtColor(img[717:719, :, :], cv2.COLOR_BGR2HSV)
        # print(dim_x)
        # print(dim_y)
        # image processing:
        # change the frame to grey scale
        # print("State: " + str(self.driving_state))
        if self.driving_state >= 0 and self.driving_state <= 4:

            # print("Img shape: " + str(img.shape))
            gray = cv2.cvtColor(img[717:719, :, :], cv2.COLOR_BGR2GRAY)

            if self.driving_state == 4:
                print("Time: " + str(int(time.time() - self.state_trans_start_time)))
                self.count_purples(hsv)

            else:
                redCount = self.count_reds(hsv)

                if self.driving_state == 0 and redCount > 500:
                    self.driving_state = 1
                    print("Detected red!: " + str(redCount))
                elif self.driving_state == 1 and redCount < 500:
                    self.driving_state = 2
                    print("Off the red!: " + str(redCount))
                elif self.driving_state == 2:
                    self.count_purples(hsv)
                    if redCount > 500:
                        self.driving_state = 3
                        print("Detected red again!: " + str(redCount))
                elif self.driving_state == 3 and redCount < 500:
                    self.driving_state = 4
                    self.state_trans_start_time = time.time()
                    print("Off the red again!: " + str(redCount))

            # not using gaussianBlur for now
            # blur it so the "road" far away become less easy to detect
            # kernel_size: Gaussian kernel size
            # sigma_x: Gaussian kernel standard deviation in X direction
            # sigma_y: Gaussian kernel standard deviation in Y direction
            # kernel_size = 13
            # sigma_x = 5
            # sigma_y = 5
            # blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x, sigma_y)  # gray scale the image
            # double_blur_gray = cv2.GaussianBlur(blur_gray[int(dim_y/3):int(3*dim_y/4), int(dim_x/5):int(4*dim_x/5)], (kernel_size, kernel_size), sigma_x, sigma_y)
            # Stopping when movement is detected
            # if self.past_image.size == 0:
            # self.past_image = double_blur_gray
            # else:
            #   subtracted_img = cv2.absdiff(double_blur_gray, self.past_image)
            # histogram = cv2.calcHist([subtracted_img], [0], None, [256], [0,256])
            # histogram /= histogram.sum()
            # cv2.imshow("camera view", subtracted_img)
            # cv2.waitKey(2)

            #  self.past_image = double_blur_gray
            # if subtracted_img.max() > 90:
            #   twist_msg.linear.x = 0
            #  twist_msg.angular.z = 0
            # return twist_msg

            # binary it
            # ret, binary = cv.threshold(blur_gray, 70, 255, cv.THRESH_BINARY)
            ret, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

            # cv2.imshow("camera view", binary)
            # cv2.waitKey(3)

            last_row = binary[-1, :]
            # print(last_row)
        elif self.driving_state == 5 or self.driving_state == 6 or self.driving_state == 7:
            filtered_img = cv2.cvtColor(cv2.medianBlur(img[630:719, :, :], 71), cv2.COLOR_BGR2HSV)
            filtered_img = cv2.cvtColor(cv2.GaussianBlur(filtered_img, (41, 41), 0), cv2.COLOR_BGR2HSV)
            # filtered_img = cv2.cvtColor(cv2.bilateralFilter(filtered_img, 13, 63, 47), cv2.COLOR_BGR2HSV)

            self.count_purples(hsv)
            print("filtered img: " + str(filtered_img))
            YELLOW_MAX = np.array([50, 245, 197], np.uint8)
            YELLOW_MIN = np.array([13, 195, 145], np.uint8)

            if self.driving_state == 7:
                YELLOW_MAX = np.array([50, 230, 193], np.uint8)
                YELLOW_MIN = np.array([13, 194, 120], np.uint8)

            # print("Y Max: " + str(YELLOW_MAX))
            # print("Y Min: " + str(YELLOW_MIN))

            frame_threshed = cv2.bitwise_not(cv2.inRange(filtered_img[87:89, :], YELLOW_MIN, YELLOW_MAX))
            for index in range(1, len(frame_threshed[1]) - 1):
                if frame_threshed[1][index - 1] == 0 and frame_threshed[1][index + 1] == 0:
                    frame_threshed[1][index] = 0
                elif frame_threshed[1][index - 1] == 1 and frame_threshed[1][index + 1] == 1:
                    frame_threshed[1][index] = 1

            last_row = frame_threshed[-1, :]

            # if self.driving_state == 7 and time.time() - self.state_trans_start_time < 5:
            # last_row = cv2.bitwise_and(last_row, 0)
            # contours, hierarchy = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # if len(contours) > 0:
            #     x_midpoint = dim_x / 2
            #     index = 0

            #     # If there is more than 1 contour, take the lowest one
            #     if len(contours) > 1:
            #         lowest_box = 0
            #         index = 0
            #         for i in range(len(contours)):
            #             rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contours[i])
            #             box_y = rect_y + rect_h
            #             if box_y > lowest_box:
            #                 lowest_box = box_y
            #                 index = i

            #     # Create a minimum area rectangle around the contour
            #     (rect_x, rect_y), (rect_w, rect_h), angle = cv2.minAreaRect(contours[index])

            #     # Calculate the horizontal distance from the center of the box to the center of the frame
            #     error = rect_x - x_midpoint
            #     move = Twist()
            #     move.angular.z = -1 * error / x_midpoint
            #     move.linear.x = 1 - (error / (2*x_midpoint))
            #     return move
            # else:
            #   return Twist()
            # print("Last row: " + str(last_row))
        # print("last row: " + str(last_row.shape))
        # elif self.driving_state == 6:

        #     PURPLE_MIN = np.array([135, 210, 210],np.uint8)
        #     PURPLE_MAX = np.array([165, 255, 255],np.uint8)
        #     purple_threshed = cv2.inRange(hsv, PURPLE_MIN, PURPLE_MAX)[:,:]
        #     purpleCount = 0
        #     for index in range(len(purple_threshed)):
        #         for index2 in range(len(purple_threshed[0])):
        #             if purple_threshed[index][index2] > 0:
        #                 purpleCount += 1

        #     if purpleCount > 400:
        #         self.driving_state = 7
        #         print("Detected purple!: " + str(purpleCount))

        if np.any(last_row == 0):
            last_list = last_row.tolist()
            first_index = last_list.index(0) if 0 in last_list else 0
            last_index = (len(last_list) - 1 - last_list[::-1].index(0)) if 0 in last_list else len(last_list) - 1
            new_mid = (first_index + last_index) / 2
            self.mid_x = new_mid

        # angular error is the different between the center of the frame and targer center
        angular_error = dim_x / 2 - self.mid_x
        self.twist_msg.angular.z = self.Kp * angular_error

        self.Kp = 0.034

        if abs(angular_error) <= self.error_threshold:
            self.twist_msg.linear.x = self.linear_val_max
        else:
            self.twist_msg.linear.x = self.linear_val_min

        if self.driving_state == 1:
            self.twist_msg.linear.x = self.linear_val_max - 0.05
        elif self.driving_state == 4:
            current_time = time.time()
            if int(current_time - self.state_trans_start_time) > 11:
                self.twist_msg.angular.z += 0.25
                if int(current_time - self.state_trans_start_time) > 13:
                    self.twist_msg.angular.z += 0.25
                    if int(current_time - self.state_trans_start_time) > 20:
                        self.twist_msg.angular.z -= 0.1
                        if int(current_time - self.state_trans_start_time) > 25:
                            self.twist_msg.angular.z += 0.35
        elif self.driving_state == 5:
            self.twist_msg.linear.x = self.linear_val_max - 0.25
        elif self.driving_state == 7:
            self.twist_msg.linear.x = self.linear_val_max - 0.2
            self.twist_msg.angular.z += 0.35
            self.Kp = 0.04
        # cropped_gray = cv2.resize(gray[int(dim_y/5):int(4*dim_y/5), int(dim_x/5):int(4*dim_x/5)], (0,0), fx=0.3, fy=0.3)
        # print("cropped shape: " + str(cropped_gray.shape))
        # self.pedestrian_SIFT(cropped_gray)

    def count_reds(self, img):
        RED_MIN = np.array([0, 235, 235], np.uint8)
        RED_MAX = np.array([25, 255, 255], np.uint8)

        red_threshed = cv2.inRange(img, RED_MIN, RED_MAX)

        redCount = 0

        for index in range(len(red_threshed)):
            for index2 in range(len(red_threshed[0])):
                if red_threshed[index][index2] > 0:
                    redCount += 1

        return redCount

    def count_purples(self, img):
        PURPLE_MIN = np.array([135, 210, 210], np.uint8)
        PURPLE_MAX = np.array([165, 255, 255], np.uint8)
        purple_threshed = cv2.inRange(img, PURPLE_MIN, PURPLE_MAX)
        purpleCount = 0
        for index in range(len(purple_threshed)):
            for index2 in range(len(purple_threshed[0])):
                if purple_threshed[index][index2] > 0:
                    purpleCount += 1

        if purpleCount > 390:
            self.on_purple = 1
            print("Detected purple!: " + str(purpleCount))
        elif self.on_purple == 1:
            if self.driving_state == 4 or self.driving_state == 5:
                # self.twist_msg.angular.z = 0
                # self.twist_msg.linear.x = 0.3
                # self.cmd_vel_pub.publish(self.twist_msg)
                self.driving_state += 1
                print("State: " + str(self.driving_state))
                self.on_purple = 0
                # time.sleep(0.7)
            elif self.driving_state == 6:
                self.state_trans_start_time = time.time()
                # self.twist_msg.angular.z = 0
                # self.twist_msg.linear.x = 0.3
                # self.cmd_vel_pub.publish(self.twist_msg)
                self.driving_state += 1
                # print("State: " + str(self.driving_state))
                self.on_purple = 0
                # time.sleep(0.7)
                # print("Would start stage 7")
                # self.spawn_position([-3.88, 0.476, 0.1], 0.0, 0.0, 3.14)
                # time.sleep(1.5)
                # self.spawn_position([-3.88, 0.476, 0.1], 0.0, 0.0, 3.14)
            elif self.driving_state == 2:
                self.driving_state = 5
                self.on_purple = 0
        else:
            self.on_purple = 0

    def spawn_position(self, position, roll, pitch, yaw):

        msg = ModelState()
        msg.model_name = 'R1'

        quat = quaternion_from_euler(roll, pitch, yaw)

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation = Quaternion(*quat)

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(msg)

        except rospy.ServiceException:
            print("Service call failed")

    def clock_callback(self, data):

        start_msg = 0
        stop_msg = -1
        string_message = '14,password,{0},NA'

        start_message = f"14,password,{start_msg},start"
        stop_message = f"14,password,{stop_msg},stop"
        reward_message = f"14,password,{self.clue_type_id}, {self.clue_value_str}"

        duration = 400
        sim_time = data.clock.secs

        if self.start_not_sent:
            print("I am going to send the first message to start! ")
            self.score_track_pub.publish(start_message)
            self.timer = sim_time
            self.start_not_sent = False

        if self.clue_detected:
            print(f"Sending clue type for {self.clue_value_str}")
            self.score_track_pub.publish(reward_message)
            self.clue_detected = False

        if sim_time >= self.timer + duration:
            if self.end_not_sent:
                print("I am going to stop the timer")
                self.score_track_pub.publish(stop_message)
                self.end_not_sent = False

    def separate_corners(self, approx_corners):

        # Reshape and find the centroid of four corners
        points = approx_corners.reshape(4, 2)
        centroid = np.mean(points, axis=0)

        # top as a subset of points that have a y-coordinate less than the y-coordinate of the centroid
        top = points[np.where(points[:, 1] < centroid[1])]

        # bottom as a subset of points that have a y-coord greater than the y-coordinate of the centroid
        bottom = points[np.where(points[:, 1] >= centroid[1])]

        # find the min x in top two points as top_left
        top_left = top[np.argmin(top[:, 0])]
        # find the max x in top two points as top_right
        top_right = top[np.argmax(top[:, 0])]
        # find the min x in bottom two points as bottom_left
        bottom_left = bottom[np.argmin(bottom[:, 0])]
        # find the max x in bottom two points as bottom_right
        bottom_right = bottom[np.argmax(bottom[:, 0])]

        return [top_left, top_right, bottom_right, bottom_left]

    def detect_blue_board(self):

        # only processing the lower portion of the camera view
        height, width, _ = self.camera_image.shape
        roi = self.camera_image[int(height / 2.5):, :]

        # change it to HSV format for better blue detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # cv2.imshow('Blue Square Detection', hsv)
        # cv2.waitKey(1)

        # Define range of blue color in HSV and threshold the HSV image to get only blue colors
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 220])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours in the masked image
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find square-like shapes
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx_corners = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx_corners) == 4:
                x, y, w, h = cv2.boundingRect(contour)

                # I am only interested in the clue board when I am close by,
                # and it is a clue board only when x dim > y dim
                if w > 225 and h > 125 and 1.0 < w / h < 2.0:
                    # # save the raw clue board
                    # clue_board_raw = roi[y:y + h, x:x + w]

                    # Properly separate four corners and
                    # perform perspective transform this blue board into a rectangle shape
                    src_pts = np.float32(self.separate_corners(approx_corners))
                    width, height = 780, 460
                    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

                    # Compute the perspective transform matrix and apply the perspective transformation
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    self.blue_board = cv2.warpPerspective(roi, matrix, (width, height))
                    # cv2.imshow('A Blue Board', self.blue_board)
                    # cv2.waitKey(1)

                    self.blue_board_detected = True

    def detect_white_board(self):

        # change it to HSV format for better blue detection
        hsv = cv2.cvtColor(self.blue_board, cv2.COLOR_BGR2HSV)

        # Define range of blue color in HSV and threshold the HSV image to get only blue colors
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 220])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        inverse_mask = cv2.bitwise_not(blue_mask)
        contours, _ = cv2.findContours(inverse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx_corners = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx_corners) == 4:
                x, y, w, h = cv2.boundingRect(contour)

                # I am only interested in the clue board when I am close by,
                # and it is a clue board only when x dim > y dim
                if w > 500 and h > 300 and 1.0 < w / h < 2.0:
                    # print("This is a white board, with weight and height:", w, h)

                    # Properly separate four corners and
                    # perform perspective transform this white board into a rectangle shape
                    src_pts = np.float32(self.separate_corners(approx_corners))
                    width, height = 600, 400
                    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

                    # Compute the perspective transform matrix and apply the perspective transformation
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    self.white_board = cv2.warpPerspective(self.blue_board, matrix, (width, height))
                    # cv2.imshow('A white Board', self.white_board)
                    # cv2.waitKey(1)

                    self.white_board_detected = True

    def char_image_format(self, bgr_image):

        target_brightness = 2

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        delta_brightness = target_brightness - brightness
        adjusted = cv2.add(gray, np.full_like(gray, delta_brightness, dtype=np.uint8))

        threshold = 240
        _, binary = cv2.threshold(adjusted, threshold, 255, cv2.THRESH_BINARY)

        return binary

    def parse_type(self, type_x0, type_y0):

        clue_type_x0 = type_x0
        clue_type_y0 = type_y0
        letter_width = 45
        letter_height = 80

        clue_type_img_raw = self.white_board[clue_type_y0:clue_type_y0 + letter_height,
                            clue_type_x0:clue_type_x0 + letter_width]
        clue_type_img = self.char_image_format(clue_type_img_raw)
        img_des = f"/home/fizzer/PycharmProjects/character_cnn/image/my_controller_image/raw/{self.raw}.jpg"

        clue_type_predict = self.predict_clue([clue_type_img])

        return clue_type_predict

    def parse_value(self, value_x0, value_y0):

        clue_value_x0 = value_x0
        clue_value_y0 = value_y0
        letter_width = 45
        letter_height = 80

        clue_value_img_list = []
        for i in range(12):
            x_start = clue_value_x0 + i * letter_width
            y_start = clue_value_y0

            clue_char_img_raw = self.white_board[y_start:y_start + letter_height, x_start:x_start + letter_width]
            clue_char_img = self.char_image_format(clue_char_img_raw)
            clue_value_img_list.append(clue_char_img)
            img_des = f"/home/fizzer/PycharmProjects/character_cnn/image/my_controller_image/raw/{self.raw}.jpg"

        clue_value_predict = self.predict_clue(clue_value_img_list)

        return clue_value_predict

    # map the int to char by their unicode code representation
    # A - Z as 0 - 25 and 0 - 9 as 26 - 35
    def int_to_char(self, my_int):
        if 0 <= my_int <= 25:
            return chr(my_int + ord('A'))
        elif 26 <= my_int <= 35:
            return chr(my_int - 26 + ord('0'))
        else:
            raise ValueError(f"Invalid character: {my_int}")

    def predict_clue(self, images):
        y_predicts = list()
        for img in images:
            try:
                min_pixel_value = img.min()
                if min_pixel_value == 0:
                    # img_reshape = cv2.resize(img, (100, 140), interpolation=cv2.INTER_LINEAR)
                    # img_aug = np.expand_dims(img_reshape / 255.0, axis=0)
                    img_reshape = cv2.resize(img, (60, 120), interpolation=cv2.INTER_LINEAR)
                    img_aug = np.expand_dims(img_reshape, axis=0)
                    y_p = self.model.predict(img_aug)[0]
                    y_predicts.append(self.int_to_char(np.argmax(y_p)))
                else:
                    # Append empty string to indicate failure
                    y_predicts.append(' ')
            except cv2.error as e:
                # Append empty string to indicate failure
                y_predicts.append(' ')

        return y_predicts


def main():
    try:
        image_subscriber = Drive()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.detroyAllWindows()


if __name__ == '__main__':
    main()
