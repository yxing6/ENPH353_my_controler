#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
import tensorflow as tf


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
        self.blue_board_detected = False
        self.white_board_detected = False
        self.clue_board_detected = False
        self.blue_board = None
        self.white_board = None
        self.clue_board = None

        # read in the fizz gear for SIFT
        gear_path = "/home/fizzer/ros_ws/src/my_controller/launch/clue_board_top_left.png"
        self.gear_image = cv2.imread(gear_path)
        self.gear_grey = cv2.cvtColor(self.gear_image, cv2.COLOR_BGR2GRAY)
        # construct a SIFT object
        self.sift = cv2.SIFT_create()
        # detect the keypoint in the image,
        # with mask being None, so every part of the image is being searched
        self.keypoint = self.sift.detect(self.gear_grey, None)
        # print("the number of key points: ", len(self.keypoint))
        # cv2.imshow("name", self.gear_image)
        # cv2.waitKey(3)
        # # draw the keypoint onto the image, show and save it
        # kp = cv2.drawKeypoints(self.gear_grey, self.keypoint, self.gear_grey, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("kp", kp)
        # cv2.waitKey(3)
        # cv2.imwrite('keypoints detected.jpg', kp)

        # calculate the descriptor for each key point
        self.kp_gear, self.des_gear = self.sift.compute(self.gear_grey, self.keypoint)

        # self.clue_type_dict = {
        #     'S': '1',
        #     'V': '2',
        #     'C': '3',
        #     'T': '4',
        #     'P': '5',
        #     'M': '6',
        #     'W': '7',
        #     'B': '8'
        # }
        # self.initial_clue_type = 'A'
        # self.clue_id = 0
        # self.clue_value_predict = 'TWO EXT'
        self.looking_for_board = 1
        self.clue_type_list = ['0', 'S', 'V', 'C', 'T', 'P', 'M', 'W', 'B', '-1']

        # import the CNN text prediction model
        self.model = tf.keras.models.load_model("/home/fizzer/ros_ws/src/my_controller/node/character_prediction.h5")

        # driving control parameters
        self.Kp = 0.02  # Proportional gain
        self.error_threshold = 20  # drive with different linear speed wrt this error_theshold
        self.linear_val_max = 0.2  # drive fast when error is small
        self.linear_val_min = 0.05  # drive slow when error is small
        self.mid_x = 0.0  # center of the frame initialized to be 0, updated at each find_middle function call

        self.timer = None
        self.start_not_sent = True
        self.end_not_sent = True

    def stop(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0
        twist_msg.angular.z = 0
        return twist_msg

    def drive_normal(self):
        return self.calculate_speed(self.camera_image)

    def drive_slower(self):
        slow_twist = self.calculate_speed(self.camera_image)
        slow_twist.linear.x = slow_twist.linear.x / 2
        return slow_twist

    def drive_straight(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.4
        twist_msg.angular.z = 0
        return twist_msg

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

        if not self.blue_board_detected and not self.clue_board_detected:
            self.detect_blue_board()
        if self.blue_board_detected:
            self.detect_white_board()
        if self.white_board_detected:
            clue_type_x0, clue_type_y0 = 250, 37
            clue_value_x0, clue_value_y0 = 30, 260

            clue_type_prediction = self.parse_type(clue_type_x0, clue_type_y0)
            print("The predicted clue type letter is: ", clue_type_prediction)
            if clue_type_prediction[0] in self.clue_type_list:
                print("clue_type_list.index(clue_type_prediction[0]): ", self.clue_type_list.index(clue_type_prediction[0]))
                print("looking for number:", self.looking_for_board)
                print("and they are equal:", self.clue_type_list.index(clue_type_prediction[0]) != self.looking_for_board)
                do_while_3_times = 0
                while do_while_3_times < 3 and self.clue_type_list.index(clue_type_prediction[0]) != self.looking_for_board:
                    clue_type_prediction = self.parse_type(clue_type_x0, clue_type_y0)
                    do_while_3_times += 1

                clue_value_prediction = self.parse_value(clue_value_x0, clue_value_y0)

                print("clue_type_prediction: ", clue_type_prediction)
                print("clue_value_prediction: ", clue_value_prediction)

            self.looking_for_board += 1
            self.blue_board_detected = False
            self.white_board_detected = False

        #     value = self.clue_type_dict.get(self.initial_clue_type)
        #     while value == 'Key not found':
        #         dst = self.SIFT_image()
        #         if dst is not None:
        #             clue_type_predict = self.parse_type(dst)
        #             print("the clue_type prediction is :", clue_type_predict)
        #             id_predict = self.clue_type_dict.get(clue_type_predict[0])
        #             while id_predict != 'Key not found':
        #                 self.initial_clue_type = clue_type_predict
        #                 value_predict = self.parse_value(dst)
        #                 print("the clue_value prediction string is :", self.clue_value_predict)
        #
        #                 self.clue_id = id_predict
        #                 self.clue_value_predict = value_predict
        #
        # self.blue_board_detected = False
        # self.white_board_detected = False
        # self.clue_board_detected = False
        # self.initial_clue_type = 'A'

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

        duration = 120
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
                if w > 250 and h > 150 and 1.0 < w / h < 2.0:

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
                    cv2.imshow('A Blue Board', self.blue_board)
                    cv2.waitKey(1)

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
                    cv2.imshow('A white Board', self.white_board)
                    cv2.waitKey(1)

                    self.white_board_detected = True

    def SIFT_image(self):

        matching_points = 4

        clue_board_grey = cv2.cvtColor(self.white_board, cv2.COLOR_BGR2GRAY)
        kp_camera, des_camera = self.sift.detectAndCompute(clue_board_grey, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # return the best 2 matches
        if des_camera is not None and len(des_camera) > 2:
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
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                # 00 for top left
                # 01 for bottom left
                # 11 for bottom right
                # 10 for top right
                x_00, y_00 = int(dst[0][0][0]), int(dst[0][0][1])
                x_10, y_10 = int(dst[3][0][0]), int(dst[3][0][1])

                # print("x_10 is:", x_10, "x_00 is", x_00, "and diff:", x_10 - x_00)

                if x_10 - x_00 > 100:
                    homography_img = cv2.polylines(self.white_board, [np.int32(dst)], True, (0, 0, 255), 4)
                    # cv2.imshow("Homography", homography_img)
                    # cv2.waitKey(1)

                    self.clue_board_detected = True
                    return dst

    def parse_type(self, type_x0, type_y0):

        # 00 for top left
        # 01 for bottom left
        # 11 for bottom right
        # 10 for top right
        # x_10, y_10 = int(dst[3][0][0]), int(dst[3][0][1])
        # print(x_10, y_10)

        clue_type_x0 = type_x0
        clue_type_y0 = type_y0
        letter_width = 45
        letter_height = 80

        clue_type_top_left = (clue_type_x0, clue_type_y0)
        clue_type_bottom_right = (clue_type_x0 + letter_width, clue_type_y0 + letter_height)

        color = (0, 255, 0)  # Green color in BGR format
        thickness = 2
        cv2.rectangle(self.white_board, clue_type_top_left, clue_type_bottom_right, color, thickness)
        cv2.imshow("Detect type", self.white_board)
        cv2.waitKey(1)

        clue_type_img = self.white_board[clue_type_y0:clue_type_y0 + letter_height,
                        clue_type_x0:clue_type_x0 + letter_width]

        clue_type_predict = self.predict_clue([clue_type_img])

        return clue_type_predict

    def parse_value(self, value_x0, value_y0):

        # 00 for top left
        # 01 for bottom left
        # 11 for bottom right
        # 10 for top right
        # x_01, y_01 = 0, 0
        # print(x_10, y_10)

        clue_value_x0 = value_x0
        clue_value_y0 = value_y0
        letter_width = 45
        letter_height = 80

        # clue_value_top_left = (clue_value_x0, clue_value_y0)
        # clue_value_bottom_right = (clue_value_x0 + letter_width, clue_value_y0 + letter_height)

        color = (255, 0, 0)
        thickness = 2
        # cv2.rectangle(self.clue_board_reshaped, clue_value_top_left, clue_value_bottom_right, color, thickness)
        # cv2.imshow("Detect value", self.clue_board_reshaped)

        clue_value_img_list = []
        for i in range(12):
            x_start = clue_value_x0 + i * letter_width
            y_start = clue_value_y0
            clue_char_top_left = (x_start, y_start)
            clue_char_bottom_right = (x_start + letter_width, y_start + letter_height)
            cv2.rectangle(self.white_board, clue_char_top_left, clue_char_bottom_right, color, thickness)
            clue_char_img = self.white_board[y_start:y_start + letter_height, x_start:x_start + letter_width]
            clue_value_img_list.append(clue_char_img)

        cv2.imshow("Detect value", self.white_board)
        cv2.waitKey(1)

        clue_value_predict = self.predict_clue(clue_value_img_list)

        return clue_value_predict

    # map the character to int by their unicode code representation
    # A - Z as 0 - 25 and 0 - 9 as 26 - 35
    def char_to_int(self, my_char):
        if 'A' <= my_char <= 'Z':
            return ord(my_char) - ord('A')
        elif '0' <= my_char <= '9':
            return 26 + ord(my_char) - ord('0')
        else:
            raise ValueError(f"Invalid character: {my_char}")

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
                img_reshape = cv2.resize(img, (100, 140), interpolation=cv2.INTER_LINEAR)
                img_aug = np.expand_dims(img_reshape / 255.0, axis=0)
                y_p = self.model.predict(img_aug)[0]
                y_predicts.append(self.int_to_char(np.argmax(y_p)))
            except cv2.error as e:
                # print("OpenCV error:", e)
                y_predicts.append(' ')  # Append empty string to indicate failure

        # print("the prediction is:", y_predicts)
        return y_predicts


def main():
    try:
        image_subscriber = Drive()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.detroyAllWindows()


if __name__ == '__main__':
    main()
