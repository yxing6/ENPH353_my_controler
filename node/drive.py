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
        self.clue_board_reshaped = None

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
        self.linear_val_max = 0.2  # drive fast when error is small
        self.linear_val_min = 0.05  # drive slow when error is small
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

    def drive_slower(self):
        slow_twist = self.calculate_speed(self.camera_image)
        slow_twist.linear.x = slow_twist.linear.x / 2
        return slow_twist

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
            # self.reshape_image()

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
            approx_corners = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx_corners) == 4:
                x, y, w, h = cv2.boundingRect(contour)

                if w > 150 and h > 100 and 1.5 < w/h < 2.5:

                    # Reshape and find the centroid of them
                    points = approx_corners.reshape(4, 2)
                    centroid = np.mean(points, axis=0)

                    # top as a subset of points that have a y-coordinate less than the y-coordinate of the centroid
                    top = points[np.where(points[:, 1] < centroid[1])]
                    bottom = points[np.where(points[:, 1] >= centroid[1])]
                    top_left = top[np.argmin(top[:, 0])]  # find the min x in top two points as top_left
                    top_right = top[np.argmax(top[:, 0])]  # find the max x in top two points as top_right
                    bottom_left = bottom[np.argmin(bottom[:, 0])]  # find the min x in bottom two points as bottom_left
                    bottom_right = bottom[np.argmax(bottom[:, 0])]  # find the max x in bottom two points as bottom_right

                    # top_left = (x, y)
                    # top_right = (x + w, y)
                    # bottom_left = (x, y + h)
                    # bottom_right = (x + w, y + h)
                    #
                    radius = 5  # Radius of the circles
                    thickness = 4  # Thickness of the circle outline
                    color = (0, 255, 0)  # Color of the circles (in BGR format)

                    # # Draw circles at each corner point
                    # cv2.circle(roi, top_left, radius, color, thickness)
                    # cv2.circle(roi, top_right, radius, color, thickness)
                    # cv2.circle(roi, bottom_left, radius, color, thickness)
                    # cv2.circle(roi, bottom_right, radius, color, thickness)

                    # print("width is: ", w)
                    # print("height is: ", h)
                    # img = cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.imshow('rectangle', img)
                    # cv2.waitKey(1)
                    # top_left = approx_corners[0]
                    # top_right = approx_corners[1]
                    # bottom_right = approx_corners[2]
                    # bottom_left = approx_corners[3]

                    self.clue_board_raw = roi[y:y + h, x:x + w]
                    cv2.imshow('Clue board RAW', self.clue_board_raw)
                    cv2.waitKey(1)

                    # First perspective transform
                    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
                    width, height = 600, 400
                    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                    # Compute the perspective transform matrix
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                    # Apply the perspective transformation
                    warped_image = cv2.warpPerspective(roi, matrix, (width, height))
                    cv2.imshow('Clue board warped', warped_image)
                    cv2.waitKey(1)

                    # Apply Harris Corner Detector
                    block_size = 5
                    aperture_size = 5
                    k = 0.1
                    # [:, :, 2] to select all rows, columns, and the elements in the third dimension (the blue channel).
                    harris_corners = cv2.cornerHarris(warped_image[:, :, 2], block_size, aperture_size, k)

                    # optimal value as a mask to be used identify strong corners
                    good_corner_mask = 0.01 * harris_corners.max()

                    # Threshold the Harris corner response values to identify strong corners
                    corners_mask = np.zeros_like(harris_corners, dtype=np.uint8)
                    corners_mask[harris_corners > good_corner_mask] = 255  # Set strong corners to white (255)

                    # Now, you can use this corners_mask to mask out the corners
                    masked_warped_image = cv2.bitwise_and(warped_image, warped_image, mask=corners_mask)

                    cv2.imshow('masked', masked_warped_image)
                    cv2.waitKey(1)

                    # Find contours in the corners mask
                    contours, _ = cv2.findContours(corners_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Initialize lists to store corner points
                    corner_points = []

                    # Extract corner points from contours
                    for c in contours:
                        # Find bounding rectangle of the contour
                        x, y, w, h = cv2.boundingRect(c)
                        # Add corners of the rectangle to corner_points list
                        corner_points.extend([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])

                    # Convert corner_points list to NumPy array
                    corner_points = np.array(corner_points)
                    # Find the extreme points to determine the four corner points
                    top_left = np.min(corner_points, axis=0)
                    top_right = [np.max(corner_points[:, 0]), np.min(corner_points[:, 1])]
                    bottom_left = [np.min(corner_points[:, 0]), np.max(corner_points[:, 1])]
                    bottom_right = np.max(corner_points, axis=0)

                    # Output the four corner points
                    print("Top Left:", top_left)
                    print("Top Right:", top_right)
                    print("Bottom Left:", bottom_left)
                    print("Bottom Right:", bottom_right)
                    cv2.circle(warped_image, top_left, radius, color, thickness)
                    cv2.circle(warped_image, top_right, radius, color, thickness)
                    cv2.circle(warped_image, bottom_left, radius, color, thickness)
                    cv2.circle(warped_image, bottom_right, radius, color, thickness)
                    cv2.imshow('corner detected', warped_image)
                    cv2.waitKey(1)

                    # perspective transform the clue_board
                    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
                    width, height = 600, 400
                    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    self.clue_board_reshaped = cv2.warpPerspective(warped_image, matrix, (width, height))

                    # for x, y in top_left, top_right, bottom_right, bottom_left:
                    #     # print((x, y))
                    #     cv2.circle(self.clue_board_reshaped, (x, y), 10, (255, 0, 0), -1)
                    cv2.imshow('Message reshaped', self.clue_board_reshaped)
                    cv2.waitKey(1)

                    # self.clue_board_detected = True

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
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography_img = cv2.polylines(self.clue_board_raw, [np.int32(dst)], True, (0, 0, 255), 4)
            cv2.imshow("Homography", homography_img)
            cv2.waitKey(1)

        self.clue_board_detected = False

    def reshape_image(self):

        gray = cv2.cvtColor(self.clue_board_raw, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the contours based on their area to get the largest contour
        # largest_contour = max(contours, key=cv2.contourArea)

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 200 and h > 150 and 1.5 < w/h < 2.5:
                    print("width is: ", w)
                    print("height is: ", h)
                    img = cv2.rectangle(self.clue_board_raw, (x, y), (x + w, y + h), (0, 0, 255), 4)
                    cv2.imshow('find contour of clue board', img)
        # Find the bounding box of the largest contour
        # x, y, w, h = cv2.boundingRect(largest_contour)

        # # Extract the four corners of the bounding box
        # top_left = (x, y)
        # top_right = (x + w, y)
        # bottom_left = (x, y + h)
        # bottom_right = (x + w, y + h)
        #
        # # Draw circles at the corners (optional)
        # cv2.circle(self.clue_board_raw, top_left, 5, (0, 0, 255), -1)
        # cv2.circle(self.clue_board_raw, top_right, 5, (0, 0, 255), -1)
        # cv2.circle(self.clue_board_raw, bottom_left, 5, (0, 0, 255), -1)
        # cv2.circle(self.clue_board_raw, bottom_right, 5, (0, 0, 255), -1)

        # cv2.imshow("find contour of clue board", self.clue_board_raw)
        # cv2.waitKey(1)

        self.clue_board_detected = False


def main():
    try:
        image_subscriber = Drive()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.detroyAllWindows()


if __name__ == '__main__':
    main()
