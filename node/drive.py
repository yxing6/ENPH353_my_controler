#! /usr/bin/env python3

import rospy
import cv2
import csv
import tensorflow as tf
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


class Drive:

    def __init__(self):

        # set a simulation time
        self.sim_time = 0
        self.pedestrian1_path = '/home/fizzer/Pictures/pedestrian1.png'
        self.pedestrian2_path = '/home/fizzer/Pictures/pedestrian4.png'
        ped_sift1 = cv2.SIFT_create()
        ped_sift2 = cv2.SIFT_create()
        self.driving_state = 0
        self.on_purple = 0

        self.model1 = tf.keras.models.load_model(r"/home/fizzer/Pictures/imitationmodel.keras")
        self.interpreter = tf.lite.Interpreter(model_path="/home/fizzer/Pictures/model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.pedestrian1_image_colored = cv2.resize(cv2.imread(self.pedestrian1_path), (120, 240))
        self.pedestrian1_image = cv2.cvtColor(self.pedestrian1_image_colored, cv2.COLOR_BGR2GRAY)
        self.pedestrian1_image_kp, self.pedestrian1_image_desc = ped_sift1.detectAndCompute(self.pedestrian1_image, None)

        self.pedestrian2_image_colored = cv2.resize(cv2.imread(self.pedestrian2_path), (120, 240))
        self.pedestrian2_image = cv2.cvtColor(self.pedestrian2_image_colored, cv2.COLOR_BGR2GRAY)
        self.pedestrian2_image_kp, self.pedestrian2_image_desc = ped_sift2.detectAndCompute(self.pedestrian2_image, None)

        # Initialize a ros node
        rospy.init_node('image_subscriber_node', anonymous=True)

        # Subscribe to image topic
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        # Publish to cmd_vel topic
        self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        # Subscribe to cmd_vel topic
        #self.cmd_vel_sub = rospy.Subscriber('/R1/cmd_vel', Twist, self.get_current_vel)
        # Subscribe to odom topic
        #self.odom_sub = rospy.Subscriber("/R1/odom", Odometry, self.odom_callback)
        # Subscribe to clock topic
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_callback)
        # Publish to score_tracker
        self.score_track_pub = rospy.Publisher("/score_tracker", String, queue_size=3)

        # Add a delay of 1 second before sending any messages
        rospy.sleep(3)
        # Set a rate to publish messages
        self.rate = rospy.Rate(100)  # 100 Hz

        # Create a bridge between ROS and OpenCV
        self.bridge = CvBridge()

        # driving control parameters
        self.Kp = 0.034  # Proportional gain
        self.error_threshold = 35  # drive with different linear speed wrt this error_theshold
        self.linear_val_max = 0.35  # drive fast when error is small
        self.linear_val_min = 0.1  # drive slow when error is small
        self.mid_x = 640  # center of the frame initialized to be 0, updated at each find_middle function call

        self.twist_msg = Twist()
        self.imgnum = 1483
        self.state_trans_start_time = 0

        self.timer = None
        self.timer_not_inited = True
        self.start_not_sent = True
        self.end_not_sent = True
        self.past_image = np.array([])

    #def get_current_vel(self, twist):
     #   self.twist_msg.linear.x = round(twist.linear.x,3)
      #  self.twist_msg.angular.z = round(twist.angular.z,3)

    #def odom_callback(self, msg):
        # Position
     #   x = msg.pose.pose.position.x
      #  y = msg.pose.pose.position.y
       # z = msg.pose.pose.position.z
        
        # Orientation as Quaternion
        #qx = msg.pose.pose.orientation.x
        #qy = msg.pose.pose.orientation.y
        #qz = msg.pose.pose.orientation.z
        #qw = msg.pose.pose.orientation.w

        #print("X: " + str(x) + ", Y: " + str(y) + ", Z: " + str(z) + ", Qx: " + str(qx) + ", Qy: " + str(qy) + ", Qz: " + str(qz) + ", Qw: " + str(qw))

    def image_callback(self, data):

        # process the scribed image from camera in openCV 
        # convert image message to openCV image 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            rospy.logerr(e)
            return

        if not self.end_not_sent:
            # if end_not_sent is false
            # I stop driving
            # Create a Twist to stop the robot and publish to cmd_vel
            self.twist_msg.linear.x = 0
            self.twist_msg.angular.z = 0
            self.cmd_vel_pub.publish(self.twist_msg)


        if self.end_not_sent and self.timer_not_inited:
            #state 0: driving, looking for red
            


            # Create Twist message and publish to cmd_vel
            self.calculate_speed(cv_image)
            # print("speed: ", twist_msg.linear.x)
            # print("angular: ", twist_msg.angular.z)
            if self.driving_state >= 5:
                self.cmd_vel_pub.publish(self.twist_msg)

            #if self.driving_state == 6 and self.imgnum < 20000:
             #   self.imgnum += 1
              #  with open('/home/fizzer/Pictures/imitationdata.csv', 'a') as file:
               #     writer = csv.writer(file)
                #    writer.writerow([self.twist_msg.linear.x, self.twist_msg.angular.z])
                    #0.9^6
                #cv2.imwrite('/home/fizzer/Pictures/stage6photos/img' + str(self.imgnum) + '.png', cv_image)


    def calculate_speed(self, img):

        dim_x = img.shape[1]
        dim_y = img.shape[0]

        hsv = cv2.cvtColor(img[717:719,:,:], cv2.COLOR_BGR2HSV)
        #print(dim_x)
        #print(dim_y)
        # image processing: 
        # change the frame to grey scale
        #print("State: " + str(self.driving_state))
        if self.driving_state >= 0 and self.driving_state <= 4:
            
            #print("Img shape: " + str(img.shape))
            gray = cv2.cvtColor(img[717:719,:,:], cv2.COLOR_BGR2GRAY)

            if self.driving_state == 4:
                print("Time: " + str(int(time.time() - self.state_trans_start_time)))
                self.count_purples(hsv)

            else:
                redCount = self.count_reds(hsv)

                if self.driving_state == 0 and redCount > 480:
                    self.driving_state = 1
                    print("Detected red!: " + str(redCount))
                elif self.driving_state == 1 and redCount < 480:
                    self.driving_state = 2
                    print("Off the red!: " + str(redCount))
                elif self.driving_state == 2:
                    self.count_purples(hsv)
                    if redCount > 480:
                        self.driving_state = 3
                        print("Detected red again!: " + str(redCount))
                elif self.driving_state == 3 and redCount < 480:
                    self.driving_state = 4
                    self.state_trans_start_time = time.time()
                    print("Off the red again!: " + str(redCount))

            # not using gaussianBlur for now
            # blur it so the "road" far away become less easy to detect
            # kernel_size: Gaussian kernel size
            # sigma_x: Gaussian kernel standard deviation in X direction
            # sigma_y: Gaussian kernel standard deviation in Y direction
            #kernel_size = 13
            #sigma_x = 5
            #sigma_y = 5
            #blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x, sigma_y)  # gray scale the image
            #double_blur_gray = cv2.GaussianBlur(blur_gray[int(dim_y/3):int(3*dim_y/4), int(dim_x/5):int(4*dim_x/5)], (kernel_size, kernel_size), sigma_x, sigma_y)
            # Stopping when movement is detected
            #if self.past_image.size == 0:
            # self.past_image = double_blur_gray
            #else:
            #   subtracted_img = cv2.absdiff(double_blur_gray, self.past_image)
                #histogram = cv2.calcHist([subtracted_img], [0], None, [256], [0,256])
                #histogram /= histogram.sum()
                #cv2.imshow("camera view", subtracted_img)
                #cv2.waitKey(2)
                
            #  self.past_image = double_blur_gray
                #if subtracted_img.max() > 90:
                #   twist_msg.linear.x = 0
                #  twist_msg.angular.z = 0
                # return twist_msg


            # binary it
            # ret, binary = cv.threshold(blur_gray, 70, 255, cv.THRESH_BINARY)
            ret, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

            #cv2.imshow("camera view", binary)
            #cv2.waitKey(3)

            last_row = binary[-1,:]
            # print(last_row)
        elif self.driving_state == 5 or self.driving_state == 7:
            filtered_img = cv2.cvtColor(cv2.medianBlur(img[630:719,:,:], 71), cv2.COLOR_BGR2HSV)
            filtered_img = cv2.cvtColor(cv2.GaussianBlur(filtered_img, (41, 41), 0), cv2.COLOR_BGR2HSV)
            #filtered_img = cv2.cvtColor(cv2.bilateralFilter(filtered_img, 13, 63, 47), cv2.COLOR_BGR2HSV)

            self.count_purples(hsv)

            if self.driving_state == 5 or self.driving_state == 7:
                print("filtered img: " + str(filtered_img))
                YELLOW_MAX = np.array([50, 245, 197],np.uint8)
                YELLOW_MIN = np.array([11, 195, 145],np.uint8)

                cur_time = time.time() - self.state_trans_start_time
                #print("Time: " + str(round(cur_time)))
                if self.driving_state == 7 and cur_time < 45:
                    YELLOW_MAX = np.array([50, 230, 193],np.uint8)
                    YELLOW_MIN = np.array([10, 194, 120],np.uint8)
                elif self.driving_state == 7:
                    YELLOW_MAX = np.array([50, 230, 192],np.uint8)
                    YELLOW_MIN = np.array([10, 194, 120],np.uint8)



                #print("Y Max: " + str(YELLOW_MAX))
                #print("Y Min: " + str(YELLOW_MIN))

                frame_threshed = cv2.bitwise_not(cv2.inRange(filtered_img[87:89,:], YELLOW_MIN, YELLOW_MAX))
                for index in range(1,len(frame_threshed[1]) - 1):
                    if frame_threshed[1][index-1] == 0 and frame_threshed[1][index+1] == 0:
                        frame_threshed[1][index] = 0
                    elif frame_threshed[1][index-1] == 1 and frame_threshed[1][index+1] == 1:
                        frame_threshed[1][index] = 1

                last_row = frame_threshed[-1,:]
            #if self.driving_state == 7 and time.time() - self.state_trans_start_time < 5:
               #last_row = cv2.bitwise_and(last_row, 0)
                #contours, hierarchy = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
            #else:
             #   return Twist()
            #print("Last row: " + str(last_row))
        #print("last row: " + str(last_row.shape))
        # elif self.driving_
        # state == 6:

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
        elif self.driving_state == 6:
            if time.time() - self.state_trans_start_time >= 7:
                full_hsv = cv2.medianBlur(cv2.cvtColor(cv2.resize(img, dsize=(640, 360), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2HSV), 3)
                #print("not following")
                last_row = []
                self.mid_x = 640
                if time.time() - self.state_trans_start_time >= 11:
                    PURPLE_MIN = np.array([135, 210, 210],np.uint8)
                    PURPLE_MAX = np.array([165, 255, 255],np.uint8)
                    purple_threshold = cv2.inRange(full_hsv, PURPLE_MIN, PURPLE_MAX)
                    purple_contours, _ = cv2.findContours(purple_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    #print("Purple contours length: " + str(len(purple_contours)))
                    if len(purple_contours) > 0:
                        pM = cv2.moments(max(purple_contours, key = cv2.contourArea))
                        self.mid_x = 640
                        if pM['m00'] != 0:
                            #print("purple following")
                            self.mid_x = int(2*pM['m10']/pM['m00'])
                            self.Kp = 0.015
                    else:
                        BROWN_MIN = np.array([4, 129, 164],np.uint8)
                        BROWN_MAX = np.array([14, 150, 186],np.uint8)
                        brown_tracker = cv2.inRange(full_hsv, BROWN_MIN, BROWN_MAX)
                        brown_contours, _ = cv2.findContours(brown_tracker, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        #print("Brown contours length: " + str(len(brown_contours)))
                        if len(brown_contours) > 0:
                            bM = cv2.moments(max(brown_contours, key = cv2.contourArea))
                            if bM['m00'] != 0:
                                #print("brown following")
                                self.mid_x = int(2*bM['m10']/bM['m00'])
                else:
                    BROWN_MIN = np.array([4, 129, 164],np.uint8)
                    BROWN_MAX = np.array([14, 150, 186],np.uint8)
                    brown_tracker = cv2.inRange(full_hsv, BROWN_MIN, BROWN_MAX)
                    brown_contours, _ = cv2.findContours(brown_tracker, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    #print("Brown contours length: " + str(len(brown_contours)))
                    if len(brown_contours) > 0:
                        bM = cv2.moments(max(brown_contours, key = cv2.contourArea))
                        if bM['m00'] != 0:
                            #print("brown following")
                            self.mid_x = int(2*bM['m10']/bM['m00'])

                self.count_purples(hsv)


        if not (self.driving_state == 6):
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

        current_time = time.time()

        if self.driving_state == 1:
            self.twist_msg.linear.x = 0.17
            # self.twist_msg.linear.x = 0.0
            # if current_time - self.state_trans_start_time > 1.1:
            #     subtraction = cv2.absdiff(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.past_image)
            #     subtraction = cv2.threshold(subtraction, 15, 255, cv2.THRESH_BINARY)
            #     for i in range(len(subtraction)):
            #         for j in range(len(subtraction[0])):
            #             if subtraction[i][j] > 0:
            #                 self.driving_state = 2
        elif self.driving_state == 2:
            self.twist_msg.linear.x = self.linear_val_max + 0.16
        elif self.driving_state == 4:
            self.twist_msg.linear.x = self.linear_val_max - 0.06
            if int(current_time - self.state_trans_start_time) > 11:
                self.twist_msg.angular.z += 0.25
                if int(current_time - self.state_trans_start_time) > 13:
                    self.twist_msg.angular.z += 0.24
                    if int(current_time - self.state_trans_start_time) > 20:
                        self.twist_msg.angular.z -= 0.1
                        if int(current_time - self.state_trans_start_time) > 25:
                            self.twist_msg.angular.z += 0.35
        elif self.driving_state == 5:
            self.twist_msg.linear.x = self.linear_val_max - 0.2053
        elif self.driving_state == 6:
            self.twist_msg.linear.x = self.linear_val_max - 0.045
            if current_time - self.state_trans_start_time < 2:
                self.twist_msg.angular.z = -0.08
                self.twist_msg.linear.x = 0.28
            elif current_time - self.state_trans_start_time < 4:
                self.twist_msg.angular.z = 0.269
                self.twist_msg.linear.x = 0.3
            elif current_time - self.state_trans_start_time < 7:
                self.twist_msg.angular.z = 0.59
                self.twist_msg.linear.x = 0.32
                
        elif self.driving_state == 7:
            self.twist_msg.linear.x = self.linear_val_max - 0.194
            if int(current_time - self.state_trans_start_time) < 3: 
                self.twist_msg.angular.z += 1.065
            #elif int(current_time - self.state_trans_start_time) < 55: 
            elif int(current_time - self.state_trans_start_time) > 8:
                #self.twist_msg.angular.z += 0.125
                self.twist_msg.angular.z += 0.062
            self.Kp = 0.042
        else:
            self.twist_msg.linear.x = self.linear_val_max

        if abs(angular_error) > self.error_threshold and not (self.driving_state == 6 or self.driving_state == 2):
            self.twist_msg.linear.x = self.linear_val_min

        if self.on_purple == 1:
            self.twist_msg.angular.z = 0.0
            self.twist_msg.linear.x = 0.23

        self.past_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    def clock_callback(self, data):

        start_msg = 0
        stop_msg = -1
        string_message = '14,password,{0},NA'

        start_message = string_message.format(start_msg)
        stop_message = string_message.format(stop_msg)

        duration = 400
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

    def count_reds(self, img):
        RED_MIN = np.array([0, 235, 235],np.uint8)
        RED_MAX = np.array([25, 255, 255],np.uint8)

        red_threshed = cv2.inRange(img, RED_MIN, RED_MAX)
        
        redCount = 0

        for index in range(len(red_threshed)):
            for index2 in range(len(red_threshed[0])):
                if red_threshed[index][index2] > 0:
                    redCount += 1
        
        return redCount
    
    def count_purples(self, img):
        PURPLE_MIN = np.array([135, 210, 210],np.uint8)
        PURPLE_MAX = np.array([165, 255, 255],np.uint8)
        purple_threshed = cv2.inRange(img, PURPLE_MIN, PURPLE_MAX)
        purpleCount = 0
        for index in range(len(purple_threshed)):
            for index2 in range(len(purple_threshed[0])):
                if purple_threshed[index][index2] > 0:
                    purpleCount += 1

        if purpleCount > 340 or (self.driving_state == 6 and purpleCount > 200):
            self.on_purple = 1
            print("Detected purple!: " + str(purpleCount))
        elif self.on_purple == 1:
            if (self.driving_state == 4 or self.driving_state == 5) and time.time() - self.state_trans_start_time > 10.0:
                self.state_trans_start_time = time.time()
                self.twist_msg.angular.z = 0
                self.twist_msg.linear.x = 0.3
                self.cmd_vel_pub.publish(self.twist_msg)
                self.driving_state += 1
                print("State: " + str(self.driving_state))
                self.on_purple = 0
            elif self.driving_state == 6:
                self.state_trans_start_time = time.time()
                self.twist_msg.angular.z = 0.7
                self.twist_msg.linear.x = 0.0
                self.cmd_vel_pub.publish(self.twist_msg)
                self.driving_state += 1
                self.on_purple = 0
                #self.spawn_position([-3.88, 0.476, 0.1], 0.0, 0.0, 3.14)
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
            resp = set_state( msg )

        except rospy.ServiceException:
            print ("Service call failed")



def main():
    try:
        image_subscriber = Drive()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
