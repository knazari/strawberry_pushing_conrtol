
#! /usr/bin/env python3
import tf
import sys
import time
import copy
import rospy
import actionlib
import numpy as np
import moveit_msgs.msg
import moveit_commander
import matplotlib.pyplot as plt
import PIL.Image as PILImage
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import quaternion_matrix

class Robot(object):

    def __init__(self, save_path):
        super(Robot, self).__init__()
        self.robot_state = moveit_commander.RobotCommander()
        self.setup_planner()
        self.listener = tf.TransformListener()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.JOINT_BASE = 0
        self.JOINT_WRIST = 6
        self.ee_to_finger = 0.13
        self.replan = False
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.sub = rospy.Subscriber('/fing_camera/color/image_raw', Image, self.finger_callback)
        self.save_path = save_path
        self.bridge = CvBridge()

    def setup_planner(self):
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.group.set_end_effector_link("panda_hand")    # planning wrt to panda_hand or link8
        self.group.set_max_velocity_scaling_factor(0.50)  # scaling down velocity
        self.group.set_max_acceleration_scaling_factor(0.5)  # scaling down velocity
        self.group.allow_replanning(True)
        self.group.set_num_planning_attempts(10)
        self.group.set_goal_position_tolerance(0.0005)
        self.group.set_goal_orientation_tolerance(0.001)
        self.group.set_planning_time(5)
        self.group.set_planner_id("RRTConnectkConfigDefault")
        rospy.sleep(2)
        print("Ready to go")

    def current_pose(self):
        wpose = self.group.get_current_pose().pose
        return wpose

    def current_state(self):
        joint_state = self.group.get_current_joint_values()
        return joint_state

    def go_home(self):
        p = PoseStamped()
        p.header.frame_id = '/panda_link0'

        p.pose.position.x = 0.43
        p.pose.position.y = 0.00
        p.pose.position.z = 0.43

        p.pose.orientation.x = 1
        p.pose.orientation.y = 0
        p.pose.orientation.z = 0
        p.pose.orientation.w = 0

        target = self.group.set_pose_target(p)

        self.group.go(target)

    def go_to_pose(self, x, y, z):
        p = PoseStamped()
        p.header.frame_id = '/panda_link0'

        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z

        p.pose.orientation.x = 1
        p.pose.orientation.y = 0
        p.pose.orientation.z = 0
        p.pose.orientation.w = 0

        target = self.group.set_pose_target(p)

        self.group.go(target)

    def go_to_pose_with_rotation(self, x, y, z, wx, wy, wz, ww):
        p = PoseStamped()
        p.header.frame_id = '/panda_link0'

        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z

        p.pose.orientation.x = wx
        p.pose.orientation.y = wy
        p.pose.orientation.z = wz
        p.pose.orientation.w = ww

        target = self.group.set_pose_target(p)

        self.group.go(target)

    def finger_callback(self, haptic_finger_msg):
        image_message = haptic_finger_msg
        haptic_finger_img = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
        haptic_finger_img = PILImage.fromarray(haptic_finger_img).resize((256, 256), PILImage.ANTIALIAS)
        haptic_finger_img_new = np.array(haptic_finger_img)
        
        self.haptic_finger_data = haptic_finger_img_new
        
    def save_localisation_result(self):
        
        np.save(self.save_path + "/force_position.npy", np.array(self.dist))
        np.save(self.save_path + "/haptic_finger_data.npy", self.haptic_finger_data)
        
    def collect_force_localisation_data(self, dist):
        self.dist = dist # this can be one of the 
        
        c_pose = self.current_pose()
        self.go_to_pose(c_pose.position.x, c_pose.position.y, c_pose.position.z-0.001)

        print("started sleeping ...")
        time.sleep(2)
        self.save_localisation_result()
        print("finished saving ...")


def main():
    save_path = "/home/kiyanoush/Cpp_ws/src/haptic_finger_control/force_localisation/dataset/124"
    robot = Robot(save_path=save_path)
    c_pose = robot.current_pose()
    robot.go_to_pose(c_pose.position.x, c_pose.position.y, c_pose.position.z)
    distance_to_base = 0.000001
    robot.collect_force_localisation_data(distance_to_base)

    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('localisation_data_collection')
    try:
        moveit_commander.roscpp_initialize(sys.argv)
        main()
    except Exception as e:
        print(e)
    else:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)