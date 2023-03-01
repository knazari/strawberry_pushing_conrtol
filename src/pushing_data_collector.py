#! /usr/bin/env python3
import os
import time
import rospy
import datetime
import numpy as np
import message_filters
import PIL.Image as PILImage
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R


class PushingDataCollector:
	def __init__(self):
		super(PushingDataCollector, self).__init__()
		
		self.stop = 0
		self.time_step = 0
		self.bridge = CvBridge()
		self.robot_pose_data = np.zeros((1000, 10))
		self.robot_vel_data = np.zeros((1000, 3))
		self.haptic_finger_data = np.zeros((1000, 3, 360, 480))
		self.save_path = "/home/kiyanoush/Cpp_ws/src/haptic_finger_control/RT-Data/reactive/042/"

		rospy.init_node('listener', anonymous=True, disable_signals=True)
		self.rot_publisher = rospy.Publisher('/stem_pose', Float64MultiArray, queue_size=1)
		self.init_sub()
		self.control_loop()
	
	def init_sub(self):
		robot_pose_sub = message_filters.Subscriber('/robot_pose', Float64MultiArray)
		robot_vel_sub = message_filters.Subscriber('/robot_vel', Float64MultiArray)
		haptic_finger_sub = message_filters.Subscriber("/fing_camera/color/image_raw", Image)

		self.sync_sub = [robot_pose_sub, robot_vel_sub, haptic_finger_sub]

		sync_cb = message_filters.ApproximateTimeSynchronizer(self.sync_sub, 1, 0.1, allow_headerless=True)
		sync_cb.registerCallback(self.callback)

	def callback(self, robot_poes_msg, robot_vel_msg, haptic_finger_msg):
		self.stop = robot_poes_msg.data[-1]
		if self.stop == 0:
			
			rot_mat = R.from_matrix([[robot_poes_msg.data[0], robot_poes_msg.data[4], robot_poes_msg.data[8]],\
									[robot_poes_msg.data[1], robot_poes_msg.data[5], robot_poes_msg.data[9]],\
									[robot_poes_msg.data[2], robot_poes_msg.data[6], robot_poes_msg.data[10]]])												
			
			euler = rot_mat.as_euler('zyx', degrees=True)
			quat  = rot_mat.as_quat()

			self.robot_pose_data[self.time_step] = np.array([robot_poes_msg.data[12], robot_poes_msg.data[13], robot_poes_msg.data[14],\
														 euler[0], euler[1], euler[2],\
														quat[0], quat[1], quat[2], quat[3]])
			self.robot_vel_data[self.time_step] = np.array([robot_vel_msg.data[0], robot_vel_msg.data[1], robot_vel_msg.data[2]])

			haptic_finger_img = self.bridge.imgmsg_to_cv2(haptic_finger_msg, desired_encoding='passthrough')
			haptic_finger_img = PILImage.fromarray(haptic_finger_img)#.resize((256, 256), PILImage.Resampling.LANCZOS)
			haptic_finger_img = np.array(haptic_finger_img)
			# haptic_finger_img[170:] = 0
			# haptic_finger_img[150:170, 100:150] = 0
			
			self.haptic_finger_data[self.time_step] = np.moveaxis(haptic_finger_img, -1, 0)

		self.time_step +=1
	
	def save_data(self):
		self.folder = str('/home/kiyanoush/Cpp_ws/src/haptic_finger_control/RT-Data/dataset/data_sample_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
		mydir = os.mkdir(self.folder)
		np.save(self.folder + "/haptic_finger.npy", self.haptic_finger_data[:self.time_step-1])
		np.save(self.folder + "/robot_pose.npy", self.robot_pose_data[:self.time_step-1]) # columns: x, y, z, eu_x, eu_y, eu_z, quat_x, quat_y, quat_z, _quat_w
		np.save(self.folder + "/robot_velocity.npy", self.robot_vel_data[:self.time_step-1]) # v_y and w_z, and w_z_des
	
	def control_loop(self):
		rate = rospy.Rate(60)
		
		while not rospy.is_shutdown():
			
			try:
				if self.time_step == 0:
					self.t0 = time.time()

				if self.stop == 0.0 and self.time_step > 0:
					pass
				elif self.stop == 1.0:
					[sub.sub.unregister() for sub in self.sync_sub]
					break

				rate.sleep()
			
			except KeyboardInterrupt:
				break
	

if __name__ == '__main__':
	pc = PushingDataCollector()
	pc.save_data()

