#! /usr/bin/env python3
import rospy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import message_filters
from pickle import load
import PIL.Image as PILImage

from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R

from ATCVP.ATCVP import Model

class PushingController:
	def __init__(self):
		super(PushingController, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.time_step = 0
		self.stop = 0
		self.bridge = CvBridge()
		self.dummy_scene_seq = torch.zeros((20, 1, 3, 256, 256))
		self.haptic_finger_data   = np.zeros((1000, 3, 256, 256))
		self.haptic_finger_data_scaled	  = np.zeros((1000, 360, 480, 3))
		self.robot_data 	      = np.zeros((1000, 6))
		self.action_data 	      = np.zeros((1000, 6))
		self.robot_data_scaled 	  = np.zeros((1000, 6))
		self.action_data_scaled   = np.zeros((1000, 6))
		self.tactile_predictions  = np.zeros((1000, 10, 3, 256, 256))
		self.tactile_predictions_descaled = np.zeros((1000, 10, 3, 256, 256))

		rospy.init_node('listener', anonymous=True, disable_signals=True)
		self.load_model()
		self.load_scalers()
		self.init_sub()
		self.control_loop()
	
	def init_sub(self):
		robot_sub = message_filters.Subscriber('/robot_pose', Float64MultiArray)
		haptic_finger_sub = message_filters.Subscriber("/fing_camera/color/image_raw", Image)
		sync_cb = message_filters.ApproximateTimeSynchronizer([robot_sub, haptic_finger_sub], 1, 0.1, allow_headerless=True)
		sync_cb.registerCallback(self.callback)
	
	def load_scalers(self):
		scaler_path = '/home/kiyanoush/Cpp_ws/src/haptic_finger_control/scalars'
		self.robot_min_max_scalar = [load(open(scaler_path + '/robot_min_max_scalar_'+feature +'.pkl', 'rb'))\
															 for feature in ['px', 'py', 'pz', 'ex', 'ey', 'ez']]

	def scale_nn_input(self, haptic, robot, action):
		scaled_haptic = haptic[np.newaxis, :, :, :, :] / 255.0
		scaled_robot = robot
		scaled_action = action
		for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
			scaled_robot[:, index] = np.squeeze(min_max_scalar.transform(scaled_robot[:, index].reshape(-1, 1)))
			scaled_action[:, index] = np.squeeze(min_max_scalar.transform(scaled_action[:, index].reshape(-1, 1)))
		scaled_action = np.concatenate((scaled_robot, scaled_action), axis=0)
		
		scaled_haptic = torch.from_numpy(scaled_haptic).view(10, 1, 3, 256, 256).double()
		scaled_action = torch.from_numpy(scaled_action).view(20, 1, 6).double()

		return scaled_haptic, scaled_action
	
	def load_model(self):
		n_past = 10
		n_future = 10
		model_dir = "/home/kiyanoush/Cpp_ws/src/haptic_finger_control/src/ATCVP/Tactile_Predictive_Model_200_Epochs/model_07_02_2023_12_50/"
		model_name_save_appendix = "ATCVP_model"

		features = dict([("device", self.device), ("n_past", n_past), ("n_future", n_future), ("model_dir", model_dir),\
										("model_name_save_appendix", model_name_save_appendix), ("criterion", nn.MSELoss())])
		self.pred_model = Model(features)
		self.pred_model.load_state_dict(torch.load("/home/kiyanoush/Cpp_ws/src/haptic_finger_control/src/ATCVP/Tactile_Predictive_Model_200_Epochs/model_07_02_2023_12_50/ATCVP_model", map_location='cpu'))
		self.pred_model = self.pred_model.double()
		self.pred_model.eval()

	def callback(self, robot_msg, haptic_finger_msg):
		self.stop = robot_msg.data[-1]
		if self.stop == 0:
			x = robot_msg.data[12]
			y = robot_msg.data[13]
			z = robot_msg.data[14]
			euler = R.from_matrix([[robot_msg.data[0], robot_msg.data[4], robot_msg.data[8]],\
									[robot_msg.data[1], robot_msg.data[5], robot_msg.data[9]],\
									[robot_msg.data[2], robot_msg.data[6], robot_msg.data[10]]])\
																	.as_euler('zyx', degrees=True)
			self.robot_data[self.time_step] = np.array([x, y, z, euler[0], euler[1], euler[2]])
			haptic_finger_img = self.bridge.imgmsg_to_cv2(haptic_finger_msg, desired_encoding='passthrough')
			haptic_finger_img = PILImage.fromarray(haptic_finger_img).resize((256, 256), PILImage.ANTIALIAS)
			haptic_finger_img = np.array(haptic_finger_img)
			self.haptic_finger_data[self.time_step] = np.moveaxis(haptic_finger_img, -1, 0)	

		self.time_step +=1
	
	def prdict_tactile_seq(self):
		haptic_finger_seq  = self.haptic_finger_data[self.time_step-10 : self.time_step]
		robot_seq  = self.robot_data[self.time_step-10 : self.time_step]
		action_seq = self.action_data[self.time_step-10 : self.time_step]
		haptic_finger_seq_scaled, action_seq_scaled = \
						self.scale_nn_input(haptic_finger_seq, robot_seq, action_seq)
		# t1 = time.time()
		tactile_prediction = self.pred_model.run(self.dummy_scene_seq,  action_seq_scaled, haptic_finger_seq_scaled)
		# print(f'inference time: {time.time() - t1}')
	
	def control_loop(self):
		rate = rospy.Rate(30)
		
		while not rospy.is_shutdown():
			
			try:
				if self.time_step == 0:
					self.t0 = time.time()

				if self.stop == 0.0 and self.time_step > 10:
					self.prdict_tactile_seq()
					pass
				elif self.stop == 1.0:
					[sub.sub.unregister() for sub in self.sync_subscriber]
					break

				rate.sleep()
			
			except KeyboardInterrupt:
				break
	

if __name__ == '__main__':
	pc = PushingController()
