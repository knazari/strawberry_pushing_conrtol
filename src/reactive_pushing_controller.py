#! /usr/bin/env python3
import cv2
import time
import rospy
import torch
import pickle
import numpy as np
import torch.nn as nn
import message_filters
import PIL.Image as PILImage
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R

class model(nn.Module):
    
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*64*64, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
    
    def forward(self, tactile):
        x = self.relu(self.maxpool(self.conv1(tactile)))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class PushingController:
	def __init__(self):
		super(PushingController, self).__init__()
		
		self.stop = 0
		self.time_step = 0
		self.bridge = CvBridge()
		self.robot_pose_data = np.zeros((1000, 10))
		self.robot_vel_data = np.zeros((1000, 6))
		self.localisation = np.zeros((1000, 1))
		self.haptic_finger_data = np.zeros((1000, 3, 256, 256))
		self.save_path = "/home/kiyanoush/Cpp_ws/src/haptic_finger_control/RT-Data/video_single_stem/failure_open_loop/004/"

		rospy.init_node('listener', anonymous=True, disable_signals=True)
		self.rot_publisher = rospy.Publisher('/stem_pose', Float64MultiArray, queue_size=1)
		self.init_sub()
		self.load_model()
		self.load_scaler()
		self.control_loop()
	
	def init_sub(self):
		robot_pose_sub = message_filters.Subscriber('/robot_pose', Float64MultiArray)
		robot_vel_sub = message_filters.Subscriber('/robot_vel', Float64MultiArray)
		haptic_finger_sub = message_filters.Subscriber("/fing_camera/color/image_raw", Image)

		self.sync_sub = [robot_pose_sub, robot_vel_sub, haptic_finger_sub]

		sync_cb = message_filters.ApproximateTimeSynchronizer(self.sync_sub, 1, 0.1, allow_headerless=True)
		sync_cb.registerCallback(self.callback)
	
	def load_model(self):
		
		self.pred_model = model()
		self.pred_model.load_state_dict(torch.load(\
									"/home/kiyanoush/Cpp_ws/src/haptic_finger_control/force_localisation/dataset/localisation_cnn_8_16_resmpl.pth"))
		self.pred_model.eval()
		for param in self.pred_model.parameters():
			param.grad = None

	def load_scaler(self):

		self.nn_pred_scaler = pickle.load(open("/home/kiyanoush/Cpp_ws/src/haptic_finger_control/force_localisation/distance_scaler.pkl", 'rb'))

	def callback(self, robot_poes_msg, robot_vel_msg, haptic_finger_msg):
		self.stop = robot_poes_msg.data[-1]
		if self.stop == 0:
			
			rot_mat = R.from_matrix([[robot_poes_msg.data[0], robot_poes_msg.data[4], robot_poes_msg.data[8]],\
									[robot_poes_msg.data[1], robot_poes_msg.data[5], robot_poes_msg.data[9]],\
									[robot_poes_msg.data[2], robot_poes_msg.data[6], robot_poes_msg.data[10]]])												
			
			euler = rot_mat.as_euler('xyz', degrees=True)
			quat  = rot_mat.as_quat()

			self.robot_pose_data[self.time_step] = np.array([robot_poes_msg.data[12], robot_poes_msg.data[13], robot_poes_msg.data[14],\
														 euler[0], euler[1], euler[2],\
														quat[0], quat[1], quat[2], quat[3]])
			self.robot_vel_data[self.time_step] = np.array(robot_vel_msg.data)

			haptic_finger_img = self.bridge.imgmsg_to_cv2(haptic_finger_msg, desired_encoding='passthrough')
			cv2.imwrite(self.save_path + "image/" + str(self.time_step) + ".png", haptic_finger_img)
			haptic_finger_img = PILImage.fromarray(haptic_finger_img).resize((256, 256), PILImage.Resampling.LANCZOS)
			haptic_finger_img = np.array(haptic_finger_img)
			haptic_finger_img[170:] = 0
			haptic_finger_img[150:170, 100:150] = 0
			
			haptic_finger_img_chanlfrst = np.moveaxis(haptic_finger_img, -1, 0)

			if self.time_step == 0:
				self.untouched_img = haptic_finger_img_chanlfrst
			
			self.haptic_finger_data[self.time_step] = haptic_finger_img_chanlfrst

		self.time_step +=1
	
	def save_data(self):
		np.save(self.save_path + "haptic_finger.npy", self.haptic_finger_data[:self.time_step-1])
		np.save(self.save_path + "robot_pose.npy", self.robot_pose_data[:self.time_step-1]) # columns: x, y, z, eu_x, eu_y, eu_z, quat_x, quat_y, quat_z, _quat_w
		np.save(self.save_path + "robot_velocity.npy", self.robot_vel_data[:self.time_step-1]) # v_y and w_z, and w_z_des
		np.save(self.save_path + "localisation.npy", self.localisation[:self.time_step-1])
	
	def preprocess_tactile(self, tactile):
		tactile_norm = tactile / 255.0
		tactile_norm = tactile_norm - self.untouched_img / 255.0
		tactile_tensor = torch.from_numpy(tactile_norm).unsqueeze(0)
		
		return tactile_tensor.float()

	def get_stem_position(self):

		tactile_input = self.preprocess_tactile(self.haptic_finger_data[self.time_step-1])
		stem_pose = self.pred_model(tactile_input).item()
		self.localisation[self.time_step] = stem_pose
		# print(stem_pose)
		
		return stem_pose
	
	def publish_stem_position(self):

		stem_pose = self.get_stem_position()
		step_pose_msg = Float64MultiArray()
		step_pose_msg.data = [stem_pose, 0.0]
		self.rot_publisher.publish(step_pose_msg)
	
	def control_loop(self):
		rate = rospy.Rate(60)
		
		while not rospy.is_shutdown():
			
			try:
				if self.time_step == 0:
					self.t0 = time.time()

				if self.stop == 0.0 and self.time_step > 0:
					self.publish_stem_position()
				elif self.stop == 1.0:
					[sub.sub.unregister() for sub in self.sync_sub]
					break

				rate.sleep()
			
			except KeyboardInterrupt:
				break
	

if __name__ == '__main__':
	pc = PushingController()
	pc.save_data()

