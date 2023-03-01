#! /usr/bin/env python3
import time
import glob
import rospy
import torch
import pickle
import numpy as np
import torch.nn as nn
import PIL.Image as PILImage
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
	pred_model = model()
	pred_model.load_state_dict(torch.load(\
								"/home/kiyanoush/Cpp_ws/src/haptic_finger_control/force_localisation/dataset/localisation_cnn_8_16_resmpl.pth"))
	pred_model.eval()
	for param in pred_model.parameters():
		param.grad = None

	name = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015",\
	 		"016", "017", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030"]
	for s in name:
		print(s)
		data_path = "/home/kiyanoush/Cpp_ws/src/haptic_finger_control/RT-Data/proactive/final_tests/" + s 
		haptic_finger_data = np.load(data_path + "/haptic_finger_raw.npy")
		localisation_offline = np.zeros((len(haptic_finger_data), 1))
		haptic_finger_data = np.ascontiguousarray(haptic_finger_data.astype(np.uint8))
		for index, sample_image in enumerate(haptic_finger_data):
			haptic_finger_img = PILImage.fromarray(sample_image).resize((256, 256), PILImage.Resampling.LANCZOS)
			haptic_finger_img = np.array(haptic_finger_img)
			haptic_finger_img[170:] = 0
			haptic_finger_img[150:170, 100:150] = 0
			haptic_finger_img = np.fromstring(haptic_finger_img.tobytes(), dtype=np.uint8)
			haptic_finger_img = haptic_finger_img.reshape((3, 256, 256))

			tactile_norm = haptic_finger_img / 255.0

			if index == 0:
				first_frame = tactile_norm
				localisation_offline[index] = 0.63
			if index > 0:
				tactile_norm = tactile_norm - first_frame
				tactile_tensor = torch.from_numpy(tactile_norm).unsqueeze(0).float()

				stem_pose = pred_model(tactile_tensor).item()
				localisation_offline[index] = stem_pose

		np.save(data_path + "/localisation_offline.npy", localisation_offline)
				
				# plt.imshow(sample_image)
