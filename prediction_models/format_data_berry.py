# -*- coding: utf-8 -*-
# RUN IN PYTHON 3

import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import torch.nn as nn
import os
import cv2
import csv
import glob
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pickle import dump
from sklearn import preprocessing
from datetime import datetime
from scipy.spatial.transform import Rotation as R


#dataset_path = "/home/gabriele/data/shelf_can_aid/shelf_can_aid/"
dataset_path = "/media/gabrielle/Extreme SSD/data/Pushing_Strawberries/"
# dataset_path = "/home/gabriele/3DRED/"
# Hyper-parameters:
train_data_dir = dataset_path + 'train/'
test_data_dir = dataset_path + 'test/'

train_out_dir = dataset_path + 'train_formatted/'
test_out_dir = dataset_path + 'test_formatted/'
scaler_out_dir = dataset_path + 'filler_scaler/'

smooth = False
image = False
image_height = 256
image_width = 256
context_length = 5       #####
horrizon_length = 10      #####
one_sequence_per_test = False
data_train_percentage = 1.0


class data_formatter:
    def __init__(self):
        self.files_train = []
        self.files_test = []
        self.full_data_finger = []
        self.full_data_robot = []
        self.smooth = smooth
        self.image = image
        self.image_height = image_height
        self.image_width = image_width
        self.context_length = context_length
        self.horrizon_length = horrizon_length
        self.one_sequence_per_test = one_sequence_per_test
        self.data_train_percentage = data_train_percentage

    def create_map(self):
        for stage in [test_out_dir]:  # [train_out_dir, test_out_dir, test_out_dir_2]:
            self.path_file = []
            index_to_save = 0
            print(stage)
            if stage == train_out_dir:
                files_to_run = self.files_train
            elif stage == test_out_dir:
                files_to_run = self.files_test
           
            print(files_to_run)

            path_save = stage
            for experiment_number, file in tqdm(enumerate(files_to_run)):

                finger, robot = self.load_file_data(file)                         #####


                for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
                    robot[:, index] = np.squeeze(min_max_scalar.transform(robot[:, index].reshape(-1, 1)))

                # save images and save space:
                finger_image_names = []
                for time_step in range(len(finger)):
                    image_name = "finger_image_" + str(experiment_number) + "_time_step_" + str(time_step) + ".npy"
                    finger_image_names.append(image_name)
                    #path_save_im = path_save + 'images/'
                    np.save(path_save + image_name, finger[time_step])


                if self.one_sequence_per_test:
                    sequence_length = self.context_length + self.horrizon_length - 1
                else:
                    sequence_length = self.context_length + self.horrizon_length
                for time_step in range(len(finger) - sequence_length):
                    robot_data_euler_sequence = [robot[time_step + t] for t in range(sequence_length)]
                    # tactile_data_sequence = [tactile[time_step + t] for t in range(sequence_length)]
                    finger_name_sequence = [finger_image_names[time_step + t] for t in range(sequence_length)]
                    
                    experiment_data_sequence = experiment_number
                    time_step_data_sequence = [time_step + t for t in range(sequence_length)]

                    ####################################### Save the data and add to the map ###########################################
                    #path_save_rde = path_save + 'robot_data_euler/'
                    np.save(path_save + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence)
                    #path_save_tds = path_save + 'tactile_data_sequence/'
                    np.save(path_save + 'finger_name_sequence_' + str(index_to_save), finger_name_sequence)
                    np.save(path_save + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
                    #path_save_tsd = path_save + 'time_step_data/'
                    np.save(path_save + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
                    ref = []
                    ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
                    ref.append('finger_name_sequence_' + str(index_to_save) + '.npy')
                    ref.append('experiment_number_' + str(index_to_save) + '.npy')
                    ref.append('time_step_data_' + str(index_to_save) + '.npy')
                    self.path_file.append(ref)
                    index_to_save += 1
                    if self.one_sequence_per_test:
                        break
                # if stage != train_out_dir:
                #     self.test_no = experiment_number
                #     self.save_map(path_save, test=True)

            self.save_map(path_save)

    def save_map(self, path, test=False):
        if test:
            with open(path + '/map_' + str(self.test_no) + '.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(
                    ['robot_data_path_euler', 'finger_name_sequence', 'experiment_number', 'time_steps'])
                for row in self.path_file:
                    writer.writerow(row)
        else:
            with open(path + '/map.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(
                    ['robot_data_path_euler', 'finger_name_sequence', 'experiment_number', 'time_steps'])
                for row in self.path_file:
                    writer.writerow(row)

    def scale_data(self):
        files = self.files_train + self.files_test
        # files = self.files_train
        for file in tqdm(files):
            finger, robot = self.load_file_data(file)
            self.full_data_robot += list(robot)

        self.full_data_robot = np.array(self.full_data_robot)

        self.robot_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.full_data_robot[:, feature].reshape(-1, 1)) for feature in range(6)]
    
        # self.save_scalars()

        return robot

    def load_file_data(self, file):
        finger_sensor = np.array(np.load(file + '/camera_finger.npy'))
        robot_state = np.array(pd.read_csv(file + '/robot_state.csv', header=None))   # robot_states.csv

        # convert orientation to euler, and remove column labels:
        robot_task_space = np.array([[state[-7], state[-6], state[-5]] + list(
            R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)) for state in
                                     robot_state[1:]]).astype(float)
       
        # Resize the image using PIL antialiasing method (Copied from CDNA data formatting)
        raw = []
        for k in range(len(finger_sensor)):
            tmp = Image.fromarray(finger_sensor[k])
            tmp = tmp.resize((image_height, image_width), Image.ANTIALIAS)
            tmp = np.fromstring(tmp.tobytes(), dtype=np.uint8)
            tmp = tmp.reshape((image_height, image_width, 3))
            tmp = tmp.astype(np.float32) / 255.0
            raw.append(tmp)
        finger_sensor = np.array(raw)

        if self.smooth:
            robot_task_space = robot_task_space[3:-3, :]
            finger_sensor = finger_sensor[3:-3]
        return finger_sensor, robot_task_space
        

    def load_file_names(self):
        self.files_train = glob.glob(train_data_dir + '/*')
        self.files_test = glob.glob(test_data_dir + '/*')
        # self.files_test_2 = glob.glob(test_data_dir_2 + '/*')
        self.files_train = random.sample(self.files_train, int(len(self.files_train) * self.data_train_percentage))

    def smooth_the_trial(self, tactile_data):
        for force in range(tactile_data.shape[1]):
            for taxel in range(tactile_data.shape[2]):
                tactile_data[:, force, taxel] = [None for i in range(3)] + list(self.smooth_func(tactile_data[:, force, taxel], 6)[3:-3]) + [None for i in range(3)]

        return tactile_data

    def smooth_func(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth


    def save_scalars(self):

        dump(self.robot_min_max_scalar[0], open(scaler_out_dir + 'robot_min_max_scalar_px.pkl', 'wb'))
        dump(self.robot_min_max_scalar[1], open(scaler_out_dir + 'robot_min_max_scalar_py.pkl', 'wb'))
        dump(self.robot_min_max_scalar[2], open(scaler_out_dir + 'robot_min_max_scalar_pz.pkl', 'wb'))
        dump(self.robot_min_max_scalar[3], open(scaler_out_dir + 'robot_min_max_scalar_ex.pkl', 'wb'))
        dump(self.robot_min_max_scalar[4], open(scaler_out_dir + 'robot_min_max_scalar_ey.pkl', 'wb'))
        dump(self.robot_min_max_scalar[5], open(scaler_out_dir + 'robot_min_max_scalar_ez.pkl', 'wb'))

    def create_image(self, tactile_x, tactile_y, tactile_z):
        # convert tactile data into an image:
        image = np.zeros((4, 4, 3), np.float32)
        index = 0
        for x in range(4):
            for y in range(4):
                image[x][y] = [tactile_x[index], tactile_y[index], tactile_z[index]]
                index += 1
        reshaped_image = np.rot90(cv2.resize(image.astype(np.float32), dsize=(self.image_height, self.image_width), interpolation=cv2.INTER_CUBIC), k=1, axes=(0, 1))
        return reshaped_image


def main():
    df = data_formatter()
    df.load_file_names()
    df.scale_data()
    df.create_map()


if __name__ == "__main__":
    main()
