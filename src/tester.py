# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import sys
import csv
# import cv2
import numpy as np
import click
import random
import time

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

# standard video and tactile prediction models:
from universal_networks.SVG import Model as SVG
from universal_networks.TBF import Model as TBF
# from universal_networks.ATCVP import Model as ATCVP
from universal_networks.BackupScripts.TP.ote.Extra_LSTM.ATCVP import Model as ATCVP
from universal_networks.FTVP.FF.FTVP64 import Model as FTVP64
from universal_networks.FF_128.two.FTVP128 import Model as FTVP128
from universal_networks.ACVP import Model as ACVP
from universal_networks.SIMVP import Model as SIMVP

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze (0).unsqueeze (0)
    window = Variable (_2D_window.expand (channel, 1, window_size, window_size).contiguous ())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d (img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d (img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow (2)
    mu2_sq = mu2.pow (2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d (img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d (img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d (img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean ()
    else:
        return ssim_map.mean (1).mean (1).mean (1)

class SSIM (torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super (SSIM, self).__init__ ()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window (window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type () == img1.data.type ():
            window = self.window
        else:
            window = create_window (self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda (img1.get_device ())
            window = window.type_as (img1)

            self.window = window
            self.channel = channel

        return _ssim (img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size ()
    window = create_window (window_size, channel)

    if img1.is_cuda:
        window = window.cuda (img1.get_device ())
    window = window.type_as (img1)

    return _ssim (img1, img2, window, window_size, channel, size_average)

class BatchGenerator:
    def __init__(self, batch_size, image_width):
        self.batch_size = batch_size
        self.image_size = image_width

    def load_full_data(self):
        dataset_test = FullDataSet()
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)
        self.data_map = []
        return test_loader

class FullDataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = data_map[1:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(tdd + value[0])

        # tactile_data = np.load(self.train_data_dir + value[1])
        # tactile_images = []
        # for tactile_data_sample in tactile_data:
        #     tactile_images.append(FullDataSet.create_image(tactile_data_sample, image_size=self.image_size))

        finger_images = []
        for image_name in np.load(tdd + value[1]):
            finger_images.append(np.load(tdd + image_name))

        # tilted_images = []
        # for image_name in np.load(tdd + value[2]):
        #     tilted_images.append(np.load(tdd + image_name))

        # front_images = []
        # for image_name in np.load(tdd + value[2]):
        #     # front_images.append(np.load(tdd + image_name))
        #     i = np.load(tdd + image_name)[:, :, :-1]
        #     front_images.append(i)

        # side_images = []
        # for image_name in np.load(tdd + value[3]):
        #     side_images.append(np.load(tdd + image_name))

        # experiment_number = np.load(tdd + value[3])
        # time_steps = np.load(tdd + value[4])
        return [robot_data.astype(np.float32), np.array(finger_images).astype(np.float32)]


class UniversalTester:
    def __init__(self, features):
        self.features = features
        self.list_of_p_measures = ["MAE", "MSE", "PSNR", "SSIM", "MAE_last", "MSE_last", "PSNR_last", "SSIM_last", "AVG_time"]

        saved_model = torch.load(features["model_save_path"] + features["model_save_name"] + features["model_name_save_appendix"])
        saved_model = '/home/gabrielle/PythonVirtual/ClusterScripts/Thesis/saved_models/FTVP128/model_24_02_2023_09_03/FTVP128_model'  #model_27_01_2023_15_33, model_28_01_2023_11_49, model_02_02_2023_16_29
        # self.features = saved_model['features']

        if features["model_name"] == "SVG": self.model = SVG(features = self.features)
        elif self.features["model_name"] == "TBF": self.model = TBF(features)
        elif self.features["model_name"] == "ATCVP": self.model = ATCVP(features)
        elif self.features["model_name"] == "FTVP64": self.model = FTVP64(features)
        elif self.features["model_name"] == "FTVP128": self.model = FTVP128(features)
        elif self.features["model_name"] == "ACVP": self.model = ACVP(features)
        elif self.features["model_name"] == "SIMVP": self.model = SIMVP(features, shape_in=[10, 3, 64, 64])

        self.test_features = features
        if self.model == SVG:
            self.model.load_model(full_model=saved_model)
        else:
            self.model.load_state_dict(torch.load(saved_model))
            self.model.to(self.features["device"])
        saved_model = None

        BG = BatchGenerator(self.features["batch_size"], self.features["image_width"])
        self.test_full_loader = BG.load_full_data()

        if self.test_features["quant_analysis"] == True:
            self.test_model()

    def test_model(self):
        batch_losses = []
        delta_t = []
        mae_frame_loss = []
        mse_frame_loss = []
        psnr_frame_loss = []

        # self.plot()
        # sys.exit()

        with torch.no_grad():
            if self.model == SVG:
                self.model.set_test()
            for index, batch_features in enumerate(self.test_full_loader):
                print(str(index) + "\r")
                # mae_frame_loss = []

                if batch_features[1].shape[0] == self.features["batch_size"]:                                               # messes up the models when running with a batch that is not complete
                    groundtruth_scene, predictions_scene, time_elapsed, groundtruth_tactile, prediction_tactile = self.format_and_run_batch(batch_features, test=True)              # run the model
                    if self.test_features["quant_analysis"] == True and prediction_tactile == 100:
                        # self.save_image(predictions_scene, groundtruth_scene, model_name=self.features["model_name"])
                        batch_losses.append(self.calculate_scores(predictions_scene, groundtruth_scene[self.features["n_past"]:], prediction_tactile))
                        delta_t.append(time_elapsed)                        # mae_frame_loss += self.calculate_mae(predictions_scene, groundtruth_scene[self.features["n_past"]:])
                        # mse_frame_loss += self.calculate_mse(predictions_scene, groundtruth_scene[self.features["n_past"]:])
                        # psnr_frame_loss += self.calculate_psnr(predictions_scene, groundtruth_scene[self.features["n_past"]:])
                        # self.save_plot(mae_frame_loss, index)
                    else:
                        batch_losses.append(self.calculate_scores(predictions_scene, groundtruth_scene[self.features["n_past"]:], prediction_tactile, groundtruth_tactile[self.features["n_past"]]))
                    # if index == 18:
                    #     self.image_comparison(predictions_scene, groundtruth_scene, index)
                    #
                    #     self.save_image(predictions_scene, groundtruth_scene, self.features["model_name"], index)
                    #     sys.exit()


        # mae_frame_loss = np.array(mae_frame_loss)
        # self.save_plot(mae_frame_loss, MAE=True, PSNR=False)
        # mse_frame_loss = np.array(mse_frame_loss)
        # self.save_plot(mse_frame_loss, MAE=False, PSNR=False)
        # psnr_frame_loss = np.array(psnr_frame_loss)
        # self.save_plot(psnr_frame_loss, MAE=False, PSNR=True)
        batch_losses = np.array(batch_losses)
        delta_t = np.array(delta_t)

        full_losses = [sum(batch_losses[:,0,i]) / batch_losses.shape[0] for i in range(batch_losses.shape[2])]
        last_ts_losses = [sum(batch_losses[:,1,i]) / batch_losses.shape[0] for i in range(batch_losses.shape[2])]
        avg_time = [sum(delta_t[:]) / delta_t.shape]

        full_losses = [float(i) for i in full_losses]
        last_ts_losses = [float(i) for i in last_ts_losses]
        avg_time = [float(i) for i in avg_time]

        if self.test_features["seen"]: data_save_path_append = "seen_"
        else:                          data_save_path_append = "unseen_"

        np.save(self.test_features["data_save_path"] + data_save_path_append  + "test_loss_scores.npy", batch_losses)
        np.save(self.test_features["data_save_path"] + data_save_path_append + "computational_time.npy", delta_t)
        lines = full_losses + last_ts_losses + avg_time
        with open (self.test_features["data_save_path"] + data_save_path_append  + "test_loss_scores.txt", 'w') as f:
            for index, line in enumerate(lines):
                f.write(self.list_of_p_measures[index] + ": " + str(line))
                f.write('\n')

        index = 3
        self.image_comparison(predictions_scene, groundtruth_scene, index)
        model_name = self.features["model_name"]
        self.save_image(predictions_scene, groundtruth_scene, model_name, index)
        # self.plot()

    def calculate_scores(self, prediction_scene, groundtruth_scene, n=7, prediction_tactile=None, groundtruth_tactile=None):
        scene_losses_full, scene_losses_last = [],[]
        for criterion in [nn.L1Loss(), nn.MSELoss(), PSNR(), SSIM(window_size=self.features["image_width"])]:  #, SSIM(window_size=self.image_width)]:
            scene_batch_loss_full = []
            for i in range(prediction_scene.shape[0]):
                scene_batch_loss_full.append(criterion(prediction_scene[i], groundtruth_scene[i]).cpu().detach().data)

            scene_losses_full.append(sum(scene_batch_loss_full) / len(scene_batch_loss_full))
            scene_losses_last.append(criterion(prediction_scene[-1], groundtruth_scene[-1]).cpu().detach().data)  # t+5

        return [scene_losses_full, scene_losses_last]

    def calculate_scores_fbf(self, prediction_scene, groundtruth_scene, n=3, prediction_tactile=None, groundtruth_tactile=None):
        scene_losses_full, scene_losses_last = [],[]
        for criterion in [nn.L1Loss(), nn.MSELoss(), PSNR(), SSIM(window_size=self.features["image_width"])]:  #, SSIM(window_size=self.image_width)]:
            scene_batch_loss_full = []
            for i in range(prediction_scene.shape[0][:n]):
                print(i)
                scene_batch_loss_full.append(criterion(prediction_scene[i], groundtruth_scene[i]).cpu().detach().data)



            scene_losses_full.append(sum(scene_batch_loss_full) / len(scene_batch_loss_full))
            scene_losses_last.append(criterion(prediction_scene[7], groundtruth_scene[7]).cpu().detach().data)  # t+5

        return [scene_losses_full, scene_losses_last]

    def calculate_mae(self, prediction_scene, groundtruth_scene, n=3, prediction_tactile=None, groundtruth_tactile=None):
        scene_losses_full, scene_losses_last = [],[]
        criterion = nn.L1Loss()  #, SSIM(window_size=self.image_width)]:
        scene_batch_loss_full = []
        for i in range(prediction_scene.shape[0]):
            scene_batch_loss_full.append(criterion(prediction_scene[i], groundtruth_scene[i]).cpu().detach().data)

        scene_losses_full.append(sum(scene_batch_loss_full) / len(scene_batch_loss_full))
        scene_losses_last.append(criterion(prediction_scene[-1], groundtruth_scene[-1]).cpu().detach().data)  # t+5

        return [scene_batch_loss_full]

    def calculate_psnr(self, prediction_scene, groundtruth_scene, n=3, prediction_tactile=None, groundtruth_tactile=None):
        scene_losses_full, scene_losses_last = [],[]
        criterion = PSNR()  #, SSIM(window_size=self.image_width)]:
        scene_batch_loss_full = []
        for i in range(prediction_scene.shape[0]):

            scene_batch_loss_full.append(criterion(prediction_scene[i], groundtruth_scene[i]).cpu().detach().data)

        scene_losses_full.append(sum(scene_batch_loss_full) / len(scene_batch_loss_full))
        scene_losses_last.append(criterion(prediction_scene[-1], groundtruth_scene[-1]).cpu().detach().data)  # t+5

        return [scene_batch_loss_full]

    def calculate_mse(self, prediction_scene, groundtruth_scene, n=3, prediction_tactile=None,
                      groundtruth_tactile=None):
        scene_losses_full, scene_losses_last = [], []
        criterion = nn.MSELoss()  # , SSIM(window_size=self.image_width)]:
        scene_batch_loss_full = []
        for i in range(prediction_scene.shape[0]):
            scene_batch_loss_full.append(criterion(prediction_scene[i], groundtruth_scene[i]).cpu().detach().data)

        scene_losses_full.append(sum(scene_batch_loss_full) / len(scene_batch_loss_full))
        scene_losses_last.append(criterion(prediction_scene[-1], groundtruth_scene[-1]).cpu().detach().data)  # t+5

        return [scene_batch_loss_full]

    def format_and_run_batch(self, batch_features, test):
        with torch.no_grad():
            mae, kld, mae_tactile, predictions = 100, 100, 100, 100
            images, predictions = [], []
            predictions_tactile, tactile = 100, 100
            if self.features["model_name"] == "SVG":
                images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
                mae, kld, predictions = self.model.run(scene=images, actions=action, test=test)

            elif self.features["model_name"] == "TBF":
                images = batch_features[2].permute(1, 0, 4, 3, 2).to(self.features["device"])
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
                mae, predictions = self.model.run(scene=images, actions=action, test=test)

            elif self.features["model_name"] == "ATCVP":
                # images = batch_features[2].permute(1, 0, 4, 3, 2).to(self.features["device"])
                touch = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
                since = time.time()
                mae, predictions = self.model.run(actions=action, touch=touch, test=test)
                time_elapsed = time.time() - since
                # print(f'forward time:{time_elapsed}')

            elif self.features["model_name"] == "FTVP64":
                images = batch_features[2].permute(1, 0, 4, 3, 2).to(self.features["device"])
                touch = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
                # since = time.time()
                mae, predictions = self.model.run(scene=images, actions=action, touch=touch, test=test)
                # time_elapsed = time.time() - since
                # print(f'forward time:{time_elapsed}')

            elif self.features["model_name"] == "FTVP128":
                # images = batch_features[2].permute(1, 0, 4, 3, 2).to(self.features["device"])
                touch = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
                since = time.time()
                mae, predictions = self.model.run(actions=action, touch=touch, test=test)
                time_elapsed = time.time() - since
                # print(f'forward time:{time_elapsed}')

            elif self.features["model_name"] == "ACVP":
                images = batch_features[2].permute(1, 0, 4, 3, 2).to(self.features["device"])
                touch = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
                mae, predictions = self.model.run(scene=images, actions=action, touch=touch, test=test)

            elif self.features["model_name"] == "SIMVP":
                images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
                mae, predictions = self.model.run(scene=images, actions=action, test=test)

            return touch, predictions, time_elapsed, tactile, predictions_tactile


    def image_comparison(self, pred, images, index):
        pred = pred.permute([1, 0, 4, 3, 2])
        images = images.permute([1, 0, 4, 3, 2])
        pred = pred.detach().cpu().numpy()
        images = images.detach().cpu().numpy()
        fig = plt.figure()
        rows = 2
        columns = 10
        j = 0
        k = 5
        w = 5
        for i in range(1, rows * columns + 1):
            if i <= 10:
                fig.add_subplot(rows, columns, i)
                plt.title(f"P {j + 1}")
                #plt.imshow((pred[0][j] * 255).astype(np.uint8))
                pred = np.clip(pred, 0, 1)
                plt.imshow((pred[0][j]))
                j = j + 1
            else:
                fig.add_subplot(rows, columns, i)
                plt.title(f"GT {w - 4}")
                #plt.imshow((images[0][w] * 255).astype(np.uint8))
                images = np.clip(images, 0, 1)
                plt.imshow((images[0][w]))
                w = w + 1
        # if train:
        #     fig.suptitle(f'{model_name}_TRAINING_AT_EPOCH_{t+1}')
        #     my_path = os.path.abspath(f'/home/gabrielle/Images/TRAINING_{model_name}/')
        #     my_file = f'{model_name}_TRAINING_AT_EPOCH_{t+1}.png'
        # else:
            fig.suptitle(f'{self.features["model_name"]}_TESTING')
            my_path = os.path.abspath(f'/home/gabrielle/Images/TESTING_{self.features["model_name"]}/')
            my_file = f'{self.features["model_name"]}_TESTING_strawberry_{index}.png'
        fig.savefig(os.path.join(my_path, my_file))
        plt.clf()
        # plt.show()

    def save_image(self, pred, images, model_name, index):
        pred = pred.permute([1, 0, 4, 3, 2])
        images = images.permute([1, 0, 4, 3, 2])
        pred = pred.detach().cpu().numpy()
        images = images.detach().cpu().numpy()
        my_path1 = os.path.abspath(f'/home/gabrielle/Images/TESTING_{model_name}/PRED_{index}/')
        my_path2 = os.path.abspath(f'/home/gabrielle/Images/TESTING_{model_name}/GT_{index}/')
        i = 0
        for i in range(len(pred[0])):
            fig1 = plt.figure()
            fig1.suptitle(f"{model_name}_PRED_frames:_{i + 1}")
            pred = np.clip(pred, 0, 1)
            plt.imshow((pred[0][i])) # [:,:,::-1]
            # plt.show()
            my_file1 = f'{model_name}_frames:_{i + 1}.png'
            fig1.savefig(os.path.join(my_path1, my_file1))
            fig2 = plt.figure()
            fig2.suptitle(f"{model_name}_GT_frames:_{i + 1}")
            images = np.clip(images, 0, 1)
            plt.imshow((images[0][i + 5]))
            my_file2 = f'{model_name}_frames:_{i + 1}.png'
            fig2.savefig(os.path.join(my_path2, my_file2))
        plt.clf()

    def save_plot(self, loss_array, MAE=True, PSNR=False):
        # path = os.path.abspath(f'/home/gabrielle/Images/TESTING_{self.features["model_name"]}/')
        # file = f'{self.features["model_name"]}_MAE_BATCH.png'
        f1_loss, f2_loss, f3_loss, f4_loss, f5_loss, f6_loss, f7_loss, f8_loss, f9_loss, f10_loss = [], [], [], [], [], [], [], [], [], []
        # f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = 0
        fs_MAE = []
        print(len(loss_array))
        for i in loss_array:
            # print(loss_array[i].shape)
            # print(loss_array[i][0].shape)
            f1_loss.append(i[0])
            f2_loss.append(i[1])
            f3_loss.append(i[2])
            f4_loss.append(i[3])
            f5_loss.append(i[4])
            f6_loss.append(i[5])
            f7_loss.append(i[6])
            f8_loss.append(i[7])
            f9_loss.append(i[8])
            f10_loss.append(i[9])

        f1 = sum(f1_loss) / len(f1_loss)
        f2 = sum(f2_loss) / len(f2_loss)
        f3 = sum(f3_loss) / len(f3_loss)
        f4 = sum(f4_loss) / len(f4_loss)
        f5 = sum(f5_loss) / len(f5_loss)
        f6 = sum(f6_loss) / len(f6_loss)
        f7 = sum(f7_loss) / len(f7_loss)
        f8 = sum(f8_loss) / len(f8_loss)
        f9 = sum(f9_loss) / len(f9_loss)
        f10 = sum(f10_loss) / len(f10_loss)

        fs_MAE = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
        fs_MAE = np.array(fs_MAE)
        if MAE == True and PSNR == False:
            np.save(self.test_features["data_save_path"] + "MAE.npy", fs_MAE)
        elif MAE == False and PSNR == False:
            np.save(self.test_features["data_save_path"] + "MSE.npy", fs_MAE)
        elif MAE == False and PSNR == True:
            np.save(self.test_features["data_save_path"] + "PSNR.npy", fs_MAE)

    def plot(self):
        path = os.path.abspath(f'/home/gabrielle/PythonVirtual/ClusterScripts/Thesis/saved_models/ATCVP/')
        file1 = f'ACTVP_vs_ACVP_MAE.png'
        file2 = f'ACTVP_vs_ACVP_MSE.png'
        path1 = '/home/gabrielle/PythonVirtual/ClusterScripts/Thesis/saved_models/ATCVP/model_02_02_2023_16_29/performance_data/MAE.npy'
        path2 = '/home/gabrielle/PythonVirtual/ClusterScripts/Thesis/saved_models/ACVP/model_03_02_2023_16_32/performance_data/MAE.npy'
        # mae_tac = np.load(path1)
        # mae_ntac = np.load(path2)
        mae_tac = [0.026565106, 0.05963047, 0.06687637, 0.07414619, 0.07979443, 0.08471672, 0.08885218, 0.09233002,
                   0.09542556, 0.09822814]
        mse_tac = [0.0087798782, 0.01462855, 0.01636316, 0.01963976, 0.0227733,  0.02526566, 0.02740126, 0.02922412,
                   0.03085461, 0.03232548]
        mae_ntac = [0.037289353, 0.05963518, 0.06634697, 0.07296637, 0.07826317, 0.08255757, 0.08618523, 0.0893674,
                    0.0922296,  0.09481674]
        mse_ntac = [0.0095235602, 0.01478265, 0.0171196,  0.01975721, 0.022146,   0.02424394, 0.02609848, 0.02777479,
                    0.02930655, 0.03070896]
        x = [1,2,3,4,5,6,7,8,9,10]
        plt.xticks(x)
        plt.plot(x, mae_tac, "-b", label='ACTVP')
        plt.plot(x, mae_ntac, "-r", label='ACVP')
        plt.legend(loc="upper left")
        # plt.plot(y_hat, color="blue")
        plt.grid()
        plt.title('MAE Comparison')
        plt.xlabel('frame')
        plt.ylabel('MAE')
        plt.savefig(os.path.join(path, file1))
        plt.clf()
        plt.xticks(x)
        plt.plot(x, mse_tac, "-b", label='ACTVP')
        plt.plot(x, mse_ntac, "-r", label='ACVP')
        plt.legend(loc="upper left")
        # plt.plot(y_hat, color="blue")
        plt.grid()
        plt.title('MSE Comparison')
        plt.xlabel('frame')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(path, file2))






@click.command()
@click.option('--model_name', type=click.Path(), default="FTVP128", help='Set name for prediction model, SVG, TBF, SIMVP, ...')
@click.option('--batch_size', type=click.INT, default=32, help='Batch size for training.')
@click.option('--lr', type=click.FLOAT, default = 0.0001, help = "learning rate")
@click.option('--beta1', type=click.FLOAT, default = 0.9, help = "Beta gain")
@click.option('--log_dir', type=click.Path(), default = 'logs/lp', help = "Not sure :D")
@click.option('--optimizer', type=click.Path(), default = 'adam', help = "what optimiser to use - only adam available currently")
@click.option('--niter', type=click.INT, default = 300, help = "")
@click.option('--seed', type=click.INT, default = 1, help = "")
@click.option('--image_width', type=click.INT, default = 128, help = "Size of scene image data")
@click.option('--dataset', type=click.Path(), default = 'Dataset3_MarkedHeavyBox', help = "name of the dataset")
@click.option('--n_past', type=click.INT, default = 5, help = "context sequence length")
@click.option('--n_future', type=click.INT, default = 10, help = "time horizon sequence length")
@click.option('--n_eval', type=click.INT, default = 15, help = "sum of context and time horizon")
@click.option('--prior_rnn_layers', type=click.INT, default = 3, help = "number of LSTMs in the prior model")
@click.option('--posterior_rnn_layers', type=click.INT, default = 3, help = "number of LSTMs in the posterior model")
@click.option('--predictor_rnn_layers', type=click.INT, default = 4, help = "number of LSTMs in the frame predictor model")
@click.option('--state_action_size', type=click.INT, default = 12, help = "size of action conditioning data")
@click.option('--z_dim', type=click.INT, default = 10, help = "number of latent variables to estimate")
@click.option('--beta', type=click.FLOAT, default = 0.0001, help = "beta gain")
@click.option('--data_threads', type=click.INT, default = 5, help = "")
@click.option('--num_digits', type=click.INT, default = 2, help = "")
@click.option('--last_frame_skip', type=click.Path(), default = 'store_true', help = "")
@click.option('--epochs', type=click.INT, default = 1, help = "number of epochs to run for ")
@click.option('--train_percentage', type=click.FLOAT, default = 0.9, help = "")
@click.option('--validation_percentage', type=click.FLOAT, default = 0.1, help = "")
@click.option('--criterion', type=click.Path(), default = "L1", help = "")
@click.option('--tactile_size', type=click.INT, default = 0, help = "size of tacitle frame - 48, if no tacitle data set to 0")
@click.option('--g_dim', type=click.INT, default = 256, help = "size of encoded data for input to prior")
@click.option('--rnn_size', type=click.INT, default = 256, help = "size of encoded data for input to frame predictor (g_dim = rnn-size)")
@click.option('--channels', type=click.INT, default = 3, help = "input channels")
@click.option('--out_channels', type=click.INT, default = 3, help = "output channels")
@click.option('--training_stages', type=click.Path(), default = "", help = "define the training stages - if none leave blank - available: 3part")
@click.option('--training_stages_epochs', type=click.Path(), default = "50,75,125", help = "define the end point of each training stage")
@click.option('--num_workers', type=click.INT, default = 20, help = "number of workers used by the data loader")
@click.option('--model_save_path', type=click.Path(), default = "/home/gabrielle/PythonVirtual/ClusterScripts/Thesis/saved_models/FTVP128/four/", help = "")
# @click.option('--train_data_dir', type=click.Path(), default = "/home/gabrielle/data/Pushing_Blocks/train_formatted/", help = "")
@click.option('--train_data_dir', type=click.Path(), default = "/home/gabrielle/Extreme SSD/data/Pushing_Strawberries/train_formatted_128/", help = "")
@click.option('--model_name_save_appendix', type=click.Path(), default = "", help = "What to add to the save file to identify the model as a specific subset (_1c= 1 conditional frame, GTT=groundtruth tactile data)")
@click.option('--tactile_encoder_hidden_size', type=click.INT, default = 0, help = "Size of hidden layer in tactile encoder, 200")
@click.option('--tactile_encoder_output_size', type=click.INT, default = 0, help = "size of output layer from tactile encoder, 100")
@click.option('--occlusion_test', type=click.Path(), default = "", help = "if you would like to train for occlusion")
@click.option('--occlusion_gain_per_epoch', type=click.FLOAT, default = 0.005, help = "increasing size of the occlusion block per epoch 0.1=(0.1 x MAX) each epoch")
@click.option('--occlusion_start_epoch', type=click.INT, default = 100, help = "size of output layer from tactile encoder, 100")
@click.option('--occlusion_max_size', type=click.FLOAT, default = 1.0, help = "max size of the window as a % of total size (0.5 = 50% of frame (32x32 squares in ))")
@click.option('--using_depth_data', type=click.BOOL, default = False, help = "if the image has depth included, set to True")
@click.option('--using_tactile_images', type=click.BOOL, default = False, help = "if the image has depth included, set to True")
@click.option('--early_stop_clock', type=click.INT, default = 5, help = "if the image has depth included, set to True")
@click.option('--model_stage', type=click.Path(), default="", help='what stage of model should you test? BEST, stage1 etc.')
@click.option('--model_folder_name', type=click.Path(), default="/home/gabrielle/PythonVirtual/ClusterScripts/Thesis/saved_models/FTVP128/model_24_02_2023_09_03/", help='Folder name where the model is stored') #model_27_01_2023_15_33, model_28_01_2023_11_49, model_02_02_2023_16_29
@click.option('--quant_analysis', type=click.BOOL, default=True, help='Perform quantitative analysis on the test data')
@click.option('--qual_analysis', type=click.BOOL, default=True, help='Perform qualitative analysis on the test data')
@click.option('--qual_tactile_analysis', type=click.BOOL, default=False, help='Perform qualitative analysis on the test tactile data')
@click.option('--test_sample_time_step', type=click.Path(), default="[1, 2, 10]", help='which time steps in prediciton sequence to calculate performance metrics for.')
@click.option('--model_name_save_appendix', type=click.Path(), default = "", help = "What to add to the save file to identify the model as a specific subset, _1c")
# @click.option('--test_data_dir', type=click.Path(), default = "/home/gabrielle/data/Pushing_Blocks/test_formatted/", help = "")
@click.option('--test_data_dir', type=click.Path(), default = "/home/gabrielle/data/test_formatted_single_128/", help = "")
# @click.option('--scaler_dir', type=click.Path(), default = "/home/gabrielle/data/Pushing_Blocks/filler_scaler/", help= "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--scaler_dir', type=click.Path(), default = "/home/gabrielle/data/push_black_128/filler_scaler_128/", help= "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--using_tactile_images', type=click.BOOL, default = False, help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--using_depth_data', type=click.BOOL, default = False, help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--seen', type=click.BOOL, default = True, help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--device', type=click.Path(), default = "cuda:0", help = "if the image has depth included, set to True")
def main(model_name, batch_size, lr, beta1, log_dir, optimizer, niter, seed, image_width, dataset,
         n_past, n_future, n_eval, prior_rnn_layers, posterior_rnn_layers, predictor_rnn_layers, state_action_size,
         z_dim, beta, data_threads, num_digits, last_frame_skip, epochs, train_percentage, validation_percentage,
         criterion, tactile_size, g_dim, rnn_size, channels, out_channels, training_stages, training_stages_epochs,
         num_workers, model_save_path, train_data_dir, model_name_save_appendix, tactile_encoder_hidden_size,
         tactile_encoder_output_size, occlusion_test, occlusion_gain_per_epoch, occlusion_start_epoch, occlusion_max_size,
         using_depth_data, using_tactile_images, early_stop_clock, model_stage, model_folder_name, quant_analysis, qual_analysis, qual_tactile_analysis, test_sample_time_step, test_data_dir, scaler_dir, seen, device):
    model_save_path = model_folder_name
    test_data_dir   = test_data_dir
    scaler_dir      = scaler_dir
    data_save_path  = model_save_path + "performance_data/"
    model_save_name = model_name + "_model"
    model_dir = model_save_path

    try:
        os.mkdir(data_save_path)
    except FileExistsError or FileNotFoundError:
        pass

    print(model_save_name)

    global data_map
    global tdd
    global uti
    global udd
    data_map = []
    tdd = test_data_dir
    uti = using_tactile_images
    udd = using_depth_data

    with open(test_data_dir + 'map.csv', 'r') as f:  # rb
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            data_map.append(row)

    features = {"lr": lr, "beta1": beta1, "batch_size": batch_size, "log_dir": log_dir,
                "optimizer": optimizer, "niter": niter, "seed": seed,
                "image_width": image_width, "channels": channels, "out_channels": out_channels, "dataset": dataset,
                "n_past": n_past, "n_future": n_future, "n_eval": n_eval, "rnn_size": rnn_size,
                "prior_rnn_layers": prior_rnn_layers, "model_dir": model_dir,
                "posterior_rnn_layers": posterior_rnn_layers, "predictor_rnn_layers": predictor_rnn_layers,
                "state_action_size": state_action_size, "z_dim": z_dim, "g_dim": g_dim, "beta": beta,
                "data_threads": data_threads, "num_digits": num_digits,
                "last_frame_skip": last_frame_skip, "epochs": epochs, "train_percentage": train_percentage,
                "validation_percentage": validation_percentage, "criterion": criterion, "model_name": model_name,
                "train_data_dir": train_data_dir,
                "training_stages": training_stages, "training_stages_epochs": training_stages_epochs,
                "tactile_size": tactile_size, "num_workers": num_workers,
                "model_save_path": model_save_path, "model_name_save_appendix": model_name_save_appendix,
                "tactile_encoder_hidden_size": tactile_encoder_hidden_size,
                "tactile_encoder_output_size": tactile_encoder_output_size,
                "occlusion_test": occlusion_test, "occlusion_gain_per_epoch": occlusion_gain_per_epoch,
                "occlusion_start_epoch": occlusion_start_epoch, "occlusion_max_size": occlusion_max_size,
                "early_stop_clock": early_stop_clock, "model_stage":model_stage, "model_folder_name":model_folder_name,
                "quant_analysis":quant_analysis, "qual_analysis":qual_analysis, "model_save_name":model_save_name,
                "qual_tactile_analysis":qual_tactile_analysis, "test_sample_time_step":test_sample_time_step,
                "test_data_dir":test_data_dir, "scaler_dir":scaler_dir,
                "using_tactile_images":using_tactile_images, "using_depth_data":using_depth_data,
                "data_save_path": data_save_path, "seen": seen, "device": device}

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # use gpu if available

    MT = UniversalTester(features)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    main()