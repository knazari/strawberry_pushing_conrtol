# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn
# from ACTVP import ConvLSTMCell
import torch.optim as optim

import time

from itertools import cycle

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, device):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, dtype=torch.double).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, dtype=torch.double).to(self.device))

class Model(nn.Module):
    def __init__(self, features):
        super(Model, self).__init__()
        self.features = features
        self.device = features["device"]
        self.context_frames = features["n_past"]
        self.n_future = features["n_future"]
        self.model_dir = features["model_dir"]
        self.model_name_save_appendix = features["model_name_save_appendix"]
        self.convlstm1 = ConvLSTMCell(input_dim=268, hidden_dim=268, kernel_size=(3, 3), bias=True, device=self.device).to(self.device)
        self.convlstm2 = ConvLSTMCell(input_dim=268, hidden_dim=268, kernel_size=(3, 3), bias=True, device=self.device)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv23 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv34 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.upconv1 = nn.Conv2d(in_channels=268, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.upconv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.upconv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.upconv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=4)
        self.relu1 = nn.ReLU().to(self.device)
        self.relu2 = nn.ReLU().to(self.device)
        self.upsample1 = nn.Upsample(scale_factor=1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=4)
        self.tanh = nn.Tanh()

    def run(self, scene, actions, touch, test=False):
        
        self.batch_size = actions.shape[1]  # 1
        state = actions[0]
        hidden_1, cell_1 = self.convlstm1.init_hidden(batch_size=self.batch_size, image_size=(8, 8))
        hidden_2, cell_2 = self.convlstm2.init_hidden(batch_size=self.batch_size, image_size=(8, 8))
        outputs = []
        for index, (sample_scene, sample_action, sample_touch) in enumerate(zip(scene[0:-1], actions[1:], cycle(touch[:-1]))):
            if index > self.context_frames - 1:
                out_touch1 = self.maxpool1(self.relu1(self.conv1(output)))
                out_touch2 = self.maxpool1(self.relu1(self.conv12(out_touch1)))
                out_touch3 = self.maxpool2(self.relu1(self.conv23(out_touch2)))
                out_touch4 = self.maxpool1(self.relu1(self.conv34(out_touch3)))

                state_action = torch.cat((state, sample_action), 1)
                robot_and_touch = torch.cat((torch.cat(8 * [torch.cat(8 * [state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), out_touch4), 1)
                # LSTM Chain
                hidden_1, cell_1 = self.convlstm1(input_tensor=robot_and_touch.float(),
                                                  cur_state=[hidden_1, cell_1])
                hidden_2, cell_2 = self.convlstm2(input_tensor=hidden_1, cur_state=[hidden_2, cell_2])
                # UPSAMPLING
                up1 = self.upsample2(self.relu2(self.upconv1(hidden_2)))
                up2 = self.upsample2(self.relu2(self.upconv2(up1)))
                up3 = self.upsample2(self.relu2(self.upconv3(up2)))
                up4 = self.upsample3(self.relu2(self.upconv4(up3)))
                skip_connection_added = torch.cat((up4, output.float()), 1)
                output = self.conv2(skip_connection_added)
                output = self.tanh(output)

                outputs.append(output)

            else:
                # 
                out_touch1 = self.maxpool1(self.relu1(self.conv1(sample_touch.view(1, 3, 256, 256))))
                out_touch2 = self.maxpool1(self.relu1(self.conv12(out_touch1)))
                out_touch3 = self.maxpool2(self.relu1(self.conv23(out_touch2)))
                out_touch4 = self.maxpool1(self.relu1(self.conv34(out_touch3)))
                # computation time of 4 conv layers: 0.25 sec
                #
                state_action = torch.cat((state, sample_action.view(1, -1)), 1)
                robot_and_touch = torch.cat((torch.cat(8 * [torch.cat(8 * [state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), out_touch4), 1)
                # computation time of concat layers: 0.0001 sec
                
                # LSTM Chain
                #
                hidden_1, cell_1 = self.convlstm1(input_tensor=robot_and_touch.float(), cur_state=[hidden_1, cell_1])
                hidden_2, cell_2 = self.convlstm2(input_tensor=hidden_1, cur_state=[hidden_2, cell_2])
                # computation time of 2 convlstm layers: 0.04 sec
                
                # UPSAMPLING
                #
                up1 = self.upsample2(self.relu2(self.upconv1(hidden_2)))
                up2 = self.upsample2(self.relu2(self.upconv2(up1)))
                up3 = self.upsample2(self.relu2(self.upconv3(up2)))
                up4 = self.upsample3(self.relu2(self.upconv4(up3)))
                # computation time of 4 upconv layers: 0.07 sec
                #
                skip_connection_added = torch.cat((up4, sample_touch.view(1, 3, 256, 256).float()), 1)
                output = self.conv2(skip_connection_added)
                # computation time of two last layers: 0.04 sec

                last_output = output

        outputs = [last_output] + outputs

        return torch.stack(outputs)


