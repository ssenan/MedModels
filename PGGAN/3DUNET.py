"""
Refer to https://github.com/HiLab-git/SSL4MIS/blob/master/code/networks/unet_3D.py as reference
"""

import torch.nn as nn
import torch.nn.functional as F

from utils import UnetConv3D, UnetUpscale3D_CT 

class Unet3D(nn.Module):
    def __init__(self, feature_channels, class_number, in_channels):
        super().__init__()
        self.feature_channels = feature_channels
        self.class_number = class_number
        self.in_channels = in_channels

        # Downsampling 
        self.conv1 = UnetConv3D(self.in_channels, self.feature_channels[0], kernel_size = (
            3, 3, 3), padding= (1, 1, 1))
        self.averagepool1 = nn.AvgPool3d(kernel_size=(2, 2, 2))

        
        self.conv2 = UnetConv3D(self.feature_channels[0], self.feature_channels[1], kernel_size = (
            3, 3, 3), padding= (1, 1, 1))
        self.averagepool2 = nn.AvgPool3d(kernel_size=(2, 2, 2))


        self.conv3 = UnetConv3D(self.feature_channels[1], self.feature_channels[2], kernel_size = (
            3, 3, 3), padding= (1, 1, 1))
        self.averagepool3 = nn.AvgPool3d(kernel_size=(2, 2, 2))


        self.conv4 = UnetConv3D(self.feature_channels[2], self.feature_channels[3], kernel_size = (
            3, 3, 3), padding= (1, 1, 1))
        self.averagepool4 = nn.AvgPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3D(self.feature_channels[3], feature_channels[4], kernel_size=(
            3, 3, 3), padding=(1, 1, 1))

        # Upsampling
        self.up_concat4 = UnetUpscale3D_CT(feature_channels[4], feature_channels[3])
        self.up_concat3 = UnetUpscale3D_CT(feature_channels[3], feature_channels[2])
        self.up_concat2 = UnetUpscale3D_CT(feature_channels[2], feature_channels[1])
        self.up_concat1 = UnetUpscale3D_CT(feature_channels[1], feature_channels[0])


        # Final Layer
        self.output_conv = nn.Conv3d(feature_channels[0], class_number, 1) 

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        averagepool1 = self.averagepool1(conv1)

        conv2 = self.conv2(averagepool1)
        averagepool2 = self.averagepool2(conv2)

        conv3 = self.conv3(averagepool2)
        averagepool3 = self.averagepool3(conv3)

        conv4 = self.conv4(averagepool3)
        averagepool4 = self.averagepool4(conv4)

        center = self.center(averagepool4)
        center = self.dropout1(center)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.output_conv(up1)

        return final
     
    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p