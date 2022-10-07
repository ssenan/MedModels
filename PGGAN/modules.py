from ast import Add
from re import I
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.functional import interpolate
from utils import PixelNorm

class BaseGenBlock(nn.Module):
    '''
    Implementing base block for generator
    Args:
    in_channels: input channels in the block
    out_channels: output channels in the block
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__(BaseGenBlock, self).__init__()

        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, (4, 4), bias=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels,(4, 4), padding=1, bias=True)

        self.pixel_norm = PixelNorm
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        y = self.pixel_norm(y)
        y = self.lrelu(self.conv1(y))
        y = self.lrelu(self.conv2(y))
        y = self.pixel_norm(y)
        return y

#--------------------------------------------------------------

class AddGenBlock(nn.Module):
    '''
    Implementing base block for generator
    Args:
    in_channels: input channels in the block
    out_channels: output channels in the block
    '''

    def __init__(self, in_channels, out_channels) -> None:
        super(AddGenBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, (3, 3), padding=1, bias=True)

        self.pixel_norm = PixelNorm
        self.lrelu = nn.LeakyRelu(0.2)

    def forward(self, x):
        y = interpolate(x, scale_factor=2)
        y = self.lrelu(self.pixel_norm(self.conv1(y)))
        y = self.lrelu(self.pixel_norm(self.conv2(y)))
        return y

#--------------------------------------------------------------

class FinalBlockClasses(nn.Module):
    """
    Final block for discriminator model
    See projection model from -> https://arxiv.org/pdf/1802.05637.pdf

    Args:
    in_channels = number of input channels
    num_classes = specified number of classes for discrimination
    """

    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(in_channels + 1, in_channels, (3, 3, 3), padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, (4, 4), padding=1, bias=True)
        self.conv3 = nn.Conv3d(in_channels, 1, (1, 1), bias=True)

        self.label_embedder = nn.Embedding(num_classes, out_channels, max_norm=1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, labels):
        y = self.lrelu(self.conv1(x))
        y = self.lrelu(self.conv2(y))

        # Embedding labels
        labels = self.label_embedder(labels)

        # Inner product for label embeddings
        y_ = torch.squeeze(torch.squeeze(y, dim=-1), dim=-1)
        projection_score = (y_ * labels).sum(dim=-1)

        # Normal distribution score
        y = self.lrelu(self.conv3(y))

        # Total score calculation
        final_score = y.view(-1) + projection_score

        return final_score

#--------------------------------------------------------------
class AddDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, (3, 3, 3), padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, (3, 3, 3), padding=1, bias=True)
        self.downsampler = nn.AvgPool3d(3)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.lrelu(self.conv1(x))
        y = self.lrelu(self.conv2(y))
        y = self.downsampler(y)

        return y
