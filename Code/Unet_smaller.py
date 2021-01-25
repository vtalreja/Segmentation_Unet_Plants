import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import initialize_weights
import numpy as np


class UnetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UnetEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class UnetDecoder(nn.Module):

    def __init__(self, in_channels, featrures, out_channels):
        super(UnetDecoder, self).__init__()
        self.in_channels = in_channels
        self.features = featrures
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.features),
                                   nn.ReLU(),
                                   nn.Conv2d(self.features, self.features, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.features),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(self.features, self.out_channels, kernel_size=2, stride=2),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(
                                    nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        self.up1 = UnetDecoder(512, 256, 128)
        self.up2 = UnetDecoder(256, 128, 64)

        self.up3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))
        # nn.BatchNorm2d(64),
        # nn.ReLU())

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        en1 = self.down1(x)
        po1 = self.pool1(en1)
        en2 = self.down2(po1)
        po2 = self.pool2(en2)
        en3 = self.down3(po2)
        po3 = self.pool3(en3)
        en4 = self.down4(po3)
        # po4 = self.pool4(en4)

        c1 = self.center(en4)

        dec1 = self.up1(torch.cat([c1, en3], 1))
        dec2 = self.up2(torch.cat([dec1, en2], 1))
        dec3 = self.up3(torch.cat([dec2, en1], 1))
        # dec4 = self.up4(torch.cat([dec3, F.upsample_bilinear(en1, dec3.size()[2:])], 1))

        out = self.output(dec3)
        return self.final(out)