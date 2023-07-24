import torch
from torch import nn
import torch.nn.functional as F


'''
Implementaci'on de Unet basado en: https://github.com/jaxony/unet-pytorch
con algunas modificaciones propias (se quita la 'ultima capa conv1x1 del decoder)
'''

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):  

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups
    )

def upconv2x2(in_channels, out_channels):

    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2
    )

def conv1x1(in_channels, out_channels, groups=1):

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1
    )

class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True, batchnorm=False):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.batchnorm = batchnorm

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.batchnorm:
            self.bn = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.bn(x)
        x = F.relu(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, batchnorm=False):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm

        self.upconv = upconv2x2(self.in_channels, self.out_channels)

        self.conv1 = conv3x3(2*self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.batchnorm:
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.bn(x)
        x = F.relu(x)
        return x


class Unet(nn.Module):

    def __init__(self, num_classes, in_channels=1, depth=3, start_filts=64, batchnorm=False):

        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(Unet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.batchnorm = batchnorm
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling, batchnorm=self.batchnorm)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, batchnorm=self.batchnorm)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x

