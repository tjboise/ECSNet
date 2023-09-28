import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.helpers import initialize_weights
from itertools import chain




class ECSBlock(nn.Module):
    def __init__(self, inchannel, outchannel=None, dilation=1, downsample=False, proj_ratio=4,
                 upsample=False, regularize=True, p_drop=None, use_prelu=True):
        super(ECSBlock, self).__init__()

        self.padding = 0
        self.upsampling = upsample
        self.downsampling = downsample
        if outchannel is None: outchannel = inchannel
        else: self.padding = outchannel - inchannel

        if regularize: assert p_drop is not None

        inter_channels = inchannel // proj_ratio

        self.conv1 = nn.Conv2d(inchannel, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.p_relu22 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(inter_channels, outchannel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu44 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x, indices=None, output_size=None):
        # Main branch
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.p_relu22(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)

        # When the input dim is odd, we might have a mismatch of one pixel
        if identity.size() != x.size():
            pad = (identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0)
            x = F.pad(x, pad, "constant", 0)

        x += identity
        x = self.prelu44(x)

        return x


class Parallel_part(nn.Module):
    def __init__(self, inchannel, outchannel=None, dilation=1, proj_ratio=4,
                  regularize=True, p_drop=None, use_prelu=True):
        super(Parallel_part, self).__init__()

        self.padding = 0

        if outchannel is None: outchannel = inchannel
        else: self.padding = outchannel - inchannel

        if regularize: assert p_drop is not None
        inter_channels = inchannel // proj_ratio

        # Main
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Bottleneck

        self.conv1 = nn.Conv2d(inchannel, inter_channels, 2, stride=2, bias=False)

        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.p_relu22 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(inter_channels, outchannel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu44 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x, indices=None, output_size=None):
        # Main branch
        identity = x

        identity, idx = self.pool(identity)


        if self.padding > 0:
            extras = torch.zeros((identity.size(0), self.padding, identity.size(2), identity.size(3)))
            if torch.cuda.is_available(): extras = extras.cuda(0)
            identity = torch.cat((identity, extras), dim = 1)

        # Bottleneck
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.p_relu22(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)

        # When the input dim is odd, we might have a mismatch of one pixel
        if identity.size() != x.size():
            pad = (identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0)
            x = F.pad(x, pad, "constant", 0)

        x += identity
        x = self.prelu44(x)

        return x, idx



class part2(nn.Module):
    def __init__(self, inchannel, outchannel=None, dilation=1,  proj_ratio=4,
                  regularize=True, p_drop=None, use_prelu=True):
        super(part2, self).__init__()

        self.padding = 0

        if outchannel is None: outchannel = inchannel
        else: self.padding = outchannel - inchannel

        if regularize: assert p_drop is not None

        inter_channels = inchannel // proj_ratio

        # Main

        self.spatil_convlution = nn.Conv2d(inchannel, outchannel, 1, bias=False)
        self.batchnorm_up = nn.BatchNorm2d(outchannel)
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)



        self.conv1 = nn.Conv2d(inchannel, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)



        self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.p_relu22 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(inter_channels, outchannel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu44 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x, indices=None, output_size=None):
        # Main branch
        identity = x

        assert (indices is not None) and (output_size is not None)
        identity = self.batchnorm_up(self.spatil_convlution(identity))
        if identity.size() != indices.size():
            pad = (indices.size(3) - identity.size(3), 0, indices.size(2) - identity.size(2), 0)
            identity = F.pad(identity, pad, "constant", 0)
        identity = self.unpooling(identity, indices=indices)

        if self.padding > 0:
            extras = torch.zeros((identity.size(0), self.padding, identity.size(2), identity.size(3)))
            if torch.cuda.is_available(): extras = extras.cuda(0)
            identity = torch.cat((identity, extras), dim = 1)

        # Bottleneck
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.p_relu22(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)

        # When the input dim is odd, we might have a mismatch of one pixel
        if identity.size() != x.size():
            pad = (identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0)
            x = F.pad(x, pad, "constant", 0)

        x += identity
        x = self.prelu44(x)

        return x


class middle_part(nn.Module):
    def __init__(self, inchannel, outchannel=None,  proj_ratio=4,
                 regularize=True, p_drop=None, use_prelu=True):
        super(middle_part, self).__init__()

        self.padding = 0

        if outchannel is None: outchannel = inchannel
        else: self.padding = outchannel - inchannel

        if regularize: assert p_drop is not None

        inter_channels = inchannel // proj_ratio


        self.conv1 = nn.Conv2d(inchannel, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)


        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU() if use_prelu else nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(5,1), padding=(2,0)),
        )

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.p_relu22 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(inter_channels, outchannel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu44 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x, indices=None, output_size=None):
        # Main branch
        identity = x

        if self.padding > 0:
            extras = torch.zeros((identity.size(0), self.padding, identity.size(2), identity.size(3)))
            if torch.cuda.is_available(): extras = extras.cuda(0)
            identity = torch.cat((identity, extras), dim = 1)

        # Bottleneck
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.p_relu22(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)

        # When the input dim is odd, we might have a mismatch of one pixel
        if identity.size() != x.size():
            pad = (identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0)
            x = F.pad(x, pad, "constant", 0)

        x += identity
        x = self.prelu44(x)

        return x


class ECSNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(ECSNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels, 16 - in_channels, 3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16)

        self.ECSblock10 = Parallel_part(16, 64, p_drop=0.01)
        self.ECSblock11 = ECSBlock(64, p_drop=0.01)
        self.ECSblock12 = ECSBlock(64, p_drop=0.01)
        self.ecsnet13 = ECSBlock(64, p_drop=0.01)
        self.ecsblock14 = ECSBlock(64, p_drop=0.01)
        self.ecsblovk15 = Parallel_part(64, 128, p_drop=0.1)

        self.ecsblovk = ECSBlock(128, p_drop=0.1)
        self.ecsblock17 = ECSBlock(128, dilation=2, p_drop=0.1)
        self.ecsblock18 = middle_part(128, p_drop=0.1)
        self.ecsblovk19 = ECSBlock(128, dilation=4, p_drop=0.1)
        self.ecsblovk20 = ECSBlock(128, p_drop=0.1)
        self.ecsblovk21 = ECSBlock(128, dilation=8, p_drop=0.1)
        self.ecsblovk22 = middle_part(128,  p_drop=0.1)
        self.ecsblovk23 = ECSBlock(128, dilation=16, p_drop=0.1)
        self.ecsblock24 = part2(128, 64, p_drop=0.1, use_prelu=False)
        self.ecsblock25 = ECSBlock(64, p_drop=0.1, use_prelu=False)
        self.ecsblock26 = ECSBlock(64, p_drop=0.1, use_prelu=False)
        self.ecsblock27 = part2(64, 16,  p_drop=0.1, use_prelu=False)
        self.ecsnet51 = ECSBlock(16, p_drop=0.1, use_prelu=False)
        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, padding=1,
                                           output_padding=1, stride=2, bias=False)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True), )
        initialize_weights(self)
        if freeze_bn: self.freeze_bn()

    def forward(self, x):

        x = torch.cat((self.pool(x), self.conv(x)), dim=1)
        x = self.bn(x)
        x = self.prelu(x)


        sz1 = x.size()
        x, indices1 = self.ECSblock10(x)
        x = self.ECSblock11(x)
        x = self.ECSblock12(x)
        x = self.ecsnet13(x)
        x = self.ecsblock14(x)
        sz2 = x.size()
        x, indices2 = self.ecsblovk15(x)
        x = self.ecsblovk(x)
        x = self.ecsblock17(x)
        x = self.ecsblock18(x)
        x = self.ecsblovk19(x)
        x = self.ecsblovk20(x)
        x = self.ecsblovk21(x)
        x = self.ecsblovk22(x)
        x = self.ecsblovk23(x)

        x = self.ecsblock24(x, indices=indices2, output_size=sz2)
        x = self.ecsblock25(x)
        x = self.ecsblock26(x)

        x = self.conv_out(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

if __name__ == "__main__":
    net = ECSNet(num_classes=1, in_channels=1)
    print(net)