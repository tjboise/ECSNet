import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.helpers import initialize_weights
from itertools import chain

class InitalBlock(nn.Module):
    def __init__(self, in_channels, use_prelu=True):
        super(InitalBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels, 16 - in_channels, 3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16) if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x):   
        x = torch.cat((self.pool(x), self.conv(x)), dim=1)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class ECSBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, activation=None, dilation=1, downsample=False, proj_ratio=4, 
                        upsample=False, asymetric=False, regularize=True, p_drop=None, use_prelu=True):
        super(ECSBlock, self).__init__()

        self.pad = 0
        self.upsample = upsample
        self.downsample = downsample
        if out_channels is None: out_channels = in_channels
        else: self.pad = out_channels - in_channels

        if regularize: assert p_drop is not None
        if downsample: assert not upsample
        elif upsample: assert not downsample
        inter_channels = in_channels//proj_ratio

        # Main
        if upsample:
            self.spatil_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn_up = nn.BatchNorm2d(out_channels)
            self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        elif downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Bottleneck
        if downsample: 
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        if asymetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(1,5), padding=(0,2)),
                nn.BatchNorm2d(inter_channels),
                nn.PReLU() if use_prelu else nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(5,1), padding=(2,0)),
            )
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu_out = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x, indices=None, output_size=None):
        # Main branch
        identity = x
        if self.upsample:
            assert (indices is not None) and (output_size is not None)
            identity = self.bn_up(self.spatil_conv(identity))
            if identity.size() != indices.size():
                pad = (indices.size(3) - identity.size(3), 0, indices.size(2) - identity.size(2), 0)
                identity = F.pad(identity, pad, "constant", 0)
            identity = self.unpool(identity, indices=indices)#, output_size=output_size)
        elif self.downsample:
            identity, idx = self.pool(identity)


        if self.pad > 0:
            extras = torch.zeros((identity.size(0), self.pad, identity.size(2), identity.size(3)))
            if torch.cuda.is_available(): extras = extras.cuda(0)
            identity = torch.cat((identity, extras), dim = 1)

        # Bottleneck
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
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
        x = self.prelu_out(x)

        if self.downsample:
            return x, idx
        return x



class ECSNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(ECSNet, self).__init__()
        self.initial = InitalBlock(in_channels)

        self.ECSblock10 = ECSBlock(16, 64, downsample=True, p_drop=0.01)
        self.ECSblock11 = ECSBlock(64, p_drop=0.01)
        self.ECSblock12 = ECSBlock(64, p_drop=0.01)
        self.ecsnet13 = ECSBlock(64, p_drop=0.01)
        self.ecsblock14 = ECSBlock(64, p_drop=0.01)
        self.ecsblovk15 = ECSBlock(64, 128, downsample=True, p_drop=0.1)
        self.ecsblovk = ECSBlock(128, p_drop=0.1)
        self.ecsblock17 = ECSBlock(128, dilation=2, p_drop=0.1)
        self.ecsblock18 = ECSBlock(128, asymetric=True, p_drop=0.1)
        self.ecsblovk19 = ECSBlock(128, dilation=4, p_drop=0.1)
        self.ecsblovk20 = ECSBlock(128, p_drop=0.1)
        self.ecsblovk21 = ECSBlock(128, dilation=8, p_drop=0.1)
        self.ecsblovk22 = ECSBlock(128, asymetric=True, p_drop=0.1)
        self.ecsblovk23 = ECSBlock(128, dilation=16, p_drop=0.1)
        self.ecsblock24 = ECSBlock(128, 64, upsample=True, p_drop=0.1, use_prelu=False)
        self.ecsblock25 = ECSBlock(64, p_drop=0.1, use_prelu=False)
        self.ecsblock26 = ECSBlock(64, p_drop=0.1, use_prelu=False)
        self.ecsblock27 = ECSBlock(64, 16, upsample=True, p_drop=0.1, use_prelu=False)
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
        x = self.initial(x)

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

