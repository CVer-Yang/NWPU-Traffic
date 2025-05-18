"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import math
from functools import partial
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nonlinearity = partial(F.relu,inplace=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GlobalContext(nn.Module):
    def __init__(self, in_channels):
        super(GlobalContext, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
      
        y = self.fc(y).view(b, c, 1, 1)
        y = self.up(x*y)
      
        return y

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


import torch
import torch.nn as nn

class CNNTransformerDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, num_heads=8, num_layers=1):
        super(CNNTransformerDecoderBlock, self).__init__()

        # CNN 分支
        self.cnn_conv = nn.Conv2d(in_channels, in_channels//4 , kernel_size=3, padding=1)
        self.cnn_norm = nn.BatchNorm2d(in_channels//4)
        self.cnn_relu = nn.ReLU(inplace=True)
        self.down = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=4, stride=4, padding=0, groups=in_channels // 4)
        self.up = nn.ConvTranspose2d(in_channels//4, in_channels//4, kernel_size=4, stride=4, padding=0, groups=in_channels //4)
        # Transformer 分支
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels//4 , nhead=num_heads),
            num_layers=num_layers
        )

        # 门控聚合模块
        self.gate_conv = nn.Conv2d(in_channels//4+in_channels//4, in_channels//4 , kernel_size=1)
        self.gate_sigmoid = nn.Sigmoid()

        # 解码层
        self.deconv = nn.ConvTranspose2d(in_channels//4 , in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # 最终输出
        self.final_conv = nn.Conv2d(in_channels // 4, n_filters, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # CNN 分支
        x_cnn = self.cnn_conv(x)
        x_cnn = self.cnn_norm(x_cnn)
        x_cnn = self.cnn_relu(x_cnn)

        # Transformer 分支

        x_Trans = self.down(x_cnn)
        b, c, h, w = x_Trans.shape
        x_transformer = x_Trans.flatten(2)  # (B, C, H, W) -> (B, C, H*W)
        x_transformer = x_transformer.permute(2, 0, 1)  # (B, C, H*W) -> (H*W, B, C)
        x_transformer = self.transformer_encoder(x_transformer)
        x_transformer = x_transformer.permute(1, 2, 0)  # (H*W, B, C) -> (B, C, H*W)
        x_transformer = x_transformer.view(b, c , h, w)  # 恢复到 (B, C//4, H, W)
        x_transformer = self.up(x_transformer)

        # 门控聚合
        x_combined = torch.cat([x_cnn, x_transformer], dim=1)  # 将 CNN 和 Transformer 特征拼接
        gates = self.gate_conv(x_combined)  # 生成门控权重
        gates = self.gate_sigmoid(gates)  # 激活到 [0, 1] 范围

        # 按权重融合特征
        x = gates * x_cnn + (1 - gates) * x_transformer

        # 解码
        x = self.deconv(x)
        x = self.norm2(x)
        x = self.relu2(x)

        # 最后的卷积输出
        x = self.final_conv(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x




class ChannelSplit(nn.Module):
    def __init__(self, split_sizes):
        """
        初始化拆分模块
        :param split_sizes: 一个列表，定义每个拆分特征图的通道数。例如 [64, 32, 16, 8]。
        """
        super(ChannelSplit, self).__init__()
        self.split_sizes = split_sizes

    def forward(self, x):
        """
        拆分输入特征图
        :param x: 输入特征图，形状为 (B, C, H, W)
        :return: 一个列表，包含按照通道拆分的特征图
        """
        # 确保输入的通道数与拆分总通道数一致
        total_channels = sum(self.split_sizes)
        assert x.size(1) == total_channels, f"Input channels ({x.size(1)}) do not match split sizes ({total_channels})."

        # 按照通道维度拆分特征图
        split_features = torch.split(x, self.split_sizes, dim=1)
        return split_features

class PixelShuffleConcat(nn.Module):
    def __init__(self):
        super(PixelShuffleConcat, self).__init__()

        # 定义 PixelShuffle 操作

        self.pixel_shuffle2 = nn.PixelShuffle(2)  # 缩小分辨率的四分之一
        self.pixel_shuffle3 = nn.PixelShuffle(4)  # 缩小分辨率的八分之一
        self.pixel_shuffle4 = nn.PixelShuffle(8)  # 缩小分辨率的八分之一

        self.pixel_unshuffle2 = nn.PixelUnshuffle(2)  # 缩小分辨率的四分之一
        self.pixel_unshuffle3 = nn.PixelUnshuffle(4)  # 缩小分辨率的八分之一
        self.pixel_unshuffle4 = nn.PixelUnshuffle(8)  # 缩小分辨率的八分之一

        self.fuse = SpatialFrequencyAttentionFusion(channels=120)

    def forward(self, features):
        """
        features: 一个包含多尺度特征的列表，形状分别为：
            - e1: (B, 64, 256, 256)
            - e2: (B, 128, 128, 128)
            - e3: (B, 256, 64, 64)
            - e4: (B, 512, 32, 32)
        """
        e1, e2, e3, e4 = features
        # 调整 e1 到 32x32
        # 调整 e2 到 32x32
        e2 = self.pixel_shuffle2(e2)  # (B, 8, 32, 32)

        # 调整 e3 到 32x32
        e3 = self.pixel_shuffle3(e3)  # (B, 4, 32, 32)
        e4 = self.pixel_shuffle4(e4)  # (B, 4, 32, 32)

        # e4 保持不变，已经是 (B, 512, 32, 32)

        c1=e1.shape[1]
        c2=e2.shape[1]
        c3=e3.shape[1]
        c4=e4.shape[1]
        split_sizes = [c1, c2, c3, c4]  # 拆分通道数

        # 拼接所有特征
        x = torch.cat([e1, e2, e3, e4], dim=1)
        x = self.fuse(x)
        channel_split = ChannelSplit(split_sizes)
        f1, f2, f3, f4 = channel_split(x)
        f2 = self.pixel_unshuffle2(f2)
        f3 = self.pixel_unshuffle3(f3)
        f4 = self.pixel_unshuffle4(f4)

        return f1,f2,f3,f4


# 定义3D通道注意力（Squeeze-and-Excitation）模块
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # Squeeze操作：全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # Excitation操作：全连接层
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1)

        # Sigmoid用于生成注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: 全局平均池化
        squeeze = self.avg_pool(x)

        # Excitation: 两层全连接层
        excite = F.relu(self.fc1(squeeze))
        excite = self.sigmoid(self.fc2(excite))

        # Scale: 使用注意力权重调整特征图
        return x * excite+x


class MultiFeatureTransformer(nn.Module):
    def __init__(self):
        super(MultiFeatureTransformer, self).__init__()
        # 定义将每个特征图通道数转换为64的卷积层
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        self.conv3d = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.SEatt = SEBlock3D(channels=64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1)

    def forward(self, features):
        e1, e2, e3, e4 = features
        # 对四个输入特征图进行卷积处理，将通道数变为64
        e1 = self.conv1(e1)
        e2 = self.conv2(e2)
        e3 = self.conv3(e3)
        e4 = self.conv4(e4)

        # 获取特征图的尺寸
        B, C, H, W = e1.size()

        # 假设我们希望将H, W拆分为多个32x32大小的块
        block_size = 32
        assert H % block_size == 0 and W % block_size == 0, "H and W must be divisible by block size"

        # 使用unfold将特征图转换成多个块
        unfolded1 = e1.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        unfolded2 = e2.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        unfolded3 = e3.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        unfolded4 = e4.unfold(2, block_size, block_size).unfold(3, block_size, block_size)

        e1_reshaped = unfolded1.reshape(unfolded1.size(0), unfolded1.size(1), -1, unfolded1.size(4), unfolded1.size(5))
        e2_reshaped = unfolded2.reshape(unfolded2.size(0), unfolded2.size(1), -1, unfolded2.size(4), unfolded2.size(5))
        e3_reshaped = unfolded3.reshape(unfolded3.size(0), unfolded3.size(1), -1, unfolded3.size(4), unfolded3.size(5))
        e4_reshaped = unfolded4.reshape(unfolded4.size(0), unfolded4.size(1), -1, unfolded4.size(4), unfolded4.size(5))



        x_concat = torch.cat((e1_reshaped, e2_reshaped, e3_reshaped, e4_reshaped), dim=2)


        output = self.conv3d(x_concat)
        output = self.SEatt(output)
        print(output.shape)

        split_sizes = [64, 16, 4, 1]
        splits = torch.split(output, split_sizes, dim=2)

        # 步骤 2: 变形每个拆分后的张量
        reshaped_splits = []
        for i, split in enumerate(splits):
            if i == 0:
                reshaped_splits.append(split.reshape(4, 64, 8,-1, 32, 32))
            elif i == 1:
                reshaped_splits.append(split.reshape(4, 64, 4,-1, 32, 32))
            elif i == 2:
                reshaped_splits.append(split.reshape(4, 64, 2,-1, 32, 32))
            else:  # i == 3
                reshaped_splits.append(split.reshape(4, 64, 1,-1, 32, 32))

        reshaped_splits[0] = reshaped_splits[0].permute(0, 1, 2, 4, 3, 5)  # 调整后形状为 [4, 128, 8, 32, 8, 32]
        reshaped_splits[0] = reshaped_splits[0].reshape(4, 64, 256,-1)  # 形状变为 [4, 128, 256, 256]
        f1 = self.conv5(reshaped_splits[0])
        reshaped_splits[1] = reshaped_splits[1].permute(0, 1, 2, 4, 3, 5)  # 调整后形状为 [4, 128, 8, 32, 8, 32]
        reshaped_splits[1] = reshaped_splits[1].reshape(4, 64, 128, -1)  # 形状变为 [4, 128, 256, 256]
        f2 = self.conv6(reshaped_splits[1])
        reshaped_splits[2] = reshaped_splits[2].permute(0, 1, 2, 4, 3, 5)  # 调整后形状为 [4, 128, 8, 32, 8, 32]
        reshaped_splits[2] = reshaped_splits[2].reshape(4, 64, 64, -1)  # 形状变为 [4, 128, 256, 256]
        f3 = self.conv7(reshaped_splits[2])
        reshaped_splits[3] = reshaped_splits[3].permute(0, 1, 2, 4, 3, 5)  # 调整后形状为 [4, 128, 8, 32, 8, 32]
        reshaped_splits[3] = reshaped_splits[3].reshape(4, 64, 32, -1)  # 形状变为 [4, 128, 256, 256]
        f4 = self.conv8(reshaped_splits[3])
        # 聚合操作：拼接四个展开的特征图

        return f1,f2,f3,f4

class SpatialFrequencyAttentionFusion(nn.Module):
    def __init__(self, channels):
        super(SpatialFrequencyAttentionFusion, self).__init__()

        # 空间域注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),  # 输出单通道注意力图
            nn.Sigmoid()
        )

        # 频域注意力
        self.frequency_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),  # 输出与输入通道一致
            nn.Sigmoid()
        )

        # 融合后的卷积
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        输入：
        x: (B, C, H, W)

        输出：
        融合后的特征图 (B, C, H, W)
        """
        # 空间域注意力
        spatial_weights = self.spatial_attention(x)  # (B, 1, H, W)
        spatial_out = x * spatial_weights  # 应用空间域注意力

        # 转换到频域并计算频域注意力
        freq_x = torch.fft.fft2(x, norm='ortho')  # 转到频域
        freq_x = torch.fft.fftshift(freq_x, dim=(-2, -1))  # 频谱中心化
        freq_amplitude = torch.abs(freq_x)  # 取频谱幅值
        freq_attention = self.frequency_attention(freq_amplitude)  # 计算频域注意力
        freq_x = freq_x * freq_attention  # 应用频域注意力
        freq_out = torch.fft.ifft2(freq_x, norm='ortho').real  # 回到空间域

        # 特征融合
        fused_out = spatial_out + freq_out  # 简单相加融合
        fused_out = self.fusion_conv(fused_out)  # 融合后卷积

        return fused_out


class CSPNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CSPNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.FS_Fuse = PixelShuffleConcat()
        self.CS_Fuse = MultiFeatureTransformer()

        self.decoder4 = CNNTransformerDecoderBlock(filters[3], filters[2])
        self.decoder3 = CNNTransformerDecoderBlock(filters[2], filters[1])
        self.decoder2 = CNNTransformerDecoderBlock(filters[1], filters[0])
        self.decoder1 = CNNTransformerDecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image):

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        feature_list = [e1,e2,e3,e4]
        CS1,CS2,CS3,CS4 = self.FS_Fuse(feature_list)
        #print(e5.shape)
        CC1,CC2,CC3,CC4 = self.CS_Fuse(feature_list)
        print(CS2.shape)
        print(CC2.shape)
        # Decoder
        e4 = e4+CS4+CC4
        d4 = self.decoder4(e4) + CS3+CC3
        d3 = self.decoder3(d4) + CS2+CC2
        d2 = self.decoder2(d3) + CS1+CC1
        d1 = self.decoder1(d2) + x

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
