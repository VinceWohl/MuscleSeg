# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:19:42 2023

@author: tp-vincentw

This script builds up Model B which is a UNet for 3D semantic segmentation of:
        - 13 muscle compartments in thigh MR images
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.doubleconv = nn.Sequential(
            # first convolution
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False, groups=2),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            # second convolution
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False, groups=2),
            nn.BatchNorm3d(out_channels),
        )
        self.lintransf = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        r = self.lintransf(x)
        x = self.doubleconv(x)
        x = self.relu(x + r)
        return x

    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_s, ch_out_int):
        super(AttentionBlock, self).__init__()
        # transfromation for x after up conv
        self.lintransf_x = nn.Sequential(
            nn.Conv3d(in_channels_x, ch_out_int, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, bias=False),
            nn.BatchNorm3d(ch_out_int)
        )
        # transformation for skip connection
        self.lintransf_s = nn.Sequential(
            nn.Conv3d(in_channels_s, ch_out_int, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, bias=False),
            nn.BatchNorm3d(ch_out_int)
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv3d(ch_out_int, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, bias=False),
            nn.BatchNorm3d(1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, skip):
        x = self.lintransf_x(x)
        skip1 = self.lintransf_s(skip)
        r = self.relu(x + skip1)
        p = self.psi(r)
        s = self.sigmoid(p)
        return skip*s
    

####################################################################################################################################################################
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()

        self.resblocks_down = nn.ModuleList()   # Residual blocks for encoder path
        self.attblocks = nn.ModuleList()        # Attention blocks fot decoder path
        self.resblocks_up = nn.ModuleList()     # Residual blocks for decoder path
        self.maxpool = nn.ModuleList()          # Max pooling operations                    
        self.upsample = nn.ModuleList()         # Transposed convolution operations
        '''
        ########################################################################################################
        Encoder path
        '''
        level = 0
        for feature in features:
            
            self.resblocks_down.append(ResidualBlock(in_channels, feature))
            in_channels = feature
            
            if level == 0 or level == -1:    
                down = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))  # max pooling for the higher two levels
            else:
                down = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # max pooling for the deeper two levels
            self.maxpool.append(down)
            level -= 1
        '''
        ########################################################################################################
        Bottleneck
        '''
        self.bottleneck = ResidualBlock(features[-1], features[-1]*2)
        '''
        ########################################################################################################
        Decoder path
        '''
        level = 0
        for feature in reversed(features):
            
            if level == 0 or level == 1:
                up = nn.Sequential(
                    nn.ConvTranspose3d(feature*2, feature, kernel_size=(2, 2, 2), stride=(2, 2, 2)),   # transposed conv for the deeper two levels
                    nn.BatchNorm3d(feature),
                    nn.LeakyReLU(inplace=True)
                )
            else:
                up = nn.Sequential(
                    nn.ConvTranspose3d(feature*2, feature, kernel_size=(2, 2, 1), stride=(2, 2, 1)),   # transposed conv for the higher two levels
                    nn.BatchNorm3d(feature),
                    nn.LeakyReLU(inplace=True),
                )
            self.upsample.append(up)
            level += 1
            
            self.attblocks.append(AttentionBlock(feature, feature, feature//2))
            self.resblocks_up.append(ResidualBlock(feature*2, feature))
        '''
        ########################################################################################################
        Final convolution
        '''
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=(1, 1, 1))


    def forward(self, x):
        skip_connections = []
        '''
        Go encoder path
        '''
        for idx in range(len(self.resblocks_down)):
            x = self.resblocks_down[idx](x)
            skip_connections.append(x) # append the output to skip connections list
            x = self.maxpool[idx](x)

        '''
        Go bottleneck
        '''
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # transverse skip connections

        '''
        Go decoder path + skip connections
        '''
        for idx in range(len(self.resblocks_up)):
            skip = skip_connections[idx]
            x = self.upsample[idx](x)

            skip = self.attblocks[idx](x, skip)
            x = torch.cat((skip, x), dim=1) # concatenate skip connection
            x = self.resblocks_up[idx](x)
            
        '''
        Apply final convolution
        '''
        x = self.final_conv(x)
        
        return torch.softmax(x, dim=1) # output: probability map as torch.tensor (B, N, H, W, D)
    
    
# ### test
# img =  torch.randn(1, 2, 368, 256, 32)
# print(f'Input: {img.shape}')

# model = UNET(in_channels=2, out_channels=3)

# pred = model(img)
# print(f'Output: {pred.shape}')

        #     print(f'Encoder: {x.shape}')
        # print(f'Bottleneck: {x.shape}')
        #     print(f'Decoder: skip:{skip.shape}, x:{x.shape}')