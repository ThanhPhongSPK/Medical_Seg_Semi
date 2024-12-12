from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.uniform import Uniform

class Convolution_block(nn.Module):
    '''Two convolution layers with BatchNorm and Leadky Relu'''
    def __init__(self, in_channels, out_channels, dropout):
        super(Convolution_block, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward_pass(self, x):
        return self.double_conv(x)
    
    
class Downsample_block(nn.Module):
    """Downsampling followed by Convolution_block"""
    def __init__(self, in_channels, out_channels, dropout):
        super(Downsample_block, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Convolution_block(in_channels, out_channels, dropout)
        )
    def forward_pass(self, x):
        return self.maxpool_conv(x)
    
class Upsample_block(nn.Module):
    """Upsampling followed by Convolution_block"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout):
        super(Upsample_block, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Convolution_block(in_channels2 * 2, out_channels, dropout)
        
    def forward_pass(self, x1, x2):
        x1 = self.conv1x1(x1)
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_channels = self.params['in_channels']
        self.ft_channels = self.params['ft_channels']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_channels) == 5)
        
        self.in_conv = Convolution_block(
            self.in_channels, self.ft_channels[0], self.dropout[0])
        self.down1 = Downsample_block(
            self.ft_channels[0], self.ft_channels[1], self.dropout[1])
        self.donw2 = Downsample_block(
            self.ft_channels[1], self.ft_channels[2], self.dropout[2])
        self.down3 = Downsample_block(
            self.ft_channels[2], self.ft_channels[3], self.dropout[3])
        self.down4 = Downsample_block(
            self.ft_channels[3], self.ft_channels[4], self.dropout[4])
        
    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]
    
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_channels = self.params['in_channels']
        self.ft_channels = self.params['ft_channels']
        self.n_class = self.params['class_num']
        assert(len(self.ft_channels == 5 ))
        
        self.up1 = Upsample_block(
            self.ft_channels[4], self.ft_channels[3], self.ft_channels[3], dropout=0.0)
        self.up2 = Upsample_block(
            self.ft_channels[3], self.ft_channels[2], self.ft_channels[2], dropout=0.0)
        self.up3 = Upsample_block(
            self.ft_channels[2], self.ft_channels[1], self.ft_channels[1], dropout=0.0)
        self.up4 = Upsample_block(
            self.ft_channels[1], self.ft_channels[0], self.ft_channels[0], dropout=0.0)

        self.out_conv = nn.Conv2d(self.ft_channels[0], self.n_class, kernel_size=3, padding=1)
        
    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_last = self.up4(x, x0)
        output = self.out_conv(x_last)
        return output, x_last
    
class UNet(nn.Module):
    def __init__(self, in_channels, class_num):
        super(UNet, self).__init__()
        
        params = {'in_channels': in_channels,
                  'ft_channels': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.01, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'activation': 'relu'}
        
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for clss in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(clss), selector)
        
        for clss in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(clss), selector)
            
    def forward_projection_head(self, features):
        return self.projection_head(features)
    #return self.decoder(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        return output, features
    
class UNet_2d(nn.Module):
    def __init__(self, in_channels, class_num):
        super(UNet_2d, self).__init__()
        
        params = {'in_channels': in_channels,
                  'ft_channels': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'activation': 'relu'
                }
        
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.RelU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for clss in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(clss), selector)
        for clss in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(clss), selector)
            
    def forward_projection_head(self, features):
        return self.projection_head(features)
    
    def forward_prediction_head(self, features):
        return self.prediction_head(features)
    
    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        return output
    
class Sep_UNet_2d(nn.Module):
    def __init__(self, in_channels, class_num):
        super(Sep_UNet_2d, self).__init__()
        
        params = {'in_channels': in_channels,
                  'ft_channels': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'activation':'relu'
        }
        
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.pool = nn.Maxpool2d(3, stride=2)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNomrm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNomrm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for clss in range(4):
            selector = nn.Sequential(
                nn.Liner(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector' + str(clss), selector)
        
        for clss in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(clss), selector)
            
    def forward_projection_head(self, features):
        return self.projection_head(features)
    
    def forward_prediction_head(self, features):
        return self.prediction_head(features)
    
    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        feature = self.pool(feature[4])
        feature = self.pool(feature)
        return feature, output
    
class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_) # Save lambda_ to use in the backward pass
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors # Retrieve lambda_ from the saved context
        grad_output = grad_output.clone() # Clone the gradient to prevent in-place operations
        return - lambda_ * grad_output, None # Reverse the gradient by multiplying by -lambda_

class GradReverseLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(GradReverseLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x, lam)
    
class net_D(nn.Module):
    def __init__(self, b_size):
        super(net_D, self).__init__()
        self.b_size = b_size
        self.total_dim = self.b_size * 256 * 3 * 3
        self.model = nn.Sequential(
            nn.Linear(self.total_dim, int(self.total/2)),
            nn.Tanh(),
            nn.Linear(int(self.total/2), int(self.total_dim/4)),
            nn.Tanh(),
            nn.Linear(int(self.total_dim/4), 1),
            nn.Sigmoid()
        ) 
        
    def forward(self, x):
        x = x.view(1, -1)
        x = self.model(x)
        return x
    
class UNet_2dBCP(nn.Module):
    def __init__(self, in_channels, class_num):
        super(UNet_2dBCP, self).__init__()
        
        params = {'in_channels': in_channels,
                  'ft_channels': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'activation':'relu'
                  }
        
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        
    def forward(self, x):
        feature = self.encoder(x)
        output = self.encoder(feature)
        return output
    
class UNet_tse(nn.Module):
    def __init__(self, in_channels, class_num):
        super(UNet_tse, self).__init__()
        
        params = {'in_channels': in_channels,
                  'ft_channels': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'activation':'relu'
                  }
        
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for clss in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__settatrr__('contrastive_class_selector_' + str(clss), selector)
            
        for clss in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memmory' + str(clss), selector)
            
    def forward_projection_head(self, features):
        return self.projection_head(features)
    
    def forward_prediction_head(self, features):
        return self.prediction_head(features)
    
    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.encoder(feature)
        return output, features