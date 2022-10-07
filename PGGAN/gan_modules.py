import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)

class genDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        layers = [nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)]
        layers.append(PixelNorm())
        layers.append(nn.LeakyReLU(0.2))

        self.layer = nn.Sequential(*layers)
        
    
    def forward(self, input):
        out = self.layer(input)
        return out

class genConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
        layers.append(PixelNorm())
        layers.append(nn.LeakyReLU(0.2))

        self.layer = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layer(input)
        return out 

class disConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__() 
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
        layers.append(PixelNorm())
        layers.append(nn.LeakyReLU(0.2))

        self.layer = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layer(input)
        return out

class rgbBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        layer = nn.Conv3d(*args, **kwargs)
        layer.weight.data.normal_()
        layer.bias.data.zero_()
        self.layer = layer
    
    def forward(self, input):
        return self.layer(input)

class linearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = linear

    def forward(self, input):
        return self.linear(input)

def upscale(features):
    return F.interpolate(features, scale_factor=2)

class Generator(nn.Module):
    def __init__(self, latent_size, in_channels, depth):
        super().__init__()
        self.depth = depth
        self.latent_size = latent_size

        self.inputLayer = genDeconv(latent_size, in_channels, 3, 1)

        self.progress4 = genConv(in_channels, in_channels, 3, 1)
        self.progress8 = genConv(in_channels, in_channels, 3, 1)
        self.progress16 = genConv(in_channels, in_channels, 3, 1)
        self.progress32 = genConv(in_channels, in_channels, 3, 1)
        self.progress64 = genConv(in_channels, in_channels//2, 3, 1)
        self.progress128 = genConv(in_channels//2, in_channels//4, 3, 1)
        self.progress256 = genConv(in_channels//4, in_channels//4, 3, 1)

        self.rgb8 = rgbBlock(in_channels, 3, 1)
        self.rgb16 = rgbBlock(in_channels, 3, 1)
        self.rgb32 = rgbBlock(in_channels, 3, 1) 
        self.rgb64 = rgbBlock(in_channels//2, 3, 1)
        self.rgb128 = rgbBlock(in_channels//4, 3, 1)
        self.rgb256 = rgbBlock(in_channels//4, 3, 1)

    
    def progress(self, features, module):
        out = F.interpolate(features, scale_factor=2)
        out = module(out)
        return out

    def output(self, feature1, feature2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skipRGB = upscale(module1(feature1))
            out = (1-alpha)*skipRGB + alpha*module2(feature2)
        else:
            out = module2(feature2)

        return out

    def forward(self, input, step=0, alpha = -1):

        out4 = self.inputLayer(input.view(-1, self.latent_size, 1, 1, 1))
        out4 = self.progress4(out4)
        
        out8 = self.progress(out4, self.progress8)
        self.rgb8(out8)

        out16 = self.progress(out8, self.progress16)
        out32 = self.progress(out16, self.progress32)
        out64 = self.progress(out32, self.progress64)
        out128 = self.progress(out64, self.progress128)
        out256 = self.progress(out128, self.progress256)

        return self.output(out128, out256, self.rgb128, self.rgb256, alpha)


class Discriminator(nn.Module):
    def __init__(self, latent_size=128,):
        super().__init__()

        self.progression = nn.ModuleList([disConv(latent_size//4, latent_size//4, 3, 1),
                                          disConv(latent_size//4, latent_size//2, 3, 1),
                                          disConv(latent_size//2, latent_size, 3, 1),
                                          disConv(latent_size, latent_size, 3, 1),
                                          disConv(latent_size, latent_size, 3, 1),
                                          disConv(latent_size, latent_size, 3, 1),
                                          disConv(latent_size+1, latent_size, 3, 1)])

        self.fromRGB = nn.ModuleList([rgbBlock(1, latent_size//4,1),
                                      rgbBlock(1, latent_size//4,1),
                                      rgbBlock(1, latent_size//2,1),
                                      rgbBlock(1, latent_size, 1),
                                      rgbBlock(1, latent_size, 1), 
                                      rgbBlock(1, latent_size, 1),
                                      rgbBlock(1, latent_size, 1)])
        
        self.nLayers = len(self.progression)
        
        self.linear = linearBlock(latent_size, 1)

    def forward(self, input, depth, alpha=-1):
        for i in range(depth, -1, -1):
            index = self.nLayers - i - 1

            if i==depth:
                out = self.fromRGB[index](input)

            if i==0:
                outStd = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                meanStd = outStd.mean()
                meanStd = meanStd.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, meanStd], 1)

            out = self.progression[index](out)

            if i > 0:
                out = F.interpolate(out, scale_factor=0.5)

                if i == depth and 0 <= alpha < 1:
                    skipRGB = F.interpolate(input, scale_factor=0.5)
                    skipRGB = self.fromRGB[index+1](skipRGB)
                    out = (1 - alpha) * skipRGB + alpha * out
            
            out = out.squeeze(2).squeeze(2)
            out = self.linear(out)

        return out