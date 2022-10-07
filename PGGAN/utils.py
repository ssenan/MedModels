import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x / torch.sqrt(torch.sum(a**2, dim=1, keepdim=True) + 1e-8)
        return y

#--------------------------------------------------------------

class MiniBatchStdDev(nn.Module):
    pass


#--------------------------------------------------------------
class WGANP:
    def __init__(self, drift: float=0.001):
        self.drift = drift

    def gradient_penalty(
        dis,
        real_image,
        fake_image,
        depth,
        alpha,
        reg_lambda,
        labels
    ):
        batch_size = real_image.shape(0)

        epsilon = torch.rand((batch_size, 1, 1, 1)).to(real_image.device)

        # Create and merge real/fake images
        merged = epsilon * real_image + ((1 - epsilon) * fake_image)
        merged.requires_grad_(True)

        # Forward pass
        op = dis(merged, depth, alpha, labels)

        # Backward pass
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        return penalty
    
    def dis_loss(
        self,
        discriminator,
        real_image,
        fake_image,
        depth,
        alpha
    ):
        real_scores = discriminator(real_image, depth, alpha)
        fake_scores = discriminator(fake_image, depth, alpha)

        loss =(
            torch.mean(fake_scores)
            - torch.mean(real_scores)
            + (self.drift * torch.mean(real_scores) ** 2)
        )

        # calculate WGAN-GP (gradient penalty)
        gp = self._gradient_penalty(
            discriminator, real_image, fake_image, depth, alpha
        )
        loss += gp
        
        return loss

    def gen_loss(
        self,
        discriminator,
        _,
        fake_image,
        depth,
        alpha,
        labels
    ):
        fake_scores = discriminator(fake_image, depth, alpha, labels)

        return -torch.mean(fake_scores)
#--------------------------------------------------------------
class UnetConv3D(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)):
        super(UnetConv3D, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, stride, padding),
                                    nn.InstanceNorm3d(out_size),
                                    nn.LeakyReLU(inplace=True),)

        self.conv2  = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding),
                                    nn.InstanceNorm3d(out_size),
                                    nn.LeakyReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

#--------------------------------------------------------------

class UnetUpscale3D(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUpscale3D, self).__init__()
        if is_deconv:
            self.conv = UnetConv3D(in_size, out_size)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            self.conv = UnetConv3D(in_size + out_size, out_size)
            self.up = nn.Upsample(scale_factor=(2,2,2), mode = "trilinear")

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

#--------------------------------------------------------------

class UnetUpscale3D_CT(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUpscale3D_CT, self).__init__()
        self.conv = UnetConv3D(in_size + out_size, out_size, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode = "trilinear")
    
    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
