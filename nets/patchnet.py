# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import Up, OutConv



class Down(nn.Module):
    def __init__(self, inchan,outd, dilated=True, bn=True, bn_affine=False, k=3, 
                 stride=1, dilation=1, relu=True, k_pool = 1, pool_type='max'):
        super().__init__()
        self.inchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.k = k
        self.stride = stride
        self.relu = relu
        self.k_pool = k_pool
        self.pool_type = pool_type
        d = self.dilation
        if self.dilated: 
            conv_params = dict(padding=((self.k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= self.stride
        else:
            conv_params = dict(padding=((self.k-1)*d)//2, dilation=d, stride=self.stride)
        self.ops1 = nn.Conv2d(self.inchan, outd, kernel_size=k, **conv_params)
        self.ops2 = nn.BatchNorm2d(outd, affine=self.bn_affine)
        self.ops3 = nn.ReLU(inplace=True)
        self.ops4 = torch.nn.AvgPool2d(kernel_size=k_pool)
        self.ops5 = torch.nn.MaxPool2d(kernel_size=k_pool)
        
    def forward(self, x):
        # as in the original implementation, dilation is applied at the end of layer, so it will have impact only from next layer
        d = self.dilation
        if self.dilated: 
            conv_params = dict(padding=((self.k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= self.stride
        else:
            conv_params = dict(padding=((self.k-1)*d)//2, dilation=d, stride=stride)
        x = self.ops1(x)
        if self.bn:
            x = self.ops2(x)
        if self.relu:
            x = self.ops3(x)
        if self.k_pool > 1:
            if self.pool_type == 'avg':
                x = self.ops4(x)
            elif self.pool_type == 'max':
                x = self.ops5(x)
        return x



class BaseNet (nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    # def normalize(self, x, ureliability, urepeatability):
    #     return dict(descriptors = F.normalize(x, p=2, dim=1),
    #                 repeatability = self.softmax( urepeatability ),
    #                 reliability = self.softmax( ureliability ))

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)



class PatchNet (BaseNet):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, k_pool = 1, pool_type='max'):
        # as in the original implementation, dilation is applied at the end of layer, so it will have impact only from next layer
        d = self.dilation * dilation
        if self.dilated: 
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        self.ops.append( nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params) )        
        if bn and self.bn:
            self.ops.append( self._make_bn(outd) )
        
        if relu:            
            self.ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd        
        if k_pool > 1:
            if pool_type == 'avg':                
                self.ops.append(torch.nn.AvgPool2d(kernel_size=k_pool))
            elif pool_type == 'max':                
                self.ops.append(torch.nn.MaxPool2d(kernel_size=k_pool))
            else:
                print(f"Error, unknown pooling type {pool_type}...")
        
    
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n,op in enumerate(self.ops):
            x = op(x)
        
        return self.normalize(x)


class L2_Net (PatchNet):
    """ Compute a 128D descriptor for all overlapping 32x32 patches.
        From the L2Net paper (CVPR'17).
    """
    def __init__(self, dim=128, **kw ):
        PatchNet.__init__(self, **kw)
        add_conv = lambda n,**kw: self._add_conv((n*dim)//128,**kw)
        add_conv(32)
        add_conv(32)
        add_conv(64, stride=2)
        add_conv(64)
        add_conv(128, stride=2)
        add_conv(128)
        add_conv(128, k=7, stride=8, bn=False, relu=False)
        self.out_dim = dim


class Quad_L2Net (PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """
    def __init__(self, dim=128, mchan=4, relu22=False, **kw ):
        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, stride=2)
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim



class Quad_L2Net_ConfCFS (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """
    def __init__(self, **kw ):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)


class Fast_Quad_L2Net (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, k_pool = downsample_factor) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        
        # Go back to initial image resolution with upsampling
        self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
        
        self.out_dim = dim
        
        
class Fast_Quad_L2Net_ConfCFS (Fast_Quad_L2Net):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Fast_Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)
class Fast_Quad_L2Net_Semantics_ConfCFS (PatchNet):
    """ Fast r2d2 architecture with semantics
    """
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.down1 = (Down(3, 8*mchan))
        self.down2 = (Down(8*mchan, 8*mchan))
        self.down3 = (Down(8*mchan, 16*mchan, k_pool = downsample_factor)) # added avg pooling to decrease img resolution
        self.down4 = (Down(16*mchan, 16*mchan))
        self.down5 = (Down(16*mchan, 32*mchan, stride=2))
        self.down6 = (Down(32*mchan, 32*mchan,dilation = 2))
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self.down7 = (Down(32*mchan, 32*mchan, k=2, stride=2, relu=relu22, dilation = 2))
        self.down8 = (Down(32*mchan, 32*mchan, k=2, stride=2, relu=relu22, dilation = 4))
        self.down9 = (Down(32*mchan, dim, k=2, stride=2, bn=False, relu=False, dilation = 8))
        
        self.down10 = (Down(32*mchan, 64*mchan))
        self.down11 = (Down(64*mchan, 128*mchan))
        
        # Go back to initial image resolution with upsampling
#         self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
        self.upsample = torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False)
        self.out_dim = dim
        # Semantic decoder
        self.up1 = (Up(128*mchan,64*mchan ,bilinear=False))
        self.up2 = (Up(64*mchan,32*mchan ,bilinear=False))
        self.up3 = (Up(32*mchan, 16*mchan,bilinear=False))
        self.up4 = (Up(16*mchan, 8*mchan,bilinear=False))
        self.outc = (OutConv(32, 1))
        
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
#         assert self.ops, "You need to add convolutions first"
#         for op in self.ops:
#             x = op(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x9 = self.down9(x8)
        x10 = self.upsample(x9)
        
        # compute the confidence maps
        ureliability = self.clf(x10**2)
        urepeatability = self.sal(x10**2)
        
        x11 = self.down10(x9)
        x12 = self.down11(x11)
        x = self.up1(x12, x11)
        x = self.up2(x, x9)
        x = self.up3(x, x4)
        x = self.up4(x, x2)
        logits = self.outc(x)
        return (self.normalize(x10, ureliability, urepeatability, logits))
    def use_checkpointing(self):
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.down6 = torch.utils.checkpoint(self.down6)
        self.down7 = torch.utils.checkpoint(self.down7)
        self.down8 = torch.utils.checkpoint(self.down8)
        self.down9 = torch.utils.checkpoint(self.down9)
        self.down10 = torch.utils.checkpoint(self.down10)
        self.down11 = torch.utils.checkpoint(self.down11)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.upsample = torch.utils.checkpoint(self.upsample)
        self.outc = torch.utils.checkpoint(self.outc)
        self.clf = torch.utils.checkpoint(self.clf)
        self.sal = torch.utils.checkpoint(self.sal)
    def normalize(self, x, ureliability, urepeatability, logits):
        return dict(descriptors = F.normalize(x, p=2, dim=1),
                    repeatability = self.softmax( urepeatability ),
                    reliability = self.softmax( ureliability ),
                    logits = logits)