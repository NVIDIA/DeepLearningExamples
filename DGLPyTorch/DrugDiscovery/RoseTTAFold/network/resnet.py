import torch
import torch.nn as nn

# original resblock
class ResBlock2D(nn.Module):
    def __init__(self, n_c, kernel=3, dilation=1, p_drop=0.15):
        super(ResBlock2D, self).__init__()
        padding = self._get_same_padding(kernel, dilation)

        layer_s = list()
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, padding=padding, dilation=dilation, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # dropout
        layer_s.append(nn.Dropout(p_drop))
        # convolution
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, dilation=dilation, padding=padding, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        self.layer = nn.Sequential(*layer_s)
        self.final_activation = nn.ELU(inplace=True)

    def _get_same_padding(self, kernel, dilation):
        return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2

    def forward(self, x):
        out = self.layer(x)
        return self.final_activation(x + out)

# pre-activation bottleneck resblock
class ResBlock2D_bottleneck(nn.Module):
    def __init__(self, n_c, kernel=3, dilation=1, p_drop=0.15):
        super(ResBlock2D_bottleneck, self).__init__()
        padding = self._get_same_padding(kernel, dilation)

        n_b = n_c // 2 # bottleneck channel
        
        layer_s = list()
        # pre-activation
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # project down to n_b
        layer_s.append(nn.Conv2d(n_c, n_b, 1, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_b, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # convolution
        layer_s.append(nn.Conv2d(n_b, n_b, kernel, dilation=dilation, padding=padding, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_b, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # dropout
        layer_s.append(nn.Dropout(p_drop))
        # project up
        layer_s.append(nn.Conv2d(n_b, n_c, 1, bias=False))

        self.layer = nn.Sequential(*layer_s)

    def _get_same_padding(self, kernel, dilation):
        return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2

    def forward(self, x):
        out = self.layer(x)
        return x + out

class ResidualNetwork(nn.Module):
    def __init__(self, n_block, n_feat_in, n_feat_block, n_feat_out, 
                 dilation=[1,2,4,8], block_type='orig', p_drop=0.15):
        super(ResidualNetwork, self).__init__()


        layer_s = list()
        # project to n_feat_block
        if n_feat_in != n_feat_block:
            layer_s.append(nn.Conv2d(n_feat_in, n_feat_block, 1, bias=False))
            if block_type =='orig': # should acitivate input
                layer_s.append(nn.InstanceNorm2d(n_feat_block, affine=True, eps=1e-6))
                layer_s.append(nn.ELU(inplace=True))

        # add resblocks
        for i_block in range(n_block):
            d = dilation[i_block%len(dilation)]
            if block_type == 'orig':
                res_block = ResBlock2D(n_feat_block, kernel=3, dilation=d, p_drop=p_drop)
            else:
                res_block = ResBlock2D_bottleneck(n_feat_block, kernel=3, dilation=d, p_drop=p_drop)
            layer_s.append(res_block)

        if n_feat_out != n_feat_block:
            # project to n_feat_out
            layer_s.append(nn.Conv2d(n_feat_block, n_feat_out, 1))
        
        self.layer = nn.Sequential(*layer_s)
    
    def forward(self, x):
        output = self.layer(x)
        return output

