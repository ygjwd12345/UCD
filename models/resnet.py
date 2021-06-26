import sys
from collections import OrderedDict
from functools import partial

import torch.nn as nn

from modules import GlobalAvgPool2d, ResidualBlock

from .util import try_index


class ResNet(nn.Module):
    """Standard residual network

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the four modules of the network
    bottleneck : bool
        If `True` use "bottleneck" residual blocks with 3 convolutions, otherwise use standard blocks
    norm_act : callable
        Function to create normalization / activation Module
    classes : int
        If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
        of the network
    dilation : int or list of int
         List of dilation factors for the four modules of the network, or `1` to ignore dilation
    keep_outputs : bool
        If `True` output a list with the outputs of all modules
    """

    def __init__(
        self,
        structure,
        bottleneck,
        norm_act=nn.BatchNorm2d,
        classes=0,
        output_stride=16,
        keep_outputs=False
    ):
        super(ResNet, self).__init__()
        self.structure = structure
        self.bottleneck = bottleneck
        self.keep_outputs = keep_outputs

        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if output_stride != 8 and output_stride != 16:
            raise ValueError("Output stride must be 8 or 16")

        if output_stride == 16:
            dilation = [1, 1, 1, 2]  # dilated conv for last 3 blocks (9 layers)
        elif output_stride == 8:
            dilation = [1, 1, 2, 4]  # 23+3 blocks (78 layers)
        else:
            raise NotImplementedError

        self.dilation = dilation

        # Initial layers
        layers = [
            ("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)), ("bn1", norm_act(64))
        ]
        if try_index(dilation, 0) == 1:
            layers.append(("pool1", nn.MaxPool2d(3, stride=2, padding=1)))
        self.mod1 = nn.Sequential(OrderedDict(layers))

        # Groups of residual blocks
        in_channels = 64
        if self.bottleneck:
            channels = (64, 64, 256)
        else:
            channels = (64, 64)
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                stride, dil = self._stride_dilation(dilation, mod_id, block_id)
                blocks.append(
                    (
                        "block%d" % (block_id + 1),
                        ResidualBlock(
                            in_channels,
                            channels,
                            norm_act=norm_act,
                            stride=stride,
                            dilation=dil,
                            last=block_id == num - 1
                        )
                    )
                )

                # Update channels and p_keep
                in_channels = channels[-1]

            # Create module
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

            # Double the number of channels for the next module
            channels = [c * 2 for c in channels]

        self.out_channels = in_channels

        # Pooling and predictor
        if classes != 0:
            self.classifier = nn.Sequential(
                OrderedDict(
                    [("avg_pool", GlobalAvgPool2d()), ("fc", nn.Linear(in_channels, classes))]
                )
            )

    @staticmethod
    def _stride_dilation(dilation, mod_id, block_id):
        d = try_index(dilation, mod_id)
        s = 2 if d == 1 and block_id == 0 and mod_id > 0 else 1
        return s, d

    def forward(self, x):
        outs = []
        attentions = []

        x = self.mod1(x)
        #attentions.append(x)
        outs.append(x)

        x, att = self.mod2(x)
        attentions.append(att)
        outs.append(x)

        x, att = self.mod3(x)
        attentions.append(att)
        outs.append(x)

        x, att = self.mod4(x)
        attentions.append(att)
        outs.append(x)

        x, att = self.mod5(x)
        attentions.append(att)
        outs.append(x)

        if hasattr(self, "classifier"):
            outs.append(self.classifier(outs[-1]))

        if self.keep_outputs:
            return outs, attentions
        else:
            return outs[-1], attentions


_NETS = {
    "18": {
        "structure": [2, 2, 2, 2],
        "bottleneck": False
    },
    "34": {
        "structure": [3, 4, 6, 3],
        "bottleneck": False
    },
    "50": {
        "structure": [3, 4, 6, 3],
        "bottleneck": True
    },
    "101": {
        "structure": [3, 4, 23, 3],
        "bottleneck": True
    },
    "152": {
        "structure": [3, 8, 36, 3],
        "bottleneck": True
    },
}

__all__ = []
for name, params in _NETS.items():
    net_name = "net_resnet" + name
    setattr(sys.modules[__name__], net_name, partial(ResNet, **params))
    __all__.append(net_name)
