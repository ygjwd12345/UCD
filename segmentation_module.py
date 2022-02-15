import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional
import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial, reduce

import models
from modules import DeeplabV3
from torch.nn import functional as F
from utils.non_local_embedded_gaussian import NONLocalBlock2D
def make_model(opts, classes=None):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex
    head_channels = 256

    if not opts.no_pretrained:
        body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        # print(pre_dict['state_dict'])
        del pre_dict['state_dict']['module.classifier.fc.weight']
        del pre_dict['state_dict']['module.classifier.fc.bias']

        # print(pre_dict['state_dict'].keys())
        kep = list(pre_dict['state_dict'].keys())

        for key in kep:
            # print(key)
            key_n=key[7:]
            # print(key_n)
            pre_dict['state_dict'][key_n]=pre_dict['state_dict'].pop(key)
        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory


        head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                         out_stride=opts.output_stride, pooling_size=opts.pooling)

    if classes is not None:
        model = IncrementalSegmentationModule(body, head, head_channels, classes=classes, fusion_mode=opts.fusion_mode)
    else:
        model = SegmentationModule(body, head, head_channels, opts.num_classes, opts.fusion_mode)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = nn.ModuleList(
            [nn.Conv2d(head_channels, c, 1) for c in classes]
        )
        ### freeze cls0 for pascal
        # if len(classes) > 1:
        self.cls[0].weight.requires_grad = False
        self.cls[0].bias.requires_grad = False

        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.softmax = nn.Softmax(dim=1)


    def att_map(self,x):
        ### sptial attention
        a = torch.sum(x ** 2, dim=1)
        ### channel attention
        for i in range(a.shape[0]):
            a[i] = a[i] / torch.norm(a[i])
        a = torch.unsqueeze(a, 1)
        x = a.detach() * x
        return x
    def _network(self, x,x_b_old=None,x_pl_old=None, ret_intermediate=False):
        # x_b.shape=[bs,2048,32,32] x_pl.shape=[bs,256,32,32] x_o.shape=[bs,ch_out+1,32,32]
        ### for origin and reproduce
        x_b = self.body(x)

        x_pl = self.head(x_b)

        out = []
        for mod in self.cls:
            out.append(mod(x_pl))
        x_o = torch.cat(out, dim=1)
        x_b=self.att_map(x_b)
        x_pl=self.att_map(x_pl)
        return x_o, x_b,  x_pl


    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, x_b_old=None,x_pl_old=None, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, x_b_old, x_pl_old,ret_intermediate)

        sem_logits = out[0]
        ### for pascal bicubic ade bilinear

        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)


        return sem_logits, {"body": out[1], "pre_logits": out[2],"sem":out[0]}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
