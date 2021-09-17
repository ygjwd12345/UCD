from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def color_map(dataset):
    if dataset=='voc':
        return voc_cmap()
    elif dataset=='city':
        return cityscapes_cmap()
    elif dataset=='ade':
        return ade_cmap()


def cityscapes_cmap():
    return np.array([(  0,  0,  0),(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30),
                         (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), 
                         (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32) ],
                         dtype=np.uint8)


def ade_cmap():
    cmap = np.zeros((256, 3), dtype=np.uint8)
    colors = [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255]
    ]

    for i in range(len(colors)):
        cmap[i] = colors[i]

    return cmap.astype(np.uint8)


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]


def convert_bn2gn(module):
    mod = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        num_features = module.num_features
        num_groups = num_features//16
        mod = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    for name, child in module.named_children():
        mod.add_module(name, convert_bn2gn(child))
    del module
    return mod

def pre_contractive_pixel(f_n, l_n, l_po=None, f_o=None):
    out_size = f_n.shape[-2:]
    ### for pascal bicubic ade bilinear
    # label_n = F.interpolate(torch.tensor(l_n.clone().detach(), dtype=torch.float32).unsqueeze(1), size=out_size,
    #                         mode="bilinear", align_corners=False).type(torch.int8)
    # label_n = nn.AvgPool2d(16, stride=16)(torch.tensor(l_n.clone().detach(),dtype=torch.float32).unsqueeze(1)).type(torch.int8)
    ## worse
    ### for vis
    label_n=F.interpolate(torch.tensor(l_n,dtype=torch.float32).unsqueeze(1), size=out_size, mode="bilinear", align_corners=False).type(torch.int8)
    # label_n = nn.AvgPool2d(16, stride=16)(torch.tensor(l_n,dtype=torch.float32).unsqueeze(1)).type(torch.int8)
    ## worse
    label_n[label_n < 0] = 0
    label_n[label_n > 20] = 0
    # print(label_n.shape)
    # print(torch.unique(label_n))
    B, N, h, w = f_n.size()
    # print(f_n.size())
    f_n = f_n.permute(0, 2, 3, 1)
    f_n = f_n.reshape(B, h * w, N)

    if f_o is not None and l_po is None:
        ### pixel to pixel double
        f_o = f_o.detach().permute(0, 2, 3, 1)
        f_o = f_o.reshape(B, h * w, N)
        Output = torch.cat((f_n.reshape(B * h * w, N), f_o.reshape(B * h * w, N)), dim=0)
        Output = F.normalize(Output, dim=1)
        Lable = torch.cat((label_n.reshape(B * h * w), label_n.reshape(B * h * w)))
    if f_o is None and l_po is None:
        ### pixel to pixel single
        Output = f_n.reshape(B * h * w, N)
        Output = F.normalize(Output, dim=1)
        Lable = label_n.reshape(B * h * w)

    ### with background
    # if f_o is not None and l_po is not None:
    #     ### double and mix label
    #     f_o = f_o.detach().permute(0, 2, 3, 1)
    #     f_o = f_o.reshape(B, h * w, N)
    #     Output = torch.cat((f_n.reshape(B * h * w, N),f_o.reshape(B * h * w, N)),dim=0)
    #     Output = F.normalize(Output, dim=1)
    #     label_n=label_n.squeeze()
    #     ### add Joint probality mask
    #     B, N, h, w = l_po.shape
    #     l_po_ = l_po.permute(0, 2, 3, 1)
    #     l_po_ = l_po_.reshape(B, h * w, N)
    #     l_po_ = l_po_.reshape(B * h * w, N)
    #     ### new from gt set to 1
    #     l_po_=torch.cat((l_po_, l_po_))
    #     JM_p=torch.mm(l_po_,l_po_.T)
    #     Lable_n=torch.cat((label_n.reshape(B * h * w),label_n.reshape(B * h * w)))
    #     Lable_n[Lable_n>0]=1
    #     Lable_n=Lable_n.unsqueeze(dim=1)
    #     M_gt=torch.mm(Lable_n,Lable_n.T)
    #     JM_p[M_gt==1]=1
    #     _,label_po = l_po.max(dim=1)
    #     label_n[label_n==0]=label_po.cpu().to(label_n.dtype)[label_n==0]
    #     Lable=torch.cat((label_n.reshape(B * h * w),label_n.reshape(B * h * w)))
    #
    #     return Output.unsqueeze(1),Lable,JM_p.detach()
    ### without background
    if f_o is not None and l_po is not None:

        version = 'v2'
        if version == 'v1':
            ## mix label
            label_mix = label_n.squeeze()
            _, label_po = l_po.max(dim=1)
            label_mix[label_mix == 0] = label_po.cpu().to(label_mix.dtype)[label_mix == 0]
            label_ = label_mix.reshape(B * h * w)
            Lable = torch.cat((label_[label_ > 0], label_[label_ > 0]))
            ##double
            f_o = f_o.detach().permute(0, 2, 3, 1)
            f_o = f_o.reshape(B, h * w, N)
            Output = torch.cat((f_n.reshape(B * h * w, N)[label_ > 0], f_o.reshape(B * h * w, N)[label_ > 0]), dim=0)
            Output = F.normalize(Output, dim=1)

            ## make joint probality mask
            B, N, h, w = l_po.shape
            l_po_ = l_po.permute(0, 2, 3, 1)
            l_po_ = torch.softmax(l_po_, dim=-1)
            l_po_ = l_po_.reshape(B, h * w, N)
            l_po_ = l_po_.reshape(B * h * w, N)
            l_po_ = torch.cat((l_po_[label_ > 0], l_po_[label_ > 0]))
            JM_p = torch.mm(l_po_, l_po_.T)
            label_N = label_n
            label_N[label_n > 0] = 1
            Label_N = torch.cat((label_N.reshape(B * h * w)[label_ > 0], label_N.reshape(B * h * w)[label_ > 0]))
            Lable_N = Label_N.unsqueeze(dim=1)
            M_gt = torch.mm(Lable_N, Lable_N.T)
            JM_p[M_gt == 1] = 1

            return Output, Lable, JM_p

        elif version == 'v2':

            ## mix label
            mask_new_classes = label_n.view(B * h * w) > 0
            min_new_classes = label_n.view(B * h * w)[mask_new_classes].min()
            label_mix = label_n.squeeze()
            _, label_po = l_po.max(dim=1)
            label_mix[label_mix == 0] = label_po.cpu().to(label_mix.dtype)[label_mix == 0]
            label_ = label_mix.reshape(B * h * w)
            Lable_anchor = label_[label_ > 0].clone()
            Lable_contrast = torch.cat((Lable_anchor,label_[(label_ > 0) & ~mask_new_classes]))
            ##double
            f_o = f_o.detach().permute(0, 2, 3, 1)
            f_o = f_o.reshape(B, h * w, N)
            Output_anchor = F.normalize(f_n.reshape(B * h * w, N)[label_ > 0], dim=1)
            Output_contrast = torch.cat((Output_anchor,F.normalize(f_o.reshape(B * h * w, N)[(label_ > 0) & ~mask_new_classes], dim=1)), dim=0).detach()

            ## make joint probality mask
            B, N, h, w = l_po.shape
            l_po_ = l_po.permute(0, 2, 3, 1)
            l_po_ = torch.softmax(l_po_, dim=-1)
            l_po_ = l_po_.reshape(B, h * w, N)
            l_po_ = l_po_.reshape(B * h * w, N)
            l_po_anchor = l_po_[label_ > 0]
            l_po_contrast = torch.cat((
                l_po_[label_ > 0],
                l_po_[(label_ > 0) & ~mask_new_classes]))
            JM_p = torch.mm(l_po_anchor, l_po_contrast.T)
            # mask old classes on anchor labels
            mask_anchor_jp = Lable_anchor.clone()
            mask_new_classes_anchor = mask_anchor_jp >= min_new_classes
            mask_anchor_jp[mask_new_classes_anchor] = 1
            mask_anchor_jp[~mask_new_classes_anchor] = 0
            # mask old classes on anchor labels
            mask_contrast_jp = Lable_contrast.clone()
            mask_new_classes_contrast = mask_contrast_jp >= min_new_classes
            mask_contrast_jp[mask_new_classes_contrast] = 1
            mask_contrast_jp[~mask_new_classes_contrast] = 0
            # fix gt with gt cases
            mask_anchor_jp=mask_anchor_jp.unsqueeze(dim=1)
            mask_contrast_jp=mask_contrast_jp.unsqueeze(dim=1)
            M_gt = torch.mm(mask_anchor_jp, mask_contrast_jp.T)
            JM_p[M_gt == 1] = 1

            return Output_anchor, Output_contrast, Lable_anchor, Lable_contrast, JM_p.detach()

    # import ipdb; ipdb.set_trace()

    return Output.unsqueeze(1), Lable


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0])/ Q.shape[0]
        c = torch.ones(Q.shape[1])/ Q.shape[1]
        r=r.cuda()
        c=c.cuda()
        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r/ u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits):
        # get assignments
        q = logits / self.epsilon
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q)
if __name__ == '__main__':
    a=torch.randn([3,3,3,3])
    a= a.permute(0, 2, 3, 1)
    a=a.reshape(3*3*3,3)
    print(a.shape)
    cluster=SinkhornKnopp()
    b=cluster(a)
    print(b.shape)
    print(b)
