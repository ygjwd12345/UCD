import torch.nn as nn
import torch.nn.functional as F
import torch


def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index=ignore_index
        self.size_average=size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class IcarlLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        if self.bkg:
            targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, :output_old.shape[1], :, :] = output_old

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        ### output encoding
        self.enc_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            )
    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        ###
        b, c, _, _ = targets.size()

        enc_target=torch.softmax(self.enc_out(targets),dim=1)
        gamma=torch.sum(enc_target[:,1:],dim=1)/(enc_target[:,0])

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        ### add 1030
        # outputs_no_bgk = torch.log_softmax(inputs, dim=1)[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W

        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c

        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]
        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs

class MaskCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets,outputs_old=None):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        ### return to normal
        # outputs[:, 0] = inputs[:, 0] - den
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)

        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets    # B, H, W
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction='none')
        mask=torch.zeros_like(targets)
        if outputs_old is not None:
            pseudo_label=torch.argmax(outputs_old,dim=1)
            mask[pseudo_label == 0 ]=1
            mask[labels > old_cl]=1
            loss = loss * mask.detach().float()
        if self.reduction == 'mean':
            loss = -torch.mean(loss)
        elif self.reduction == 'sum':
            loss = -torch.sum(loss)
        return loss

class MaskKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        new_cl = inputs.shape[1] - targets.shape[1]

        ###
        b, c, _, _ = targets.size()

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c

        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            Mask = torch.zeros_like(mask)
            Mask[mask==0]=1
            loss = loss * Mask.detach().float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs

def pre_contrastive_pixel(f_n, l_n, l_po=None, f_o=None):
    out_size = f_n.shape[-2:]
    ### for pascal bicubic ade bilinear
    label_n = F.interpolate(torch.tensor(l_n.clone().detach(), dtype=torch.float32).unsqueeze(1), size=out_size,
                            mode="bilinear", align_corners=False).type(torch.int8)
    # label_n = nn.AvgPool2d(16, stride=16)(torch.tensor(l_n.clone().detach(),dtype=torch.float32).unsqueeze(1)).type(torch.int8)
    ## worse
    ### for vis
    # label_n=F.interpolate(torch.tensor(l_n,dtype=torch.float32).unsqueeze(1), size=out_size, mode="bilinear", align_corners=False).type(torch.int8)
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



class PixelConLossV2(nn.Module):
    """Supervised Contrastive Learning for segmentation"""

    def __init__(self, sample_method='none', temperature=0.07):
        super(PixelConLossV2, self).__init__()
        self.temperature = temperature
        self.sample_method = sample_method
        print(temperature)

    def forward(self, anchor_features, contrast_feature, anchor_labels, contrast_labels, P=None):
        """
        Args:
            achor_features: hidden vector of shape [bsz, 1, 256].
            contrast_features: hidden vector of shape [bsz_prime, 1, 256].
            anchor_labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if anchor_features.is_cuda
                  else torch.device('cpu'))

        # if len(anchor_features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(anchor_features.shape) > 3:
        #     features = anchor_features.view(anchor_features.shape[0], anchor_features.shape[1], -1)

        anchor_labels = anchor_labels.view(-1, 1)
        contrast_labels = contrast_labels.view(-1, 1)

        batch_size = anchor_features.shape[0]
        R = torch.eq(anchor_labels, contrast_labels.T).float().requires_grad_(False).to(device)
        mask_p = R.clone().requires_grad_(False)
        mask_p[:, :batch_size] -= torch.eye(batch_size).to(device)
        mask_p = mask_p.detach()
        mask_n = 1 - R
        mask_n = mask_n.detach()

        # compute logits
        # realise inner product and temperature

        anchor_dot_contrast = torch.div(
            torch.mm(anchor_features, contrast_feature.T),
            self.temperature)
        ### compute negtive : anchor*R-
        neg_contrast = (torch.exp(anchor_dot_contrast) * mask_n).sum(dim=1,keepdim=True)
        # print(anchor_dot_contrast.shape)
        # print(mask_p.shape)
        # print(P.shape)
        # print(neg_contrast.shape)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()
        if P is None:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * mask_p - torch.log(
                torch.exp(anchor_dot_contrast) + neg_contrast) * mask_p
        else:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * mask_p * P - torch.log(
                torch.exp(anchor_dot_contrast) + neg_contrast) * mask_p * P

        num = mask_p.sum(dim=1)
        loss = -torch.div(pos_contrast.sum(dim=1)[num != 0], num[num != 0])
        return loss.mean()