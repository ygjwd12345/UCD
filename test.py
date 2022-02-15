import utils
import argparser
import os
from utils.logger import Logger

from apex.parallel import DistributedDataParallel
from apex import amp
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import VOCSegmentationIncremental, AdeSegmentationIncremental, CitySegmentationIncremental
from dataset import transform
from metrics import StreamSegMetrics

from segmentation_module import make_model

from train import Trainer
import tasks

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = transform.Compose([
            transform.Resize(size=opts.crop_size),
            transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),

        ])
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),

        ])

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    elif opts.dataset == 'city':
        dataset = CitySegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(root=opts.data_root, train=True, transform=train_transform,
                        labels=list(labels), labels_old=list(labels_old),
                        idxs_path=path_base + f"/train-{opts.step}.npy",
                        masking=not opts.no_mask, overlap=opts.overlap)

    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst)-train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(root=opts.data_root, train=False, transform=val_transform,
                          labels=list(labels), labels_old=list(labels_old),
                          idxs_path=path_base + f"/val-{opts.step}.npy",
                          masking=not opts.no_mask, overlap=True)

    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = dataset(root=opts.data_root, train=opts.val_on_trainset, transform=val_transform,
                       labels=list(labels_cum),
                       idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy")

    return train_dst, val_dst, test_dst, len(labels_cum)


def main(opts):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = opts.MASTER_PORT
    distributed.init_process_group(backend='nccl')

    # distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    train_dst, val_dst, test_dst, n_classes = get_dataset(opts)
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    step_checkpoint = None
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))

    if opts.fix_bn:
        model.fix_bn()

    logger.debug(model)

    # xxx Set up optimizer
    params = []
    if not opts.freeze:
        params.append({"params": filter(lambda p: p.requires_grad, model.body.parameters()),
                       'weight_decay': opts.weight_decay})

    params.append({"params": filter(lambda p: p.requires_grad, model.head.parameters()),
                   'weight_decay': opts.weight_decay})

    params.append({"params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                   'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)



    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    # load best model

    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    # Put the model on GPU
    model = DistributedDataParallel(model.cuda(device))
    path = opts.step_ckpt
    step_checkpoint = torch.load(path, map_location="cpu")
    logger.info(f"*** Model restored from {path}")

    model.load_state_dict(step_checkpoint['model_state'], strict=False)
    ### test old just for test
    # model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step-1))
    # # Put the model on GPU
    # model_old = DistributedDataParallel(model_old.cuda(device))
    # path = './checkpoints/step/19-1-voc_att_0.pth'
    # step_checkpoint = torch.load(path, map_location="cpu")
    # model_old.load_state_dict(step_checkpoint['model_state'], strict=False)

    # logger.info(f"*** Model restored from {path}")
    # trainer = Trainer(model, model_old, device=device, opts=opts)
    trainer = Trainer(model, None, device=device, opts=opts)

    model.eval()
    val_metrics = StreamSegMetrics(n_classes)

    val_loss, val_score, ret_samples = trainer.test(loader=test_loader, metrics=val_metrics, logger=logger)

    logger.print("Done test")
    logger.info(f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
                f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
    logger.info(val_metrics.to_str(val_score))
    ###save output just simple gpu for save operation !!!!
    import cv2
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    colormap_dir = './voc'
    if not os.path.isdir(colormap_dir):
        os.mkdir(colormap_dir)

    ###
    for k, (img, target, lbl, att) in enumerate(ret_samples):

        img = (denorm(img) * 255).transpose(1,2,0).astype(np.uint8)
        ## BGR to RGB
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        cv2.imwrite(os.path.join(colormap_dir, str(k).zfill(4) + 'pre.png'),lbl)
        cv2.imwrite(os.path.join(colormap_dir, str(k).zfill(4) + 'gt.jpg'), target)
        target = label2color(target).astype(np.uint8)
        lbl = label2color(lbl).astype(np.uint8)
        cv2.imwrite(os.path.join(colormap_dir, str(k).zfill(4) + 'pre_clo.png'),lbl)
        cv2.imwrite(os.path.join(colormap_dir, str(k).zfill(4) + 'gt_clo.jpg'), target)
        cv2.imwrite(os.path.join(colormap_dir, str(k).zfill(4) + 'rgb.jpg'), img)
        ### for att vis
        # import matplotlib.colors as colors
        # import matplotlib.cm as cmx
        # import matplotlib.pyplot as plt
        # jet = plt.get_cmap('jet')
        # cNorm = colors.Normalize()
        # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        # att=255*att/np.max(att)
        # att = scalarMap.to_rgba(att)
        # cv2.imwrite(os.path.join(colormap_dir, str(k).zfill(4) + 'att.png'), att.astype(np.uint16))



        # concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
        # print(concat_img.shape)
        # logger.add_image(f'Sample_{k}', concat_img)



    # logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    # logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    # logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    # logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    # logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.info(f"Validation, T_MeanIoU={val_score['Mean IoU']}, T_MeanAcc={val_score['Mean Acc']} (without scaling)")
    # logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)


    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)
    print(opts.dataset)
    os.makedirs("checkpoints/step", exist_ok=True)
    main(opts)
