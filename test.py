import copy
import os
import random

import numpy as np
import torch
from apex import amp
from apex.parallel import DistributedDataParallel
from torch import distributed
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

import argparser
import tasks
import utils
from dataset import (AdeSegmentationIncremental,CitySegmentationIncremental,
                     CityscapesSegmentationIncrementalDomain,
                     VOCSegmentationIncremental, transform)
from metrics import StreamSegMetrics
from segmentation_module import make_model
from train import Trainer
from utils.logger import Logger
withwandb = False
try:
    import wandb
except ImportError:
    withwandb = False
    print('WandB disabled')

def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict()
    }
    torch.save(state, path)


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose(
        [
            transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if opts.crop_val:
        val_transform = transform.Compose(
            [
                transform.Resize(size=opts.crop_size),
                transform.CenterCrop(size=opts.crop_size),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose(
            [
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    elif opts.dataset == 'cityscapes_domain':
        dataset = CityscapesSegmentationIncrementalDomain
    elif opts.dataset == 'city':
        dataset = CitySegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)
    if opts.dataset == 'city':
        train_dst = dataset(root=opts.data_root, train=True, transform=train_transform,
                            labels=list(labels), labels_old=list(labels_old),
                            idxs_path=path_base + f"/train-{opts.step}.npy",
                            masking=not opts.no_mask, overlap=opts.overlap)
    else:
        train_dst = dataset(
            root=opts.data_root,
            train=True,
            transform=train_transform,
            labels=list(labels),
            labels_old=list(labels_old),
            idxs_path=path_base + f"/train-{opts.step}.npy",
            masking=not opts.no_mask,
            overlap=opts.overlap,
            disable_background=opts.disable_background,
            data_masking=opts.data_masking,
            test_on_val=opts.test_on_val,
            step=opts.step
        )

    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst) - train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        if opts.dataset == 'city':
            val_dst = dataset(root=opts.data_root, train=False, transform=val_transform,
                              labels=list(labels), labels_old=list(labels_old),
                              idxs_path=path_base + f"/val-{opts.step}.npy",
                              masking=not opts.no_mask, overlap=True)
        else:
            val_dst = dataset(
                root=opts.data_root,
                train=False,
                transform=val_transform,
                labels=list(labels),
                labels_old=list(labels_old),
                idxs_path=path_base + f"/val-{opts.step}.npy",
                masking=not opts.no_mask,
                overlap=True,
                disable_background=opts.disable_background,
                data_masking=opts.data_masking,
                step=opts.step
            )

    image_set = 'train' if opts.val_on_trainset else 'val'
    if opts.dataset == 'city':
        test_dst = dataset(root=opts.data_root, train=opts.val_on_trainset, transform=val_transform,
                           labels=list(labels_cum),
                           idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy")
    else:
        test_dst = dataset(
            root=opts.data_root,
            train=opts.val_on_trainset,
            transform=val_transform,
            labels=list(labels_cum),
            idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy",
            disable_background=opts.disable_background,
            test_on_val=opts.test_on_val,
            step=opts.step,
            ignore_test_bg=opts.ignore_test_bg
        )

    return train_dst, val_dst, test_dst, len(labels_cum)


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    if len(opts.lr) == 1 and len(opts.step) > 1:
        opts.lr = opts.lr * len(opts.step)

    os.makedirs("results", exist_ok=True)

    print(f"Learning for {len(opts.step)} with lrs={opts.lr}.")
    all_val_scores = []
    for i, (step, lr) in enumerate(zip(copy.deepcopy(opts.step), copy.deepcopy(opts.lr))):
        if i > 0:
            opts.step_ckpt = None

        opts.step = step
        opts.lr = lr

        val_score = run_step(opts, world_size, rank, device)
        if rank == 0:
            all_val_scores.append(val_score)

        torch.cuda.empty_cache()

        if rank == 0:
            with open(f"results/{opts.date}_{opts.dataset}_{opts.task}_{opts.name}.csv", "a+") as f:
                classes_iou = ','.join(
                    [str(val_score['Class IoU'].get(c, 'x')) for c in range(opts.num_classes)]
                )
                f.write(f"{step},{classes_iou},{val_score['Mean IoU']}\n")


def run_step(opts, world_size, rank, device):
    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(
            logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step
        )
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

    train_loader = data.DataLoader(
        train_dst,
        batch_size=opts.batch_size,
        sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
        num_workers=opts.num_workers,
        drop_last=True
    )
    val_loader = data.DataLoader(
        val_dst,
        batch_size=opts.batch_size if opts.crop_val else 1,
        sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
        num_workers=opts.num_workers
    )
    logger.info(
        f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
        f" Test set: {len(test_dst)}, n_classes {n_classes}"
    )
    logger.info(f"Total batch size is {opts.batch_size * world_size}")

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    opts.inital_nb_classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)[0]

    step_checkpoint = None
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old = make_model(
            opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1)
        )

    if opts.fix_bn:
        model.fix_bn()

    logger.debug(model)

    # xxx Set up optimizer
    params = []
    if not opts.freeze:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.body.parameters()),
                'weight_decay': opts.weight_decay
            }
        )

    params.append(
        {
            "params": filter(lambda p: p.requires_grad, model.head.parameters()),
            'weight_decay': opts.weight_decay
        }
    )

    if opts.lr_old is not None and opts.step > 0:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls[:-1].parameters()),
                'weight_decay': opts.weight_decay,
                "lr": opts.lr_old * opts.lr
            }
        )
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls[-1:].parameters()),
                'weight_decay': opts.weight_decay
            }
        )
    else:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                'weight_decay': opts.weight_decay
            }
        )
    if model.scalar is not None:
        params.append({"params": model.scalar, 'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(
            optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power
        )
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor
        )
    else:
        raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)


    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(
            len(val_loader), opts.sample_num, replace=False
        )  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    if opts.dataset == "cityscapes_domain":
        val_metrics = StreamSegMetrics(opts.num_classes)
    else:
        val_metrics = StreamSegMetrics(n_classes)
    results = {}



    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(
        test_dst,
        batch_size=opts.batch_size if opts.crop_val else 1,
        sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
        num_workers=opts.num_workers
    )

    # load best model

    model = make_model(
        opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
    )
    # Put the model on GPU
    model = DistributedDataParallel(model.cuda(device))
    ckpt = opts.ckpt
    checkpoint = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=False)
    logger.info(f"*** Model restored from {ckpt}")
    del checkpoint
    trainer = Trainer(model, None, device=device, opts=opts, step=opts.step)

    model.eval()

    val_loss, val_score, ret_samples = trainer.validate(
        loader=test_loader, metrics=val_metrics, logger=logger, end_task=True
    )
    logger.print("Done test")
    logger.info(
        f"*** End of Test, Total Loss={val_loss[0] + val_loss[1]},"
        f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}"
    )
    logger.info(val_metrics.to_str(val_score))
    ###save output just simple gpu for save operation !!!!
    logger.print("start save result")

    import cv2
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    colormap_dir = './voc'
    if not os.path.isdir(colormap_dir):
        os.mkdir(colormap_dir)

    ###
    for k, (img, target, lbl) in enumerate(ret_samples):

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
    logger.print(" save result done")


    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)

    logger.close()

    del model
    if model_old is not None:
        del model_old

    return val_score


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs(f"{opts.checkpoint}", exist_ok=True)
    print(opts.dataset)
    if withwandb:
        wandb.init(project="plop_cil_pami", group=opts.dataset)
    os.makedirs("checkpoints/step", exist_ok=True)
    main(opts)
