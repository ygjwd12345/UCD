import os
import random
import torch.utils.data as data
from torch import distributed
import torchvision as tv
import numpy as np
from .utils import Subset, filter_images, group_images
from PIL import Image
import torch
from torchvision import transforms

classes={
    0:'void',
    1:'road',
    2:'sidewalk',
    3:'building',
    4:'wall',
    5:'fence',
    6:'pole',
    7:'traffic light',
    8:'traffic sign',
    9:'vegetation',
    10:'terrain',
    11:'sky',
    12:'person',
    13:'rider',
    14:'car',
    15:'truck',
    16:'bus',
    17:'train',
    18:'motorcycle',
    19:'bicycle'
}

class CitySegmentation(data.Dataset):
    def __init__(self,
                 root,
                 train):
        root = os.path.expanduser(root)
        base_dir = 'Cityscapes'
        city_root = os.path.join(root, base_dir)
        if train == True:
            split = 'train'
        else:
            split = 'val'
        self.images, self.mask_paths = get_city_pairs(city_root, split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + city_root + "\n")
        self._indices = np.array(range(-1, 20))
        self._classes = np.array([0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 31, 32, 33])
        self._key = np.array([0, 0, 0, 0, 0, 0,
                              0, 0,  1,  2, 0, 0,
                              3,   4,  5, 0, 0, 0,
                              6,  0,  7,  8,  9,  10,
                              11, 12, 13, 14, 15, 16,
                              0, 0, 17, 18, 19])
        self._mapping = np.array(range(-1, len(self._key)-1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert (values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        if os.path.exists(mask_file):
            masks = torch.load(mask_file)
            return masks
        masks = []
        print("Preprocessing mask, this will take a while." + \
              "But don't worry, it only run once for each split.")
        for fname in self.mask_paths:
            mask = Image.fromarray(self._class_to_index(
                np.array(Image.open(fname))).astype('int8'))
            masks.append(mask)
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        # mask = self.masks[index]
        mask = Image.open(self.mask_paths[index])
        mask = self._mask_transform(mask)
        mask = transforms.ToPILImage()(mask)
        return img, mask

    def _mask_transform(self, mask):
        # target = np.array(mask).astype('int32') - 1
        target = self._class_to_index(np.array(mask)).astype('int32')
        return target

    def __len__(self):
        return len(self.images)

class CitySegmentationIncremental(data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 labels=None,
                 labels_old=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True):

        self.labels = []
        self.labels_old = []

        full_data = CitySegmentation(root,train)
        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(l in labels_old for l in labels), "labels and labels_old must be disjoint sets"

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old

            self.order = [0] + labels_old + labels
            # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_data, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if train:
                masking_value = 0
            else:
                masking_value = 255

            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = masking_value

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value))

            if masking:
                tmp_labels = self.labels + [255]
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value))
            else:
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_data, idxs, transform, target_transform)
        else:
            self.dataset = full_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)
def get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, directories, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit','gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split == 'train' or split == 'val' or split == 'test':
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/'+ split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths