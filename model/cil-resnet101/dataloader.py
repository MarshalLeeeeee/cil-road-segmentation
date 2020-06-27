import os
import cv2
import torch
import numpy as np
from torch.utils import data
from config import config
from util import random_scale, random_mirror, normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape

data_setting = {'img_root': config.img_root_folder,
                'gt_root': config.gt_root_folder,
                'train_source': config.train_source,
                'eval_source': config.eval_source,
                'test_source': config.test_source}

def img_to_black(img, threshold=50):
    img = img.astype(np.int64)
    idx = img[:, :] > threshold
    idx_0 = img[:, :] <= threshold
    img[idx] = 1
    img[idx_0] = 0
    return img


def get_train_loader(dataset):
    train_preprocess = TrainPre(config.image_mean, config.image_std)
    train_preprocess_no_crop = TrainPreOri(config.image_mean, config.image_std)

    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config.batch_size * config.niters_per_epoch)

    train_dataset_no_crop = dataset(data_setting, 'train', train_preprocess_no_crop,
                                    config.batch_size * config.niters_per_epoch)

    train_dataset = data.ConcatDataset([train_dataset,train_dataset_no_crop])
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True,
                                   sampler=None)

    return train_loader


def get_val_loader(dataset):
    val_preprocess_no_crop = TrainPreOri(config.image_mean, config.image_std)
    val_dataset = dataset(data_setting, 'val', val_preprocess_no_crop, config.batch_size * config.niters_per_epoch)
    val_loader = data.DataLoader(val_dataset,
                                   batch_size=1,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=False,
                                   pin_memory=True,
                                   sampler=None)

    return val_loader


class TrainPre(object):
    """pre-processing on train set."""
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt):
        img, gt = random_mirror(img, gt) # flip image at random

        gt = img_to_black(gt)

        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array) # scale the image at random

        img = normalize(img, self.img_mean, self.img_std)

        crop_size = (200, 200)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0) # resize cropped images
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        # add resize for down sampling
        p_img = cv2.resize(p_img, (config.image_width // config.gt_down_sampling,
                                 config.image_height // config.gt_down_sampling),
                          interpolation=cv2.INTER_NEAREST)

        p_gt = cv2.resize(p_gt, (config.image_width // config.gt_down_sampling,
                                 config.image_height // config.gt_down_sampling),
                          interpolation=cv2.INTER_NEAREST)

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = None

        return p_img, p_gt, extra_dict

class TrainPreOri(TrainPre):

    def __call__(self, img, gt):
        (img, gt) = random_mirror(img, gt)
        gt = img_to_black(gt)
        if config.train_scale_array is not None:
            (img, gt, scale) = random_scale(img, gt,
                    config.train_scale_array)
        img = normalize(img, self.img_mean, self.img_std)
        (p_img, p_gt) = (img, gt)

        p_img = cv2.resize(p_img, (config.image_width // config.gt_down_sampling,
                           config.image_height // config.gt_down_sampling),
                           interpolation=cv2.INTER_NEAREST)
        p_gt = cv2.resize(p_gt, (config.image_width // config.gt_down_sampling,
                           config.image_height // config.gt_down_sampling),
                           interpolation=cv2.INTER_NEAREST)
        p_img = p_img.transpose(2, 0, 1)
        extra_dict = None
        return (p_img, p_gt, extra_dict)


class OurDataLoader(data.Dataset):
    trans_labels = [0, 1] # binary label

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(OurDataLoader, self).__init__()
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._test_source = setting['test_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        """Retrieve img, gt from directory by index."""
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        img_path = os.path.join(self._img_path, names[0])
        gt_path = os.path.join(self._gt_path, names[1])
        item_name = names[1].split("/")[-1].split(".")[0]

#         img, gt = self._fetch_data(img_path, gt_path)
        img = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR), dtype=None)
        gt = np.array(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE), dtype=None)
        
        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt)

        if self._split_name is 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, label=gt, fn=str(item_name),
                           n=len(self._file_names))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _get_file_names(self, split_name):
        """Obtain filename from tab-separated files."""
        assert split_name in ['train', 'val', 'test']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source
        if split_name == "test":
            source = self._test_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            item = item.strip()
            item = item.split('\t')
            img_name = item[0]
            gt_name = item[1]
            file_names.append([img_name, gt_name])

        return file_names

    def _construct_new_file_names(self, length):
        """Ensure correct name from relative directory"""
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        """Get size of the dataset."""
        return self.__len__()

    @classmethod
    def get_class_colors(*args):
        """color for visualization and saving images."""
        return [[255, 255, 255], [0, 0, 0]]

    @classmethod
    def get_class_names(*args):
        """Label names."""
        return ['road', 'non-road']
