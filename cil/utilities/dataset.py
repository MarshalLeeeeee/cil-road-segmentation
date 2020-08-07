import os
import torch
import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

from cil.utilities.transform import TrainTransform, ValidTransform, EvalTransform
from cil.utilities.visualize import display_training_samples


class CilDataset(Dataset):
    def __init__(self, root_dir, transform, mode='training', train_all=False):
        """
        Labeled dataset. Assume the default layout in root_dir.
        :param root_dir: root directory of data set, containing 'train.txt', etc.
        :param transform: a functor which does transformation (data augmentation)
        :param mode: either 'training' or 'validation'. 'test' dataset is not included as it is unlabeled.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        if mode == 'training':
            if train_all:
                index_path = os.path.join(root_dir, 'all.txt')
            else:
                index_path = os.path.join(root_dir, 'train.txt')
        elif mode == 'validation':
            index_path = os.path.join(root_dir, 'valid.txt')
        else:
            raise ValueError('mode parameter of CilDataset must be in "training" or "validation"')

        with open(index_path, 'r') as index:
            lines = index.readlines()

        self.samples = list()
        for line in lines:
            sample_pair = line.strip().split(' ')
            self.samples.append(sample_pair)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, groundtruth_path = self.samples[idx]

        image = cv2.imread(image_path)
        groundtruth = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)
        edge_mask = None
        image_id = image_path.split('/')[-1].split('.')[0]
        
        if self.transform:
            image, groundtruth, edge_mask = self.transform(image, groundtruth)

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        groundtruth = torch.from_numpy(np.ascontiguousarray(groundtruth)).long()
        edge_mask = torch.from_numpy(np.ascontiguousarray(edge_mask)).float()

        sample = {'image': image, 'groundtruth': groundtruth, 'edge_mask': edge_mask, 'image_id': image_id}

        return sample


class CilTestDataset(Dataset):
    def __init__(self, root_dir, transform):
        """
        Unlabeled dataset. Assume the default layout in root_dir.
        :param root_dir: root directory of data set, containing 'train.txt', etc.
        :param transform: a functor which does input normalization
        """
        self.root_dir = root_dir
        self.transform = transform

        index_path = os.path.join(root_dir, 'test.txt')
        with open(index_path, 'r') as index:
            content = index.read()

        self.samples = content.strip().split('\n')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.samples[idx]
        image = cv2.imread(image_path)
        image_id = image_path.split('/')[-1].split('.')[0]

        if self.transform:
            image = self.transform(image)

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        sample = {'image': image, 'image_id': image_id}

        return sample


def get_training_set(root_dir, train_all=False):
    return CilDataset(root_dir, TrainTransform(), train_all=train_all)


def get_validation_set(root_dir):
    return CilDataset(root_dir, ValidTransform(), 'validation')


def get_evaluation_set(root_dir):
    return CilTestDataset(root_dir, EvalTransform())


def _display_sample(sample):
    display_training_samples(sample['image'], sample['groundtruth'], figure_title=sample['image_id'])


def _test():
    root_dir = 'dataset/'

    training_set = CilDataset(root_dir, None)
    print('training set contains ' + str(len(training_set)) + ' samples. Show some.')
    for i in range(len(training_set)):
        _display_sample(training_set[i])
        if i == 2:
            break

    validation_set = CilDataset(root_dir, None, 'validation')
    print('validation set contains ' + str(len(validation_set)) + ' samples. Show some.')
    for i in range(len(validation_set)):
        _display_sample(validation_set[i])
        if i == 2:
            break

    test_set = CilTestDataset(root_dir, None)
    print('validation set contains ' + str(len(test_set)) + ' samples. Show none.')

    print('done')


if __name__ == "__main__":
    _test()
