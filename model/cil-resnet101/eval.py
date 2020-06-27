import os
import cv2
import argparse
import numpy as np

import torch

from config import config
from utils.pyt_utils import parse_devices
from engine.evaluator import Evaluator
from engine.logger import get_logger
from dataloader import get_train_loader, OurDataLoader, data_setting
from network import Backbone_Res101

logger = get_logger()

# add pixel-wise RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

def img_to_black(img, threshold=50):
    """Binary filter on greyscale image."""
    img = img.astype(np.int64)
    idx = img[:,:] > threshold
    idx_0 = img[:,:] <= threshold
    img[idx] = 1
    img[idx_0] = 0
    return img

def img_to_uint8(img, threshold=0.50, patch_size = 16):
    img = img_to_black(img)
    """Reads a single image and outputs the strings that should go into the submission file"""
    for j in range(0, img.shape[1], patch_size):
        for i in range(0, img.shape[0], patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            if np.mean(patch) > threshold:
                img[i:i + patch_size, j:j + patch_size] = np.ones_like(patch)
            else:
                img[i:i + patch_size, j:j + patch_size] = np.zeros_like(patch)
    return img

class SegEvaluator(Evaluator):
    """Evalation based on RMSE between gt and predicted labels."""

    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']

        img = cv2.resize(img, (config.image_width, config.image_height),
                         interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label,
                           (config.image_width // config.gt_down_sampling,
                            config.image_height // config.gt_down_sampling),
                           interpolation=cv2.INTER_NEAREST)

        pred = self.whole_eval(img,
                               (config.image_height // config.gt_down_sampling,
                                config.image_width // config.gt_down_sampling),
                               device)
        pred = 237 * pred
        
        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), img)
            fnl = name + '-label.png'
            cv2.imwrite(os.path.join(self.save_path, fnl), label)
            fnp = name + '-pred.png'
            cv2.imwrite(os.path.join(self.save_path, fnp), pred)

        results_dict = {'rmse': np.sqrt(mean_squared_error(pred, label))}

        return results_dict

    def compute_metric(self, results):
        """Calculate the RMSE per images."""
        count = 0
        rmse = 0
        for d in results:
            count += 1
            rmse += d['rmse']

        rmse_print = 'average RMSE for {} images: {}\n'.format(count, rmse / count)
        print(rmse_print)
        return rmse_print

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--input_size', type=str, default='1x3x400x400',
                        help='Input size. '
                             'channels x height x width (default: 1x3x224x224)')
    parser.add_argument('-speed', '--speed_test', action='store_true')
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('-summary', '--summary', action='store_true')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Backbone_Res101(config.num_classes, is_training=False)

    dataset = OurDataLoader(data_setting, 'val', None)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
