import os
import cv2
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import parse_devices
from engine.evaluator import Evaluator
from engine.logger import get_logger
from dataloader import get_train_loader, OurDataLoader, data_setting
from network import Backbone_Res101

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']

        pred = self.whole_eval(img,
                               output_size=(config.test_image_height,config.test_image_width), 
                               input_size=(config.test_image_height,config.test_image_width),
                               device=device)
        # binary to greyscale
        pred = 237 * pred
        pred = cv2.resize(pred, (608, 608), interpolation=cv2.INTER_NEAREST)

        results_dict = {'rmse': 1}

        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), img)
            fnPred = name + '-pred.png'
            cv2.imwrite(os.path.join(self.save_path, fnPred), pred)
            # logger.info("Save the image " + fn)

        return results_dict

    def compute_metric(self, results):
        """No metric is calculated during prediction."""
        return 'no result'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--input_size', type=str, default='1x3x608x608',
                        help='Input size. '
                             'channels x height x width (default: 1x3x224x224)')
    parser.add_argument('-speed', '--speed_test', action='store_true')
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('-summary', '--summary', action='store_true')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Backbone_Res101(config.num_classes, is_training=False)

    dataset = OurDataLoader(data_setting, 'test', None)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
