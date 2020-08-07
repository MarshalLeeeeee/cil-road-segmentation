import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cil.models.unet_multi_dilation import UNetSegMultiDilation
from cil.utilities.checkpoint import CheckpointManager
from cil.utilities.dataset import get_evaluation_set
from cil.utilities.train import bn_eps, bn_momentum, dataset_root_dir, batch_size
from cil.utilities.tta import TTA
from cil.utilities.utils import ensure_dir, OutputType, get_name

# output directory
model_dir = 'trained_models/'
eval_dir = 'evaluation/'

# model loading: from savepoint (one saved after full training) or checkpoint (intermediate state)
load_from_checkpoint = True
# savepoint config
savepoint_name = 'UNetSegMultiDilation'
# checkpoint config
checkpoint_name = 'UNetSegMultiDilation-5000'

# post processing
do_binarize = False
threshold = 0.25

# if using tta
use_tta = True
# whether do tta by cropping and recovery
tta_crop = False


def _get_eval_dataloader():
    dataset = get_evaluation_set(dataset_root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def _load_model(model):
    if load_from_checkpoint:
        assert checkpoint_name.startswith(get_name(model))
        checkpoint_manager = CheckpointManager(model_dir)
        checkpoint_manager.load_model(checkpoint_name, model)
    else:
        assert savepoint_name.startswith(get_name(model))
        savepoint_path = os.path.join(model_dir, savepoint_name)
        model.load_state_dict(torch.load(savepoint_path))


def evaluate():
    ensure_dir(eval_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSegMultiDilation(in_channels=3, out_channels=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
    model.to(device)
    _load_model(model)
    model.eval()

    dataloader = _get_eval_dataloader()
    with torch.no_grad():
        for sample in dataloader:
            # Load data
            images = sample['image'].to(device)
            image_ids = sample['image_id']

            # Make predictions
            def predict(tensor): return _to_prob_maps(model(tensor), model.output_type)
            if use_tta:
                prob_maps = TTA.augmented_predict(predict, images, tta_crop=tta_crop)
            else:
                prob_maps = predict(images)

            # Post processing
            results = _post_process(prob_maps)

            # Output as images
            for result, image_id in zip(results, image_ids):
                save_path = os.path.join(eval_dir, image_id + '.png')
                cv2.imwrite(save_path, result)

    print("Evaluation finished.")


def _post_process(prob_maps):
    prob_maps = prob_maps.squeeze(dim=1)
    prob_maps = prob_maps.to('cpu').numpy()

    if do_binarize:
        results = np.zeros_like(prob_maps, dtype='uint8')
        results[prob_maps > threshold] = 255
    else:
        results = prob_maps * 255
        results = results.astype('uint8')
    return results


def _to_prob_maps(outputs, output_type):
    """ return a 4D tensor (number of channels = 1) and ensure that the output is an numpy ndarray in CPU """

    if output_type == OutputType.LOGIT:
        prob_maps = nn.functional.softmax(outputs, dim=1)[:, 1:2]
    else:
        prob_maps = outputs

    return prob_maps


if __name__ == "__main__":
    evaluate()
