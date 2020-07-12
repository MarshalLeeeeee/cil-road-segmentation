import os
import cv2
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from collections import OrderedDict

from config import config
from dataloader import get_val_loader, OurDataLoader
from network import Backbone_Res101
from util import normalize_reverse, ensure_dir

def eval():

    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    valid_loader = get_val_loader(OurDataLoader)

    model = Backbone_Res101(out_planes=config.num_classes)

    ensure_dir(config.snapshot_dir)
    checkpoint = os.path.join(config.snapshot_dir, 'epoch-last.pth')

    state_dict = torch.load(checkpoint)
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=255)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar_eval = tqdm(range(config.num_eval_imgs), file=sys.stdout, bar_format=bar_format)
    dataloader_eval = iter(valid_loader)
    loss_sum_eval = 0.0
    for idx in pbar_eval:
        minibatch = dataloader_eval.next()
        imgs = minibatch['data']
        gts = minibatch['label']

        imgs = imgs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)

        pred = model(imgs)
        loss = criterion(pred, gts)

        imgs_sv = imgs.cpu().detach().numpy().squeeze().transpose((1,2,0))
        imgs_sv = normalize_reverse(imgs_sv, config.image_mean, config.image_std)
        gts_sv = gts.cpu().detach().numpy().squeeze() * 255
        pred_sv = pred.cpu().detach().numpy().squeeze().transpose((1, 2, 0)).argmax(2) * 237

        name = str(idx)
        fn = name + '-img-eval.png'
        cv2.imwrite(os.path.join('.', fn), imgs_sv)
        fnl = name + '-label-eval.png'
        cv2.imwrite(os.path.join('.', fnl), gts_sv)
        fnp = name + '-pred-eval.png'
        cv2.imwrite(os.path.join('.', fnp), pred_sv)

        loss_sum_eval += loss.item()
        print_str = '=====val=====Iter{}/{}:'.format(idx + 1, config.num_eval_imgs) \
                    + ' loss=%.4f' % (loss_sum_eval / (idx+1))

        pbar_eval.set_description(print_str, refresh=False)


if __name__ == "__main__":
    eval()