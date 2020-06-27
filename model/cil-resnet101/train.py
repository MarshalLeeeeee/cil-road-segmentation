from __future__ import division
import os
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.modules.batchnorm import BatchNorm2d

from config import config
from dataloader import get_train_loader, OurDataLoader, data_setting
from network import Backbone_Res101
from util import init_weight, group_weight, normalize


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def train():
    cudnn.benchmark = True

    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader = get_train_loader(OurDataLoader)
    valid_loader = get_val_loader(OurDataLoader)
    
    model = Backbone_Res101(out_planes=config.num_classes, is_training=True)

    init_weight(model.layers, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model.context, BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.class_refine, BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.context_refine, BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.arms, BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.ffm, BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.refines, BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.res_top_refine, BatchNorm2d, base_lr)

    optimizer = torch.optim.Adam(params_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(config.nepochs):
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            
            loss = model(imgs, gts)
                
            lr = optimizer.param_groups[0]['lr']

            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.4f' % loss

            pbar.set_description(print_str, refresh=False)

        if (epoch%5==0):
            ensure_dir(config.snapshot_dir)
            current_epoch_checkpoint = os.path.join(config.snapshot_dir, 'epoch-{}.pth'.format(epoch))
            torch.save(model, current_epoch_checkpoint)
            last_epoch_checkpoint = os.path.join(config.snapshot_dir,'epoch-last.pth')
            link_file(current_epoch_checkpoint, last_epoch_checkpoint)
            model.eval()
            for idx in range(config.num_eval_imgs):
                img_val = valid_loader[idx]['data']
                gt_val = valid_loader[idx]['label']
                img_val = normalize(p_img, config.image_mean, config.image_std)
                img_val = img_val.transpose(2, 0, 1)


            
if __name__ == "__main__":
    train()