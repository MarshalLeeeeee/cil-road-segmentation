import os
import sys
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.modules.batchnorm import BatchNorm2d

from config import config
from dataloader import get_train_loader, get_val_loader, OurDataLoader, data_setting
from network import Backbone_Res101
from util import init_weight, group_weight, normalize, link_file, ensure_dir


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
    
    model = Backbone_Res101(out_planes=config.num_classes)
    init_weight(model.layers, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    params_list = []
    params_list = group_weight(params_list, model.context, BatchNorm2d, config.lr)
    params_list = group_weight(params_list, model.class_refine, BatchNorm2d, config.lr)
    params_list = group_weight(params_list, model.context_refine, BatchNorm2d, config.lr)
    params_list = group_weight(params_list, model.arms, BatchNorm2d, config.lr)
    params_list = group_weight(params_list, model.ffm, BatchNorm2d, config.lr)
    params_list = group_weight(params_list, model.refines, BatchNorm2d, config.lr)
    params_list = group_weight(params_list, model.res_top_refine, BatchNorm2d, config.lr)

    optimizer = torch.optim.Adam(params_list)
    criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=255)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(config.nepochs):
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar_train = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader_train = iter(train_loader)
        loss_sum_train = 0.0
        for idx in pbar_train:
            optimizer.zero_grad()

            minibatch = dataloader_train.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            
            pred = model(imgs)
            loss = criterion(pred, gts)

            loss.backward()
            optimizer.step()

            loss_sum_train += loss.item()
            print_str = '====train====Epoch{}/{}'.format(epoch+1, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx+1, config.niters_per_epoch) \
                        + ' loss=%.4f' % (loss_sum_train / (idx+1))

            pbar_train.set_description(print_str, refresh=False)

        if (epoch%config.checkiter != 0):
            continue

        # save model
        ensure_dir(config.snapshot_dir)
        current_epoch_checkpoint = os.path.join(config.snapshot_dir, 'epoch-{}.pth'.format(epoch))

        state_dict = {}
        new_state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v
        state_dict['model'] = new_state_dict
        torch.save(state_dict, current_epoch_checkpoint)
        del state_dict
        del new_state_dict

        last_epoch_checkpoint = os.path.join(config.snapshot_dir,'epoch-last.pth')
        link_file(current_epoch_checkpoint, last_epoch_checkpoint)

        # evaluate
        model.eval()
        pbar_val = tqdm(range(config.num_eval_imgs), file=sys.stdout, bar_format=bar_format)
        dataloader_val = iter(valid_loader)
        loss_sum_val = 0.0
        for idx in pbar_val:
            minibatch = dataloader_val.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            pred = model(imgs)
            loss = criterion(pred, gts)

            loss_sum_val += loss.item()
            print_str = '=====val=====Epoch{}/{}'.format(epoch+1, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx+1, config.niters_per_epoch) \
                        + ' loss=%.4f' % (loss_sum_val / (idx+1))

            pbar_val.set_description(print_str, refresh=False)
        model.train()

            
if __name__ == "__main__":
    train()