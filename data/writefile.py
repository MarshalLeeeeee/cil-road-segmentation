import os
from os import walk
import glob
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trainall', default=False, action='store_true')
parser.add_argument('-v', '--verbose', default=True, action='store_true')
parser.add_argument('-s', '--size', default=0.9, action='store', type=int)

args = parser.parse_args()

dataset_path = os.path.dirname(os.path.abspath(__file__))

gt = []
img = []
test = []
# get img
for name in glob.glob('training/images/satImage*'):
    img.append(name)
# get gt
for name in glob.glob('training/groundtruth/satImage*'):
    gt.append(name)
# get test
for name in glob.glob('test_images/*'):
    test.append(name)

img.sort()
gt.sort()
test.sort()

train_size = args.size
assert len(img) == len(gt)
idx = np.arange(len(img))
train_idx = np.random.choice(idx, int(train_size * len(img)), replace=False)
val_idx = list(set(idx) - set(train_idx))
train_idx.sort()
val_idx.sort()

if args.trainall:
    train_idx = idx

if args.verbose:
    print("Writing images from {}".format(dataset_path))
    print("{:<10d} images {:<10d} groundtruth".format(len(img), len(gt)))
    print("training   size: {0:.0%}".format(train_size))
    print("validation size: {0:.0%}".format(1 - train_size))

with open('train.txt', 'w') as f:
    for i in train_idx:
        line = img[i] + '\t' + gt[i] + '\n'
        f.write(line)
if args.verbose:
    print("Writing training   {:3} images to {}".format(len(train_idx), f.name))
f.close()

with open('val.txt', 'w') as f:
    for i in val_idx:
        line = img[i] + '\t' + gt[i] + '\n'
        f.write(line)
if args.verbose:
    print("Writing validation {:3} images to {}".format(len(val_idx), f.name))
f.close()

with open('test.txt', 'w') as f:
    for i in range(len(test)):
        line = test[i] + '\t' + test[i] + '\n'
        f.write(line)
if args.verbose:
    print("Writing test       {:3} images to {}".format(len(test), f.name))
f.close()