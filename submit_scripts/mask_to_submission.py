#!/usr/bin/env python3

import os
import time

import numpy as np
import matplotlib.image as mpimg
import re

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch


# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


eval_dir = 'evaluation/'
submission_dir = 'submission/'

if __name__ == '__main__':
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    timestamp = time.localtime()
    time_string = time.strftime("%m-%d_%H-%M", timestamp)
    submission_file_name = time_string + '.csv'
    submission_file_path = os.path.join(submission_dir, submission_file_name)

    image_file_names = os.listdir(eval_dir)
    image_file_paths = map(lambda name: os.path.join(eval_dir, name), image_file_names)

    masks_to_submission(submission_file_path, *image_file_paths)

    print('Encoding finished. (Output: {})'.format(submission_file_path))
