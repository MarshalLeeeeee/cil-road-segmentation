import os
import numpy as np
import matplotlib.image as mpimg
import re
import glob
import argparse

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
    tmp = os.path.basename(image_filename) # prevent digits in the path
    img_number = int(re.search(r"\d+", tmp).group(0))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--image_path', default='pred', type=str)
    parser.add_argument('-n', '--name', default='submission', type=str)

    args = parser.parse_args()

    submission_filename = args.name + '.csv'
    image_filenames = []
    # cnt = 0
    for i in glob.glob(args.image_path + '*png'):
        if i[-5] == 'd':
            image_filenames.append(i)
            # cnt += 1
            # print(cnt, i)
        # image_filenames.append(i)

    masks_to_submission(submission_filename, *image_filenames)
