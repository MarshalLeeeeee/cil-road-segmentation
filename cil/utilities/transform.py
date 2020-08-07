import random

import numpy as np
import cv2

# from cil.utilities.visualize import display_training_samples

# mean and std from ImageNet
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# mean and std of training set
mean1 = np.array([0.330, 0.327, 0.293])
std1 = np.array([0.183, 0.176, 0.175])

binarize_threshold = 40

max_rotation_angle = 15
# probability of being enlarged, unchanged, shrunk during training (0.7, 0.2, 0.1)
p_rescaling = [0.5, 0.6]
crop_size = [(200, 200), (250, 250), (300, 300), (350, 350)]
shrunk_range = (0.6, 0.95)
use_reflect_padding = True
# random translation after shrinking
max_translation_ratio = 0.15


def random_horizontal_flip(img, gt, u=0.5):
    if np.random.random() < u:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)

    return img, gt


def random_vertical_flip(img, gt, u=0.5):
    if np.random.random() < u:
        img = cv2.flip(img, 0)
        gt = cv2.flip(gt, 0)

    return img, gt


def random_scale(img, gt, pad_reflect=False):
    sample = np.random.random()

    if sample < p_rescaling[0]:
        return _crop_and_rescale(img, gt)
    elif sample < p_rescaling[1]:
        return _random_shift(img, gt)
    else:
        return _shrink_and_pad(img, gt, pad_reflect)


def random_rotate_90(img, gt):
    # Number of times the array is rotated by 90 degrees. Allow 0, 90, 180 and 270
    k = np.random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k)
        gt = np.rot90(gt, k)

    return img, gt


# Minor, random rotation
def random_minor_rotate(img, gt):
    angle = np.random.uniform(-max_rotation_angle, max_rotation_angle)
    img = _rotate_image(img, angle)
    gt = _rotate_image(gt, angle)
    return img, gt


def normalize(img):
    img = img.astype(np.float32) / 255.0
    img = img - mean1
    img = img / std1

    return img


def discretize(gt, threshold=40):
    # The order matters
    gt[gt < threshold] = 0
    gt[gt >= threshold] = 1

    return gt


def get_edge_mask(image):
    """ Accept image before binarization """
    edge_mask = cv2.Canny(image, 0, 255)
    edge_mask[image < binarize_threshold] = 0
    edge_mask[edge_mask != 0] = 1
    return edge_mask


# Data augmentation, used in test time
class TrainTransform:
    def __call__(self, img, gt):
        img, gt = random_horizontal_flip(img, gt)
        img, gt = random_vertical_flip(img, gt)
        img, gt = random_rotate_90(img, gt)
        img, gt = random_minor_rotate(img, gt)
        img, gt = random_scale(img, gt)

        mask = get_edge_mask(gt)
        img = normalize(img)
        gt = discretize(gt, binarize_threshold)

        return img, gt, mask


# Basic normalization, used in validation
class ValidTransform:
    def __call__(self, img, gt):
        mask = get_edge_mask(gt)
        img = normalize(img)
        gt = discretize(gt, binarize_threshold)

        return img, gt, mask


# Basic normalization, used in evaluation
class EvalTransform:
    def __call__(self, img):
        return normalize(img)


# debug tool
def _show_image(image, title='image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)


def _crop_and_rescale(img, gt):
    original_shape = img.shape[:2]

    # cropping
    crop_shape = random.choice(crop_size)
    img, gt = _random_crop(img, gt, crop_shape)

    # rescale
    img = cv2.resize(img, original_shape, interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, original_shape, interpolation=cv2.INTER_LINEAR)

    return img, gt


def _random_shift(img, gt):
    original_shape = img.shape[:2]
    max_translation = np.multiply(max_translation_ratio, original_shape).astype(np.int64)

    delta_h = random.randint(-max_translation[0], max_translation[0])
    delta_w = random.randint(-max_translation[1], max_translation[1])

    img = _shift_w(_shift_h(img, delta_h), delta_w)
    gt = _shift_w(_shift_h(gt, delta_h), delta_w)

    return img, gt


def _shift_h(img, delta):
    if delta == 0:
        return img

    translated_img = np.empty_like(img)
    if delta >= 0:
        translated_img[:delta] = 0
        translated_img[delta:] = img[:-delta]
    elif delta < 0:
        translated_img[:delta] = img[-delta:]
        translated_img[delta:] = 0

    return translated_img


def _shift_w(img, delta):
    if delta == 0:
        return img

    translated_img = np.empty_like(img)
    if delta >= 0:
        translated_img[:, :delta] = 0
        translated_img[:, delta:] = img[:, :-delta]
    elif delta < 0:
        translated_img[:, :delta] = img[:, -delta:]
        translated_img[:, delta:] = 0

    return translated_img


def _shrink_and_pad(img, gt, pad_reflect):
    original_shape = img.shape[:2]

    ratio = np.random.uniform(shrunk_range[0], shrunk_range[1])
    img, gt = _shrink(img, gt, ratio)
    img, gt = _pad(img, gt, original_shape, pad_reflect)

    return img, gt


def _random_crop(img, gt, crop_shape):
    original_shape = img.shape[:2]

    # choose cropped region randomly
    h0_max, w0_max = np.subtract(original_shape, crop_shape)
    h0, w0 = np.random.randint(0, h0_max), np.random.randint(0, w0_max)

    # cropping
    cropped_img = img[h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    cropped_gt = gt[h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]

    return cropped_img, cropped_gt


def _shrink(img, gt, ratio):
    original_shape = img.shape[:2]

    shrunk_shape = np.multiply(original_shape, ratio)
    shrunk_shape = (int(shrunk_shape[0]), int(shrunk_shape[1]))

    # shrink
    img = cv2.resize(img, shrunk_shape, interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, shrunk_shape, interpolation=cv2.INTER_LINEAR)

    return img, gt


def _pad(img, gt, target_shape, pad_reflect):
    """ Add random padding to image and gt so that the shape will be `target_shape` after padding """
    original_shape = img.shape[:2]

    # put to center and padding
    margin = np.subtract(target_shape, original_shape)

    # random translation: limited by max_ratio and remained margin
    max_translation = np.multiply(max_translation_ratio, original_shape)
    max_translation = np.minimum((margin // 2), max_translation)
    max_translation = max_translation.astype(np.int64)

    # place image with random translation
    pad_top = margin[0] // 2 + random.randint(-max_translation[0], max_translation[0])
    pad_left = margin[1] // 2 + random.randint(-max_translation[1], max_translation[1])
    pad_bottom, pad_right = margin[0] - pad_top, margin[1] - pad_left

    # padding to original size
    if pad_reflect:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        gt = cv2.copyMakeBorder(gt, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    else:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        gt = cv2.copyMakeBorder(gt, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    return img, gt


def _rotate_image(img, angle):
    if -1 < angle < 1:
        return img

    shape_2d = (img.shape[1], img.shape[0])
    center_2d = (img.shape[1] / 2, img.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_2d, angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, shape_2d, flags=cv2.INTER_LINEAR)
    return img
