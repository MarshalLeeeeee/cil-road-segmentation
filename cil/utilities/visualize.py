import os

import cv2
import numpy as np
from matplotlib import pyplot
from matplotlib.image import imread


def display_training_samples_by_path(image_path, groundtruth_path, wait_seconds=0):
    display_image_files_in_row(
        titles=["image", "groundtruth"],
        paths=[image_path, groundtruth_path],
        wait_seconds=wait_seconds
    )


def display_training_samples(image, groundtruth, wait_seconds=0, figure_title=None):
    display_images_in_row(
        titles=["image", "groundtruth"],
        images=[image, groundtruth],
        wait_seconds=wait_seconds,
        figure_title=figure_title
    )


def display_image_files_in_row(titles, paths, wait_seconds=0, figure_title=None):
    assert len(titles) == len(paths)

    images = list()
    for path in paths:
        images.append(imread(path, 'png'))
    display_images_in_row(titles, images, wait_seconds, figure_title=figure_title)


def display_images_in_row(titles, images, wait_seconds=0, figure_title=None):
    pyplot.ion()
    assert len(titles) == len(images)

    figure = pyplot.figure()
    figure.tight_layout()
    if figure_title is not None:
        figure.suptitle(figure_title)

    num_of_images = len(titles)
    for i in range(num_of_images):
        axis = figure.add_subplot(1, num_of_images, i + 1)
        axis.set_title(titles[i])
        # color image or grayscale images
        if len(images[i].shape) == 3:
            axis.imshow(images[i])
        else:
            axis.imshow(images[i], cmap='gray')
        axis.axis('off')

    # choose interactive mode
    if wait_seconds == -1:
        # blocked wait
        pyplot.show(block=True)
    elif wait_seconds == 0:
        # wait for newline keystroke
        pyplot.show(block=False)
        input()
        pyplot.close(figure)
    else:
        # timed wait
        pyplot.show(block=False)
        pyplot.pause(wait_seconds)
        pyplot.close(figure)


def compare_image_dir(base_dir, dirs):
    file_names = os.listdir(base_dir)
    for name in file_names:
        titles, images = list(), list()
        titles.append(base_dir)
        images.append(imread(os.path.join(base_dir, name), 'png'))
        for directory in dirs:
            titles.append(directory)
            images.append(imread(os.path.join(directory, name), 'png'))
        display_images_in_row(titles, images)


def display_images_heat_map_by_path(paths):
    images = []
    for path in paths:
        images.append(cv2.imread(path))
    display_heat_map(images)


def display_heat_map(images):
    num_of_images = len(images)
    img = None
    for i in range(num_of_images):
        heat_map = None
        heat_map = cv2.normalize(images[i], heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

        border_size = 10
        border = cv2.copyMakeBorder(
            heat_map,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

        if i == 0:
            img = border
        else:
            img = np.concatenate((img, border), axis=1)

    cv2.imshow("heat_map", img)
    cv2.waitKey(0)


def _test_sample_display():
    compare_image_dir('dataset/test_images', ['4000/', '5000/', '6000/'])
    display_training_samples_by_path("dataset/training/images/satImage_001.png",
                                     "dataset/training/groundtruth/satImage_001.png")
    display_training_samples_by_path("dataset/training/images/satImage_002.png",
                                     "dataset/training/groundtruth/satImage_002.png")


def _test_heatmap():
    # Example1: display activation map which is already in memory
    resolution = (400, 500)

    # Generate activation, and map to [0, 1] range
    x, y = np.arange(resolution[0]), np.arange(resolution[1])
    activation = np.outer(x, y)
    activation = activation / float((resolution[0] - 1) * (resolution[1] - 1))

    display_heat_map([activation])

    # Example2: compare grayscale images
    display_images_heat_map_by_path(['validation/satImage_026.png', 'dataset/training/groundtruth/satImage_026.png'])


if __name__ == "__main__":
    _test_heatmap()
