import numpy as np
import cv2

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

def _get_eval_sets():
    f = open("dataset/test.txt")
    test_dirs = f.readlines()
    image_ids = []
    for dir in test_dirs:

        image_ids.append(dir[20:-1])
        dir = dir[:-1]

    # eval_dirs = []
    # for id in image_ids:
    #     eval_dirs.append("../../../cil-road-segmentation-2020/evaluation/" + id)

    return test_dirs, image_ids

class DenseCRF:
    # img should be h*w*c
    def __call__(self, img, labels, is_label = True):
        n_labels = 2
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        if is_label :
            U = unary_from_labels(labels, n_labels, gt_prob = 0.7, zero_unsure=False)
            d.setUnaryEnergy(U)
        else:
            U = unary_from_softmax(labels.transpose([2, 0, 1])) #-np.log(labels).transpose([2, 0, 1]).reshape([n_labels, -1]).astype(np.float32)
            d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=(3, 3), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=(10, 10), srgb=(3, 3, 3), rgbim=img,
                                compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(5)
        return np.argmax(Q, axis = 0).reshape([img.shape[0], img.shape[1]]), np.array(Q).transpose([1, 0]).reshape([img.shape[0], img.shape[1], -1])

def _validation():
    valid_img_ids = ["026", "029", "032", "045", "063", "069", "078", "080", "083", "094"]
    valid_pred_dir = "/home/wzrain/cil/evaluation/"
    valid_img_dir = "/home/wzrain/cil/cil-road-segmentation-2020/training/training/images/"
    valid_gt_dir = "/home/wzrain/cil/cil-road-segmentation-2020/training/training/groundtruth/"

    # valid_pred = []
    # valid_img = []
    # valid_gt = []
    for id in valid_img_ids:
        labels = cv2.imread(valid_pred_dir + "satImage_" + id + ".png", cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(valid_img_dir + "satImage_" + id + ".png")
        gt = cv2.imread(valid_gt_dir + "satImage_" + id + ".png", cv2.IMREAD_GRAYSCALE)

        print(labels.shape)
        labels = labels / 255.0
        softmax = np.ones([labels.shape[0], labels.shape[1], 2])
        print(softmax[:,:,1].shape)
        softmax[:,:,1] = labels
        softmax[:,:,0] = 1 - labels

        denseCrf=DenseCRF()
        _, smx = denseCrf(img, softmax, False)
        smx = smx[:,:,1] * 255
        cv2.imwrite("/home/wzrain/cil/valid_crf/satImage_" + id + ".png", smx.astype(np.uint8))
        # cv2.imwrite("/home/wzrain/cil/valid_gt/satImage_" + id + ".png", gt.astype(np.uint8))

        # valid_pred.append(pred)
        # valid_img.append(img)
        # valid_gt.append(gt)


def dense_crf(is_label=True):
    test_dirs,image_ids = _get_eval_sets()
    for i in range(len(test_dirs)):
        img = cv2.imread("dataset/test_images/" + image_ids[i])
        print(img.shape)
        labels = cv2.imread("evaluation/" + image_ids[i], cv2.IMREAD_GRAYSCALE)
        if is_label:
            labels[labels <= 255 * 0.25] = 0
            labels[labels >= 255 * 0.25] = 1
            denseCrf = DenseCRF()
            _, smx = denseCrf(img, labels)

            smx = smx[:,:,1] * 255
            cv2.imwrite("evalution/" + image_ids[i], smx.astype(np.uint8))
        else:
            labels = labels / 255.0 #+ 1 / 255.0
            # print(np.unique(labels))
            softmax = np.ones([labels.shape[0], labels.shape[1], 2])
            print(softmax[:,:,1].shape)
            softmax[:,:,1] = labels
            softmax[:,:,0] = 1 - labels
            print(softmax)

            denseCrf=DenseCRF()
            _, smx = denseCrf(img, softmax, False)
            smx = smx[:,:,1] * 255
            cv2.imwrite("evaluation/" + image_ids[i], smx.astype(np.uint8))



def _test():
    img = cv2.imread("../../../cil-road-segmentation-2020/test_images/test_images/test_136.png")

    labels = cv2.imread("../../../cil-road-segmentation-2020/evaluation/test_136.png", cv2.IMREAD_GRAYSCALE)
    # print(img.shape, labels.shape)
    # print(np.unique(labels))
    labels[labels <= 255 * 0.25] = 0
    labels[labels >= 255 * 0.25] = 1
    # labels[labels == 1] = 255
    # cv2.imshow('labels', labels)
    # cv2.waitKey(0)

    dense_crf = DenseCRF()
    refined_img, smx = dense_crf(img, labels)


    smx = smx[:,:,1]
    print(smx.shape)
    print(np.unique(smx))

    smx = smx * 255
    print(np.unique(smx.astype(np.uint8)))
    cv2.imshow('img', smx.astype(np.uint8))
    cv2.waitKey(0)

    # print(np.unique(refined_img))
    # print(len(refined_img[refined_img == 1]))
    # refined_img[refined_img == 1] = 255
    # print(refined_img.shape)

    # cv2.imshow('img', refined_img.astype(np.uint8))
    # cv2.waitKey(0)
    # mpimg.imshow(refined_img, cmap='gray')

if __name__ == '__main__':
    # _get_eval_sets()
    dense_crf(False)