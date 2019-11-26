# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

""" Compute Jaccard Index. """

import numpy as np
import matplotlib.pyplot as plt


def db_eval_iou_multi(annotations, segmentations):
    iou = 0.0
    batch_size = annotations.shape[0]

    for i in range(batch_size):
        annotation = annotations[i, 0, :, :]
        segmentation = segmentations[i, 0, :, :]

        iou += db_eval_iou(annotation, segmentation)

    iou /= batch_size
    return iou


def db_eval_iou(annotation,segmentation):

    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation = annotation > 0.5
    segmentation = segmentation > 0.5

    if np.isclose(np.sum(annotation), 0) and\
            np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation), dtype=np.float32)