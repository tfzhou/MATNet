import pydensecrf.densecrf as dcrf
import numpy as np
import sys
import time

import os
from tqdm import tqdm
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral,\
    create_pairwise_gaussian, unary_from_softmax

from os import listdir, makedirs
from os.path import isfile, join

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

image_dir = 'data/DAVIS2017/JPEGImages/480p'
davis_result_dir = 'output/davis16'
model_name = 'MATNet_epoch0' # specify the folder name of saliency results
mask_dir = os.path.join(davis_result_dir, model_name)
save_dir = join(davis_result_dir, model_name + '_crf')

for seq in tqdm(listdir(mask_dir)):
    seq_dir = join(image_dir, seq)
    seq_mask_dir = join(mask_dir, seq)
    res_dir = join(save_dir, seq)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    for f in listdir(seq_mask_dir):

        frameName = f[:-4]

        image = imread(join(seq_dir, f[:-4] + '.jpg'))
        mask = imread(join(seq_mask_dir, f))

        H, W = mask.shape

        min_val = np.min(mask.ravel())
        max_val = np.max(mask.ravel())
        out = (mask.astype('float') - min_val) / (max_val - min_val)
        labels = np.zeros((2, image.shape[0], image.shape[1]))
        labels[1, :, :] = out
        labels[0, :, :] = 1 - out

        tau = 1.05
        EPSILON = 1e-8
        anno_norm = mask / 255
        n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))
        labels[1, :, :] = n_energy
        labels[0, :, :] = p_energy

        colors = [0, 255]
        colorize = np.empty((len(colors), 1), np.uint8)
        colorize[:, 0] = colors

        n_labels = 2

        crf = dcrf.DenseCRF(image.shape[1] * image.shape[0], n_labels)

        U = unary_from_softmax(labels)
        crf.setUnaryEnergy(U)

        feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
        crf.addPairwiseEnergy(feats, compat=3,
                              kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        feats = create_pairwise_bilateral(sdims=(30, 30), schan=(5, 5, 5),
                                          img=image, chdim=2)
        crf.addPairwiseEnergy(feats, compat=5,
                              kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q, tmp1, tmp2 = crf.startInference()
        for i in range(5):
            temp = crf.klDivergence(Q)
            crf.stepInference(Q, tmp1, tmp2)
            if abs(crf.klDivergence(Q)-temp) < 500:
                break

        MAP = np.argmax(Q, axis=0)
        MAP = colorize[MAP]

        imsave(res_dir + '/' + frameName + '.png', MAP.reshape(mask.shape))
        #print("Saving: " + res_dir + '/' + frameName + '.png')
