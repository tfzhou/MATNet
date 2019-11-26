from __future__ import division

import torch
from torch.utils import data

import os
import cv2
import glob
import lmdb
import numpy as np
from PIL import Image
import os.path as osp
from scipy.misc import imresize
from matplotlib import pyplot as plt

from torchvision import transforms
from dataloader import custom_transforms as tr
from .base import Sequence, Annotation
from misc.config import cfg, phase, db_read_sequences

class DAVISLoader(data.Dataset):
    '''
    Dataset for DAVIS
    '''

    def __init__(self, args, split, inputRes, augment=False,
                 transform=None, target_transform=None):
        self._year = args.year
        self._phase = split
        self.transform = transform
        self.target_transform = target_transform
        self.inputRes = inputRes
        self.augment = augment
        self.augment_transform = None
        self._single_object = False

        assert args.year == "2017" or args.year == "2016"

        if augment:
            self.augment_transform = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.ScaleNRotate(rots=(-args.rotation, args.rotation),
                                scales=(.75, 1.25))])

        self._db_sequences = db_read_sequences(args.year, self._phase)

        # Check lmdb existance. If not proceed with standard dataloader.
        lmdb_env_seq_dir = osp.join(cfg.PATH.DATA, 'lmdb_seq')
        lmdb_env_annot_dir = osp.join(cfg.PATH.DATA, 'lmdb_annot')

        if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
            lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
            lmdb_env_annot = lmdb.open(lmdb_env_annot_dir)
        else:
            lmdb_env_seq = None
            lmdb_env_annot = None
            print('LMDB not found. This could affect the data loading time. It is recommended to use LMDB.')

        self.sequences = [Sequence(self._phase, s.name, lmdb_env=lmdb_env_seq) for s in self._db_sequences]
        self._db_sequences = db_read_sequences(args.year, self._phase)

        # Load annotations
        self.annotations = [Annotation(self._phase,s.name, self._single_object, lmdb_env=lmdb_env_annot) for s in self._db_sequences]
        self._db_sequences = db_read_sequences(args.year, self._phase)

        # Load Videos
        self.videos = []
        for seq, s in zip(self.sequences, self._db_sequences):
            if s['set'] == self._phase:
                self.videos.append(s['name'])

        self.imagefiles = []
        self.maskfiles = []
        self.flowfiles = []
        self.edgefiles = []

        for _video in self.videos:
            imagefiles = sorted(glob.glob(os.path.join(cfg.PATH.SEQUENCES, _video, '*.jpg')))
            maskfiles = sorted(glob.glob(os.path.join(cfg.PATH.ANNOTATIONS, _video, '*.png')))
            flowfiles = sorted(glob.glob(os.path.join(cfg.PATH.FLOW, _video, '*.png')))
            edgefiles = sorted(glob.glob(os.path.join(cfg.PATH.ANNOTATIONS_EDGE, _video, '*.png')))

            self.imagefiles.extend(imagefiles[:-1])
            self.maskfiles.extend(maskfiles[:-1])
            self.flowfiles.extend(flowfiles)
            self.edgefiles.extend(edgefiles[:-1])

        print('images: ', len(self.imagefiles))
        print('masks: ', len(self.maskfiles))

        assert(len(self.imagefiles) == len(self.maskfiles) == len(self.flowfiles) == len(self.edgefiles))


    def __len__(self):
        return len(self.imagefiles)


    def __getitem__(self, index):
        imagefile = self.imagefiles[index]
        maskfile = self.maskfiles[index]
        flowfile = self.flowfiles[index]
        edgefile = self.edgefiles[index]

        image = Image.open(imagefile).convert('RGB')
        flow = Image.open(flowfile).convert('RGB')

        mask = cv2.imread(maskfile, 0)
        mask[mask > 0] = 255

        bdry = cv2.imread(edgefile, 0)

        #plt.imshow(bdry)
        #plt.show()

        mask = Image.fromarray(mask)
        bdry = Image.fromarray(bdry)

        if self.inputRes is not None:
            image = imresize(image, self.inputRes)
            flow = imresize(flow, self.inputRes)
            mask = imresize(mask, self.inputRes, interp='nearest')
            bdry = imresize(bdry, self.inputRes, interp='nearest')

        sample = {'image': image, 'flow': flow, 'mask': mask, 'bdry': bdry}

        if self.augment_transform is not None:
            sample = self.augment_transform(sample)

        image, flow, mask, bdry = sample['image'], sample['flow'], sample['mask'], sample['bdry']

        if self.transform is not None:
            image = self.transform(image)
            flow = self.transform(flow)

        if self.target_transform is not None:
            mask = self.target_transform(mask)
            bdry = self.target_transform(bdry)

        return image, flow, mask, bdry
