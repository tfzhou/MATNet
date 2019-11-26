from __future__ import division

from torch.utils import data

import os
import cv2
import glob
import lmdb
import numpy as np
from PIL import Image
import os.path as osp
from scipy.misc import imresize

from torchvision import transforms
from dataloader import custom_transforms as tr
from .base import Sequence, Annotation

from misc.config import cfg as cfg_davis
from misc.config_youtubeVOS import cfg as cfg_youtube
from misc.config import db_read_sequences as db_read_sequences_davis
from misc.config_youtubeVOS import db_read_sequences_train as db_read_sequences_train_youtube


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

        self.imagefiles = []
        self.maskfiles = []
        self.flowfiles = []
        self.edgefiles = []
        self.hedfiles = []

        if split == 'train':
            self.load_davis(args)
            self.load_youtubevos(args)
        else:
            self.load_davis(args)

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, index):
        imagefile = self.imagefiles[index]
        maskfile = self.maskfiles[index]
        flowfile = self.flowfiles[index]
        edgefile = self.edgefiles[index]
        hedfile = self.hedfiles[index]

        image = Image.open(imagefile).convert('RGB')
        flow = Image.open(flowfile).convert('RGB')

        mask = cv2.imread(maskfile, 0)
        mask[mask > 0] = 255

        bdry = cv2.imread(edgefile, 0)
        hed = cv2.imread(hedfile, 0)

        # enlarge the object mask
        kernel = np.ones((11, 11), np.uint8)  # use a large kernel
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        inverse_dilated_mask = (255.0 - dilated_mask) / 255.0
        negative_pixels = hed * inverse_dilated_mask
        kernel = np.ones((5, 5), np.uint8)  # use a small kernel
        negative_pixels = cv2.dilate(negative_pixels, kernel, iterations=1)

        mask = Image.fromarray(mask)
        bdry = Image.fromarray(bdry)
        negative_pixels = Image.fromarray(negative_pixels)

        if self.inputRes is not None:
            image = imresize(image, self.inputRes)
            flow = imresize(flow, self.inputRes)
            mask = imresize(mask, self.inputRes, interp='nearest')
            bdry = imresize(bdry, self.inputRes, interp='nearest')
            negative_pixels = imresize(negative_pixels, self.inputRes,
                                       interp='nearest')

        sample = {'image': image, 'flow': flow, 'mask': mask, 'bdry': bdry,
                  'negative_pixels': negative_pixels}

        if self.augment_transform is not None:
            sample = self.augment_transform(sample)

        image, flow, mask, bdry, negative_pixels =\
            sample['image'], sample['flow'],\
            sample['mask'], sample['bdry'], sample['negative_pixels']

        if self.transform is not None:
            image = self.transform(image)
            flow = self.transform(flow)

        if self.target_transform is not None:
            mask = mask[:, :, np.newaxis]
            bdry = bdry[:, :, np.newaxis]
            negative_pixels = negative_pixels[:, :, np.newaxis]
            mask = self.target_transform(mask)
            bdry = self.target_transform(bdry)
            negative_pixels = self.target_transform(negative_pixels)

        return image, flow, mask, bdry, negative_pixels

    def load_youtubevos(self, args):
        self._db_sequences = db_read_sequences_train_youtube()

        # Check lmdb existance. If not proceed with standard dataloader.
        lmdb_env_seq_dir = osp.join(cfg_youtube.PATH.DATA, 'lmdb_seq')
        lmdb_env_annot_dir = osp.join(cfg_youtube.PATH.DATA, 'lmdb_annot')

        if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
            lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
            lmdb_env_annot = lmdb.open(lmdb_env_annot_dir)
        else:
            lmdb_env_seq = None
            lmdb_env_annot = None
            print('LMDB not found. This could affect the data loading time.'
                  ' It is recommended to use LMDB.')

        # Load sequences
        self.sequences = [Sequence(self._phase, s, lmdb_env=lmdb_env_seq)
                          for s in self._db_sequences]

        # Load sequences
        videos = []
        for seq, s in zip(self.sequences, self._db_sequences):
            videos.append(s)

        for _video in videos:
            imagefile = sorted(glob.glob(os.path.join(
                cfg_youtube.PATH.SEQUENCES_TRAIN, _video, '*.jpg')))
            maskfile = sorted(glob.glob(os.path.join(
                cfg_youtube.PATH.ANNOTATIONS_TRAIN, _video, '*.png')))
            flowfile = sorted(glob.glob(os.path.join(
                cfg_youtube.PATH.FLOW, _video, '*.png')))
            edgefile = sorted(glob.glob(os.path.join(
                cfg_youtube.PATH.ANNOTATIONS_TRAIN_EDGE, _video, '*.png')))
            hedfile = sorted(glob.glob(os.path.join(
                cfg_youtube.PATH.HED, _video, '*.jpg')))

            self.imagefiles.extend(imagefile[:-1:10])
            self.maskfiles.extend(maskfile[:-1:10])
            self.flowfiles.extend(flowfile[::10])
            self.edgefiles.extend(edgefile[:-1:10])
            self.hedfiles.extend(hedfile[:-1:10])

        print('images: ', len(self.imagefiles))
        print('masks: ', len(self.maskfiles))
        print('hed: ', len(self.hedfiles))
        print('flow: ', len(self.flowfiles))
        print('edge: ', len(self.edgefiles))

        assert(len(self.imagefiles) == len(self.maskfiles) ==
               len(self.flowfiles) == len(self.edgefiles) ==
               len(self.hedfiles))

    def load_davis(self, args):
        self._db_sequences = db_read_sequences_davis(args.year, self._phase)

        # Check lmdb existance. If not proceed with standard dataloader.
        lmdb_env_seq_dir = osp.join(cfg_davis.PATH.DATA, 'lmdb_seq')
        lmdb_env_annot_dir = osp.join(cfg_davis.PATH.DATA, 'lmdb_annot')

        if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
            lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
            lmdb_env_annot = lmdb.open(lmdb_env_annot_dir)
        else:
            lmdb_env_seq = None
            lmdb_env_annot = None
            print('LMDB not found. This could affect the data loading time.'
                  ' It is recommended to use LMDB.')

        self.sequences = [Sequence(self._phase, s.name, lmdb_env=lmdb_env_seq)
                          for s in self._db_sequences]
        self._db_sequences = db_read_sequences_davis(args.year, self._phase)

        # Load annotations
        self.annotations = [Annotation(
            self._phase, s.name, self._single_object, lmdb_env=lmdb_env_annot)
            for s in self._db_sequences]
        self._db_sequences = db_read_sequences_davis(args.year, self._phase)

        # Load Videos
        videos = []
        for seq, s in zip(self.sequences, self._db_sequences):
            if s['set'] == self._phase:
                videos.append(s['name'])

        for _video in videos:
            imagefile = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.SEQUENCES, _video, '*.jpg')))
            maskfile = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.ANNOTATIONS, _video, '*.png')))
            flowfile = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.FLOW, _video, '*.png')))
            edgefile = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.ANNOTATIONS_EDGE, _video, '*.png')))
            hedfile = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.HED, _video, '*.jpg')))

            self.imagefiles.extend(imagefile[:-1])
            self.maskfiles.extend(maskfile[:-1])
            self.flowfiles.extend(flowfile)
            self.edgefiles.extend(edgefile[:-1])
            self.hedfiles.extend(hedfile[:-1])

        print('images: ', len(self.imagefiles))
        print('masks: ', len(self.maskfiles))
        print('hed: ', len(self.hedfiles))
        print('flow: ', len(self.flowfiles))
        print('edge: ', len(self.edgefiles))

        assert(len(self.imagefiles) == len(self.maskfiles) ==
               len(self.flowfiles) == len(self.edgefiles) ==
               len(self.hedfiles))