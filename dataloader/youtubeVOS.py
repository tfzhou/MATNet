#TODO

from collections import namedtuple

import os
import cv2
import numpy as np

from PIL import Image
from .base_youtube import Sequence, SequenceClip, Annotation, AnnotationClip, BaseLoader, Segmentation, SequenceClip_simple, AnnotationClip_simple
from misc.config_youtubeVOS import cfg,phase,db_read_sequences_train,db_read_sequences_val, db_read_sequences_test, db_read_sequences_trainval
import os.path as osp
import glob
import lmdb

from scipy.misc import imresize

from torch.utils import data
from torchvision import transforms
from dataloader import custom_transforms as tr

class YoutubeVOSLoader(data.Dataset):
    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split = 'train',
                 inputRes = None):

        self._phase = split
        self._single_object = args.single_object
        self._length_clip = args.length_clip
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.inputRes = inputRes
        self.max_seq_len = args.gt_maxseqlen
        self.dataset = args.dataset
        self.flip = augment

        if augment:
            self.augment_transform = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.ScaleNRotate(rots=(-args.rotation, args.rotation), scales=(.75, 1.25))])
        else:
            self.augment_transform = None

        if self._phase == phase.TRAIN.value:
            self._db_sequences = db_read_sequences_train()
        elif self._phase == phase.VAL.value:
            self._db_sequences = db_read_sequences_val()
        elif self._phase == phase.TRAINVAL.value:
            self._db_sequences = db_read_sequences_trainval()
        else: #self._phase == 'test':
            self._db_sequences = db_read_sequences_test()

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

        # Load sequences
        self.sequences = [Sequence(self._phase, s, lmdb_env=lmdb_env_seq) for s in self._db_sequences]

        # Load annotations
        self.annotations = [Annotation(self._phase,s,self._single_object, lmdb_env=lmdb_env_annot) for s in self._db_sequences]

        # Load sequences
        self.videos = []
        for seq, s in zip(self.sequences, self._db_sequences):
            self.videos.append(s)

        self.imagefiles = []
        self.maskfiles = []

        for _video in self.videos:
            imagefiles = sorted(glob.glob(os.path.join(cfg.PATH.SEQUENCES_TRAIN, _video, '*.jpg')))
            maskfiles = sorted(glob.glob(os.path.join(cfg.PATH.ANNOTATIONS_TRAIN, _video, '*.png')))

            self.imagefiles.extend(imagefiles)
            self.maskfiles.extend(maskfiles)
        print('images: ', len(self.imagefiles))
        print('masks: ', len(self.maskfiles))

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, index):
        imagefile = self.imagefiles[index]
        maskfile = self.maskfiles[index]

        image = Image.open(imagefile).convert('RGB')
        mask = cv2.imread(maskfile, 0)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

        if self.inputRes is not None:
            image = imresize(image, self.inputRes)
            mask = imresize(mask, self.inputRes, interp='nearest')

        sample = {'image': image, 'gt': mask}

        if self.augment_transform is not None:
            sample = self.augment_transform(sample)

        image, mask = sample['image'], sample['gt']
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask