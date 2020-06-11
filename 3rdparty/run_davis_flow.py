import torch
import glob
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from run import estimate
import flow_vis, cv2 

def main():
    davis_folder = '/home/tianfei/dataset/DAVIS2017/JPEGImages/480p'
    save_dir = './davis2017'

    videos = os.listdir(davis_folder)

    for idx, video in enumerate(videos):
        print('process {}[{}/{}]'.format(video, idx, len(videos)))
        save_dir_video = os.path.join(save_dir, video)
        if not os.path.exists(save_dir_video):
            os.makedirs(save_dir_video)

        imagefiles = sorted(glob.glob(os.path.join(davis_folder, video, '*.jpg')))

        for i in range(len(imagefiles)-1):
            f1 = imagefiles[i]
            f2 = imagefiles[i+1]

            save_name = os.path.basename(f1)[:-4] + '_' + os.path.basename(f2)[:-4] + '.png'
            save_file = os.path.join(save_dir_video, save_name)
            run(f1, f2, save_file)


def run(imagefile1, imagefile2, save_file):
	tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(imagefile1))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
	tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(imagefile2))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

	tensorOutput = estimate(tensorFirst, tensorSecond)

	flow_color = flow_vis.flow_to_color(tensorOutput.numpy().transpose(1,2,0), convert_to_bgr=True)
	cv2.imwrite(save_file, flow_color)

	#objectOutput = open(save_file, 'wb')

	#numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
	#numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
	#numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)

	#objectOutput.close()


if __name__ == '__main__':
    main()
