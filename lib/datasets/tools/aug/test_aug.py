import tqdm
from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2
import os
import random
from PIL import Image
from imagecorruptions import corrupt

if __name__ == '__main__':
    input_dir = '/data/liuweixing/NWPU/'
    # for i in range(15):
    corrupt_type = 3
    if corrupt_type:
        # corrupt_severity = 3
        # if corrupt_severity == 3:
        #     save_dir = input_dir+str(corrupt_severity)+'_'+str(corrupt_type+1)+'JPEGImages/'
        # else:
        #     save_dir = input_dir + str(corrupt_type + 1) + 'JPEGImages/'

        corrupt_severity = 3
        save_dir = input_dir + '3_17JPEGImages/'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        images = glob.glob(input_dir+'3_1JPEGImages/*.jpg')
        for index in tqdm.tqdm(range(len(images))):
            name = os.path.basename(images[index])
            image = np.asarray(Image.open(images[index]))
            corrupted = corrupt(image, corruption_number=corrupt_type, severity=corrupt_severity)
            corrupted_bgr = cv2.cvtColor(corrupted, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_dir+name,corrupted_bgr)


