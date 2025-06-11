# import cv2
# tiff = 'C:\\Users\\19331\\PycharmProjects\\mypaddlelite_demo_picodet_nano\\0.tif'
# img = cv2.imread(tiff)
# img = cv2.resize(img,None,fx=0.1,fy=0.1)
# cv2.imwrite('C:\\Users\\19331\\PycharmProjects\\mypaddlelite_demo_picodet_nano\\1.jpg',img)
#


import torch
import torch.nn.functional as F
# def euclidean_dist(x, y):
#     # x: N x D
#     # y: M x D
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)
#
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#
#     return torch.pow(x - y, 2).sum(2)
#
# def pro_classifier(x,centroids):
#     # x = torch.randn(34, 256)
#     # centroids = torch.randn(2, 256)
#     dists = euclidean_dist(x,centroids)/10
#     log_p_y = F.softmax(-dists, dim=1)
#     return log_p_y


# import numpy as np
# if __name__ == '__main__':
#     # errors = '39.6 38.4 39.5 29.1 41.5 30.0 29.1 34.0 33.2 40.2 26.4 31.5 36.4 31.4 38.9 34.6'
#     # errors = errors.split(' ')
#     # errors = [float(e) for e in errors]
#     errors = [38.9 ,      38.0  ,        39.7    ,     28.4   ,   40.5    ,    29.0   ,   28.3 ,33.9,33.9,  35.1    ,   26.6  ,  29.2  ,           36.4   ,  30.4      ,       38.3]
#     acc = [100. - e for e in errors]
#     print(acc,len(acc))
#     print(np.asarray(acc[:15]).mean())
#     res = ''
#     for a in acc:
#         res = res + '&' + str(a) + '    '
#     print(res)

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from matplotlib import pyplot as plt
def torch_feat_show(img,name=None):
    img = img.permute(1, 0, 2, 3).detach().cpu()
    img = torchvision.utils.make_grid(img, nrow=16)
    img = img.numpy()
    img = np.uint8(img * 10 + 128)

    # img = img/2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))
    # plt.show()
    # plt.savefig(save_dir + save_name, dpi=1000, bbox_inches='tight')

def plt_feat_show(feat, out_channal=256):
    rows = int(out_channal / 8)
    columns = 8
    fig, axes = plt.subplots(rows, columns)

    for row in range(rows):
        for column in range(columns):
            axis = axes[row][column]
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            axis.imshow(feat[0,row * 8 + column,:,:])
    plt.show()

img1 = cv2.imread('C:\\Users\\19331\\Documents\\dataset\\NWPUVHR-10\\positive image set\\285.jpg')
# feat2 = cv2.imread('C:\\Users\\19331\\PycharmProjects\\DA_Detection-master\\lib\\datasets\\tools\\temp\\after\\24.png')
feat2 = cv2.imread('C:\\Users\\19331\\PycharmProjects\\DA_Detection-master\\testing\\1\\27clean_15.png')
# feat1 = cv2.resize(feat1,(1200,425))
feat2 = cv2.resize(feat2,(1264,987))
# img_b = 0.3*img1 + 0.7*feat1
img_after = 0.3*img1 + 0.7*feat2
cv2.imwrite('clean285source.png', np.uint8(img_after))
# cv2.imwrite('noise285adaptation.png', np.uint8(img_after))