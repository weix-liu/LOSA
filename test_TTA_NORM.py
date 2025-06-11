# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pprint
import time
import _init_paths
import torch
from torch.autograd import Variable
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
# from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.parser_func import parse_args, set_dataset_args

import pdb


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
import torch.nn as nn
import torch.nn.functional as F


class BayesianBatchNorm(nn.Module):
    """Use the source statistics as a prior on the target statistics"""

    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            print(name)
            # child.requires_grad_(False)
            for p in child.parameters(): p.requires_grad = False

            if isinstance(child, nn.BatchNorm2d):
                module = BayesianBatchNorm(child, prior)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BayesianBatchNorm.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        replace_mods = BayesianBatchNorm.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for parent, name, child in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0).cuda()

        self.prior = prior

    def forward(self, input):
        self.norm(input)

        running_mean = (
            self.prior * self.layer.running_mean
            + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
            self.prior * self.layer.running_var
            + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )

def adapt_bayesian(model, prior):
    return BayesianBatchNorm.adapt_model(model, prior=prior)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args, test=True)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    # initilize the network here.
    from model.TTA_faster_rcnn.resnet import resnet
    from model.TTA_faster_rcnn.vgg16 import vgg16

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (args.load_name))
    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'],strict=False)
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # TTA
    fasterRCNN.eval()
    fasterRCNN = adapt_bayesian(fasterRCNN, 8./9)
    if args.cuda:
        fasterRCNN.cuda()

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    max_per_image = 200

    thresh = 0.0

    save_name = args.load_name.split('/')[-1]


    # for Corrupt_type in range(16):
    Corrupt_type = 17
    if True:

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False, Corrupt_type)
        imdb.competition_mode(on=True)
        num_images = len(imdb.image_index)
        output_dir = get_output_dir(imdb, save_name)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                                 imdb.num_classes, training=False, normalize=False)

        TTAdataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=True)

        data_iter = iter(TTAdataloader)

        _t = {'im_detect': time.time(), 'misc': time.time()}
        det_file = os.path.join(output_dir, 'detections.pkl')
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        rois_temp = torch.zeros(1, 1).cuda()
        time_start = time.time()
        for i in range(num_images):
            try:
                data = next(data_iter)
            except:
                continue
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).zero_()
            num_boxes.data.resize_(data[3].size()).zero_()

            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls_s_fake,
                rpn_loss_box_s_fake,
                RCNN_loss_cls_s_fake,
                RCNN_loss_bbox_s_fake,
                rois_label_s_fake,_,_,_
            ) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)



            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            misc_tic = time.time()

            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                             .format(i + 1, num_images, nms_time))
            sys.stdout.flush()

        print('Corrupt_type {} TTA Done!'.format(Corrupt_type))

        end = time.time()
        print("test time: %0.4fs" % (end - time_start))
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        imdb.evaluate_detections(all_boxes, output_dir)