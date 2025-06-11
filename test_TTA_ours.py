# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import pdb
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
from model.utils.net_utils import save_checkpoint
from model.utils.loss import gradient_norm_penalty_simple
import torch.nn as nn


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
import torch.nn.functional as F
import math,json

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        out = torch.cat(self.outputs,dim=0)
        out = torch.mean(out, dim=0)
        return out

    def get_out_var(self):
        out = torch.cat(self.outputs,dim=0)
        out = torch.var(out, dim=0)
        return out

def test(dataset, fasterRCNN, num_images):
    TTAdataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                shuffle=False, num_workers=0,
                                                pin_memory=True)

    data_iter = iter(TTAdataloader)

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    _t = {'im_detect': time.time(), 'misc': time.time()}
    output_dir = get_output_dir(imdb, save_name)
    det_file = os.path.join(output_dir, 'detections.pkl')
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    time_start = time.time()
    for img_index in range(num_images):
        try:
            data = next(data_iter)
        except:
            break

        with torch.no_grad():
            fasterRCNN.eval()

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
                rois_label_s_fake, ent, base_feat, _, _, _, _, _, _
            ) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, use_meta=True, vis=False)


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

                all_boxes[j][img_index] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][img_index] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][img_index][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][img_index][:, -1] >= image_thresh)[0]
                    all_boxes[j][img_index] = all_boxes[j][img_index][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                         .format(img_index + 1, num_images, nms_time))
        sys.stdout.flush()


    end = time.time()
    print("test time: %0.4fs" % (end - time_start))
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    source_args = copy.deepcopy(args)
    # source_args.dataset = 'pascal_voc_0712'
    # source_args.dataset = 'pascal_voc_water'
    # source_args.dataset = 'nwpu10'
    # source_args.dataset = 'dior'
    source_args.dataset = 'gta_car'
    # source_args.dataset = 'xview'
    # source_args.dataset = 'ucas_plane'
    # source_args.dataset = 'corsadd'
    # source_args.dataset = 'dronevehicle_rgb'

    args = set_dataset_args(args, test=True)
    source_args = set_dataset_args(source_args, test=False)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False #True

    imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb(source_args.imdbval_name, False)
    imdb_s.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb_s)))

    # initilize the network here.
    from model.TTA_faster_rcnn.resnet_ours import resnet
    from model.TTA_faster_rcnn.vgg16_ours import vgg16

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb_s.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb_s.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb_s.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)

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

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 200

    thresh = 0.0

    save_name = args.load_name.split('/')[-1]
    num_images_s = len(roidb_s)

    dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, 1, \
                             imdb_s.num_classes, training=False, normalize=False)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter_s = iter(dataloader_s)

    chosen_bn_layers = []
    for m in fasterRCNN.modules():
        if isinstance(m, nn.BatchNorm2d):
            chosen_bn_layers.append(m)

    chosen_bn_layers = chosen_bn_layers[-10:]
    n_chosen_layers = len(chosen_bn_layers)
    save_outputs = [SaveOutput() for _ in range(n_chosen_layers)]
    clean_mean_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
    clean_var_act_list = [AverageMeter() for _ in range(n_chosen_layers)]

    clean_mean_list_final = []
    clean_var_list_final = []

    for i in range(num_images_s):
    # for i in range(1000):
        try:
            data = next(data_iter_s)
        except:
            continue
        with torch.no_grad():
            fasterRCNN.eval()
            hook_list = [chosen_bn_layers[i].register_forward_hook(save_outputs[i]) for i in range(n_chosen_layers)]

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
                rois_label_s_fake,ent,base_feat,feat1,feat2,feat3
            ) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            fasterRCNN.feat1 = 0.99 * fasterRCNN.feat1 + feat1 * 0.01
            fasterRCNN.feat2 = 0.99 * fasterRCNN.feat2 + feat2 * 0.01
            fasterRCNN.feat3 = 0.99 * fasterRCNN.feat3 + feat3 * 0.01

            pooled_feat_base = fasterRCNN.RCNN_roi_align(base_feat, rois.view(-1, 5)).mean(3).mean(2)
            cindex = torch.argmax(cls_prob[0], 1)
            for c in range(fasterRCNN.n_classes):
                fg = pooled_feat_base[cindex == c]
                if len(fg) > 0:
                    fasterRCNN.global_pro[c,] = 0.99 * fasterRCNN.global_pro[c,] + 0.01 * torch.mean(fg, 0)

            for j in range(n_chosen_layers):
                clean_mean_act_list[j].update(save_outputs[j].get_out_mean())  # compute mean from clean data
                clean_var_act_list[j].update(save_outputs[j].get_out_var())  # compute variane from clean data

                save_outputs[j].clear()
                hook_list[j].remove()

    for i in range(n_chosen_layers):
        clean_mean_list_final.append(clean_mean_act_list[i].avg)  # [C, H, W]
        clean_var_list_final.append(clean_var_act_list[i].avg)  # [C, H, W]



    save_name = args.load_name.split('/')[-1]
    mmm = 5
    nnn = 5
    if True:
        if True:
            pseuso_label = mmm*0.2
            lambdaweight = nnn*0.2
            use_meta = True
            lr = args.lr
            # for Corrupt_type in range(1,16):
            # Corrupt_type = 17
            Corrupt_type = 0
            if True:
                Loss = []

                fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
                fasterRCNN.init_meta()

                # no meta
                # fasterRCNN.train()

                fasterRCNN.eval()
                for nm, m in fasterRCNN.named_modules():
                    # 0. only bn params
                    # m.eval()
                    # m.requires_grad = False
                    # for p in m.parameters(): p.requires_grad = False
                    # if isinstance(m, nn.BatchNorm2d):
                    #     m.requires_grad = True
                    #     for p in m.parameters(): p.requires_grad = True

                    # 1. only meta
                    if 'meta' in nm:
                       m.requires_grad = True
                       for p in m.parameters(): p.requires_grad = True
                    else:
                       m.eval()
                       m.requires_grad = False
                       for p in m.parameters(): p.requires_grad = False
                    if isinstance(m, nn.BatchNorm2d):
                       m.eval()

                    # 2. bn+meta
                    # m.eval()
                    # m.requires_grad = False
                    # for p in m.parameters(): p.requires_grad = False
                    # if 'meta' in nm:
                    #    m.requires_grad = True
                    #    for p in m.parameters(): p.requires_grad = True
                    # if isinstance(m, nn.BatchNorm2d):
                    #    m.requires_grad = True
                    #    for p in m.parameters(): p.requires_grad = True

                params = []
                for key, value in dict(fasterRCNN.named_parameters()).items():
                    if value.requires_grad:
                        if 'bias' in key:
                            params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                        else:
                            params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

                optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

                imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False, Corrupt_type)
                imdb.competition_mode(on=True)
                num_images = len(imdb.image_index)
                dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                                         imdb.num_classes, training=False, normalize=False)
                TTAdataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                         shuffle=False, num_workers=0,
                                                         pin_memory=True)

                data_iter = iter(TTAdataloader)

                all_boxes = [[[] for _ in xrange(num_images)]
                             for _ in xrange(imdb.num_classes)]
                _t = {'im_detect': time.time(), 'misc': time.time()}
                output_dir = get_output_dir(imdb, save_name)

                det_file = os.path.join(output_dir, 'detections.pkl')
                empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
                time_start = time.time()

                for img_index in range(num_images):
                    vis = False
                    try:
                        data = next(data_iter)
                    except:
                        continue

                    training_samples = 100
                    if img_index<training_samples:


                        # warmup+linear
                        warmup_steps = 10
                        if img_index<warmup_steps:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = (img_index+1)/warmup_steps * lr
                        else:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = (1-(img_index-warmup_steps+1)/(training_samples-warmup_steps)) * lr

                        optimizer.zero_grad()
                        save_outputs_tta = [SaveOutput() for _ in range(n_chosen_layers)]

                        hook_list_tta = [chosen_bn_layers[x].register_forward_hook(save_outputs_tta[x])
                                         for x in range(n_chosen_layers)]

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
                            rois_label_s_fake,eata_ent,base_feat, _, _,feat1,feat2,feat3,_
                        ) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, use_meta=use_meta,vis=vis)
                        batch_mean_tta = [save_outputs_tta[x].get_out_mean() for x in range(n_chosen_layers)]
                        batch_var_tta = [save_outputs_tta[x].get_out_var() for x in range(n_chosen_layers)]

                        loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
                        loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()

                        for i in range(n_chosen_layers):
                            loss_mean += torch.nn.functional.l1_loss(batch_mean_tta[i].cuda(), clean_mean_list_final[i].cuda())
                            loss_var += torch.nn.functional.l1_loss(batch_var_tta[i].cuda(), clean_var_list_final[i].cuda())

                        loss = (loss_mean*lambdaweight + loss_var*lambdaweight)

                        if eata_ent is not None:
                            loss = loss + eata_ent*pseuso_label

                        pooled_feat_base = fasterRCNN.RCNN_roi_align(base_feat, rois.view(-1, 5)).mean(3).mean(2)
                        cindex = torch.argmax(cls_prob[0], 1)
                        pro_loss = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()

                        for c in range(1,fasterRCNN.n_classes):
                            fg_idx = cindex == c
                            fg = pooled_feat_base[fg_idx]
                            if len(fg) > 0:
                                prototype = torch.mean(fg, 0)
                                pro_loss += torch.nn.functional.l1_loss(prototype,fasterRCNN.global_pro[c,])
                        loss = loss + pro_loss*pseuso_label

                        feat_loss = torch.nn.functional.l1_loss(feat1, fasterRCNN.feat1) + \
                                    torch.nn.functional.l1_loss(feat2, fasterRCNN.feat2) + \
                                    torch.nn.functional.l1_loss(feat3, fasterRCNN.feat3)
                        loss = loss + feat_loss*pseuso_label

                        loss.backward()


                        print(loss.item())
                        Loss.append(loss.item())


                        optimizer.step()
                        for z in range(n_chosen_layers):
                            save_outputs_tta[z].clear()
                            hook_list_tta[z].remove()

                     

                    else:
    

                        with torch.no_grad():
                            fasterRCNN.eval()

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
                                rois_label_s_fake, ent, base_feat, base_feat1, base_feat2,feat1,feat2,feat3,_
                            ) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, use_meta=use_meta,vis=vis)

          

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

                            all_boxes[j][img_index] = cls_dets.cpu().numpy()
                        else:
                            all_boxes[j][img_index] = empty_array

                    # Limit to max_per_image detections *over all classes*
                    if max_per_image > 0:
                        image_scores = np.hstack([all_boxes[j][img_index][:, -1]
                                                  for j in xrange(1, imdb.num_classes)])
                        if len(image_scores) > max_per_image:
                            image_thresh = np.sort(image_scores)[-max_per_image]
                            for j in xrange(1, imdb.num_classes):
                                keep = np.where(all_boxes[j][img_index][:, -1] >= image_thresh)[0]
                                all_boxes[j][img_index] = all_boxes[j][img_index][keep, :]

                    misc_toc = time.time()
                    nms_time = misc_toc - misc_tic

                    sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                                     .format(img_index + 1, num_images, nms_time))
                    sys.stdout.flush()


                    # if img_index % 10 == 0:
                    #     print(img_index)
                    #     test(dataset, fasterRCNN, num_images)

                print('Corrupt_type {} TTA Done!'.format(Corrupt_type))

                end = time.time()
                print("test time: %0.4fs" % (end - time_start))
                with open(det_file, 'wb') as f:
                    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

                Loss0 = np.array(Loss)

                print(mmm,nnn)
                print('Evaluating detections')
                imdb.evaluate_detections(all_boxes, output_dir)
