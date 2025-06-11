from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.TTA_faster_rcnn.faster_rcnn_ours import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class LayerNorm(nn.Module):
  """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
  The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
  shape (batch_size, height, width, channels) while channels_first corresponds to inputs
  with shape (batch_size, channels, height, width).
  """

  def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(normalized_shape))
    self.bias = nn.Parameter(torch.zeros(normalized_shape))
    self.eps = eps
    self.data_format = data_format
    if self.data_format not in ["channels_last", "channels_first"]:
      raise NotImplementedError
    self.normalized_shape = (normalized_shape,)

  def forward(self, x):
    if self.data_format == "channels_last":
      return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    elif self.data_format == "channels_first":
      u = x.mean(1, keepdim=True)
      s = (x - u).pow(2).mean(1, keepdim=True)
      x = (x - u) / torch.sqrt(s + self.eps)
      x = self.weight[:, None, None] * x + self.bias[:, None, None]
      return x


class SE(nn.Module):
  def __init__(self, dim, outdim,n=1):
    super(SE, self).__init__()

    self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)  # depthwise conv
    self.norm = LayerNorm(dim, eps=1e-6)
    self.avg_pool = nn.AdaptiveAvgPool2d(n)
    self.pwconv1 = nn.Linear(dim, dim)
    self.act = nn.ReLU()
    self.grn = LayerNorm(dim, eps=1e-6)
    self.pwconv2 = nn.Linear(dim, outdim)

  def forward(self, input):

    x = self.avg_pool(input)
    x = self.dwconv(x)
    x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    x = self.norm(x)
    x = self.pwconv1(x)
    x = self.act(x)
    x = self.grn(x)
    x = self.pwconv2(x)
    x = x.permute(0, 3, 1, 2)
    x = input + x

    return x

class SE1(nn.Module):
  def __init__(self, dim, outdim):
    super(SE1, self).__init__()
    r = 16
    self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
    self.norm0 = LayerNorm(dim, eps=1e-6)
    self.conv1 = nn.Conv2d(dim,dim//r,kernel_size=1,padding=0)
    self.norm = LayerNorm(dim//r)
    self.act = nn.ReLU()
    self.conv2 = nn.Conv2d(dim//r,outdim,kernel_size=1,padding=0)

  def forward(self, input):
    x = self.dwconv(input)
    x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    x = self.norm0(x)
    x = x.permute(0, 3, 1, 2)
    x = self.conv1(x)
    x = x.permute(0, 2, 3, 1)
    x = self.norm(x)
    x = x.permute(0, 3, 1, 2)
    x = self.conv2(self.act(x)) + input
    return x

class SE_BN(nn.Module):
  def __init__(self, dim, outdim,n=1):
    super(SE_BN, self).__init__()
    self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)  # depthwise conv
    self.norm = nn.BatchNorm2d(dim)
    self.avg_pool = nn.AdaptiveAvgPool2d(n)
    self.pwconv1 = nn.Conv2d(dim, dim, 1)
    self.act = nn.ReLU()
    self.grn = nn.BatchNorm2d(dim)
    self.pwconv2 = nn.Conv2d(dim, outdim, 1)


  def forward(self, input):
    x = self.avg_pool(input)
    x = self.dwconv(x)
    x = self.norm(x)
    x = self.pwconv1(x)
    x = self.act(x)
    x = self.grn(x)
    x = self.pwconv2(x)

    x = input + x
    return x
from torch.nn.modules.batchnorm import _BatchNorm


def group_norm(input, group, running_mean, running_var, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5):
    r"""Applies Group Normalization for channels in the same group in each data sample in a
    batch.

    See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
    :class:`~torch.nn.GroupNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(input, group, running_mean=None, running_var=None, weight=None,
                       bias=None, use_input_stats=None, momentum=None, eps=None):
        # Repeat stored stats and affine transform params if necessary
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)

        #norm_shape = [1, b * c / group, group]
        #print(norm_shape)
        # Apply instance norm
        input_reshaped = input.contiguous().view(1, int(b * c/group), group, *input.size()[2:])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight=weight, bias=bias,
            training=use_input_stats, momentum=momentum, eps=eps)

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(b, int(c/group)).mean(0, keepdim=False))
        if running_var is not None:
            running_var_orig.copy_(running_var.view(b, int(c/group)).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])
    return _instance_norm(input, group, running_mean=running_mean,
                          running_var=running_var, weight=weight, bias=bias,
                          use_input_stats=use_input_stats, momentum=momentum,
                          eps=eps)


class _GroupNorm(_BatchNorm):
    def __init__(self, num_features, num_groups=1, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(int(num_features/num_groups), eps,
                                         momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)


class GroupNorm2d(_GroupNorm):
    r"""Applies Group Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    https://arxiv.org/pdf/1803.08494.pdf
    `Group Normalization`_ .

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        num_groups:
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = GroupNorm2d(100, 4)
        >>> # With Learnable Parameters
        >>> m = GroupNorm2d(100, 4, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)

    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, input):
    return input

class SE_GN(nn.Module):
  def __init__(self, dim, outdim,n=1):
    super(SE_GN, self).__init__()
    self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)  # depthwise conv
    # self.norm = nn.BatchNorm2d(dim)
    self.norm = GroupNorm2d(dim,32, affine=True)
    self.avg_pool = nn.AdaptiveAvgPool2d(n)
    self.pwconv1 = nn.Conv2d(dim, dim, 1)
    self.act = nn.ReLU()
    # self.grn = nn.BatchNorm2d(dim)
    self.grn = GroupNorm2d(dim,32, affine=True)
    self.pwconv2 = nn.Conv2d(dim, outdim, 1)


  def forward(self, input):
    x = self.avg_pool(input)
    x = self.dwconv(x)
    x = self.norm(x)
    x = self.pwconv1(x)
    x = self.act(x)
    x = self.grn(x)
    x = self.pwconv2(x)

    x = input + x
    return x
class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = cfg.RESNET_PATH
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.layers = num_layers
    if self.layers == 50:
      self.model_path = '/data/liuweixing/DA/resnet50_caffe.pth'
    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = resnet101()
    if self.layers == 50:
      resnet = resnet50()
    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
    self.RCNN_base2 = nn.Sequential(resnet.layer2)
    self.RCNN_base3 = nn.Sequential(resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base1[0].parameters(): p.requires_grad = False
    for p in self.RCNN_base1[1].parameters(): p.requires_grad = False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base3.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base2.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base1[-1].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base1.apply(set_bn_fix)
    self.RCNN_base2.apply(set_bn_fix)
    self.RCNN_base3.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def init_meta(self):
    self.meta1_se = SE(256, 256).cuda()
    self.meta2_se = SE(512, 512).cuda()

    # self.meta1_se = SE1(256, 256).cuda()
    # self.meta2_se = SE1(512, 512).cuda()

    # self.meta1_se = SE_GN(256, 256).cuda()
    # self.meta2_se = SE_GN(512, 512).cuda()

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base1.eval()
      self.RCNN_base2.train()
      self.RCNN_base3.train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base1.apply(set_bn_eval)
      self.RCNN_base2.apply(set_bn_eval)
      self.RCNN_base3.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7