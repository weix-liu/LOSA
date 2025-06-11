from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__D = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg_d = __D
#
# Training options
#with regard to pascal, the directories under the path will be ./VOC2007, ./VOC2012"
__D.PASCAL = "/dat01/liuweixing/DA/DA_Detection-master/VOCdevkit"
#__D.PASCAL = "/dat01/liuweixing/DA/DA_Detection-master/ucas"
__D.PASCALCLIP = ""
__D.PASCALWATER = "/dat01/liuweixing/DA/DA_Detection-master/VOCdevkit"

#For these datasets, the directories under the path will be Annotations  ImageSets  JPEGImages."
__D.CLIPART = "/dat01/liuweixing/DA/DA_Detection-master/clipart"
__D.KITTI = "/dat01/liuweixing/DA/DA_Detection-master/kitti"
__D.WATER = "/dat01/liuweixing/DA/DA_Detection-master/watercolor"
__D.SIM10K = "/dat01/liuweixing/DA/DA_Detection-master/sim10k"
__D.CITYSCAPE_CAR = "/dat01/liuweixing/DA/DA_Detection-master/cityscape"
__D.CITYSCAPE = "/dat01/liuweixing/DA/DA_Detection-master/cityscape"
__D.FOGGYCITY = "/dat01/liuweixing/DA/DA_Detection-master/FOGGY_cityscape"

__D.GTA_CAR = "/dat01/liuweixing/DA/DA_Detection-master/GTA_CAR"
__D.UCAS_CAR = "/dat01/liuweixing/DA/DA_Detection-master/UCAS_CAR"
__D.DIOR_CAR = "/dat01/liuweixing/DA/DA_Detection-master/DIOR_CAR"
__D.HRSSD_CAR = "/dat01/liuweixing/DA/DA_Detection-master/HRSSD_CAR"
__D.CARPK_CAR = "/dat01/liuweixing/DA/DA_Detection-master/CARPK_CAR"

__D.DLR3K_CAR="/dat01/liuweixing/DA/DA_Detection-master/DLR3K_CAR"
__D.VISDRONEnir = "/dat01/liuweixing/DA/DA_Detection-master/VISDRONEnir"
__D.ITCVD = "/dat01/liuweixing/DA/DA_Detection-master/ITCVD"
__D.LEVIR = "/dat01/liuweixing/dataset/LEVIR_VOC"
__D.XVIEW = "/dat01/liuweixing/dataset/XviewVOC"

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __D)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __D
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
