# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.pascal_voc_cyclewater import pascal_voc_cyclewater
from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart
from datasets.sim10k import sim10k
from datasets.water import water
from datasets.clipart import clipart
from datasets.sim10k_cycle import sim10k_cycle
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape
from datasets.gta_car import gta_car
from datasets.ucas_car import ucas_car
from datasets.dlr3k_car import dlr3k_car
from datasets.dior10 import dior10
from datasets.xview import xview
from datasets.itcvd import itcvd
from datasets.UDImix import UDImix
from datasets.XUNmix import XUNmix
from datasets.ucas_plane import ucas_plane
from datasets.nwpu10 import nwpu10
from datasets.dior import dior
from datasets.dior_plane import dior_plane
from datasets.nwpu import nwpu

from datasets.postdam import postdam
from datasets.vaihingen import vaihingen
from datasets.corsadd import corsadd
from datasets.synu10k import synu10k

from datasets.sat_mtb import sat_mtb
from datasets.dronevehicle_inf import dronevehicle_inf

from datasets.dronevehicle_rgb import dronevehicle_rgb
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'dronevehicle_rgb_{}'.format(split)
    __sets[name] = (lambda split=split : dronevehicle_rgb(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'dronevehicle_inf_{}'.format(split)
    __sets[name] = (lambda split=split : dronevehicle_inf(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'synu10k_{}'.format(split)
    __sets[name] = (lambda split=split : synu10k(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'corsadd_{}'.format(split)
    __sets[name] = (lambda split=split : corsadd(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'vaihingen_{}'.format(split)
    __sets[name] = (lambda split=split : vaihingen(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'postdam_{}'.format(split)
    __sets[name] = (lambda split=split : postdam(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'nwpu_{}'.format(split)
    __sets[name] = (lambda split=split : nwpu(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'ucas_plane_{}'.format(split)
    __sets[name] = (lambda split=split : ucas_plane(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'dior_{}'.format(split)
    __sets[name] = (lambda split=split : dior(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'dior_plane_{}'.format(split)
    __sets[name] = (lambda split=split : dior_plane(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'nwpu10_{}'.format(split)
    __sets[name] = (lambda split=split : nwpu10(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'xview_{}'.format(split)
    __sets[name] = (lambda split=split : xview(split,year))
for split in ['trainval', 'test']:
  name = 'itcvd_{}'.format(split)
  __sets[name] = (lambda split=split : itcvd(split))
for split in ['trainval', 'test']:
  name = 'UDImix_{}'.format(split)
  __sets[name] = (lambda split=split : UDImix(split))
for split in ['trainval', 'test']:
  name = 'XUNmix_{}'.format(split)
  __sets[name] = (lambda split=split : XUNmix(split))
for split in ['train', 'trainval','val','test']:
  name = 'cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape(split))
for split in ['train', 'trainval','val','test']:
  name = 'cityscape_car_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_car(split))
for split in ['train', 'trainval','test']:
  name = 'foggy_cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : foggy_cityscape(split))
for split in ['train','val']:
  name = 'sim10k_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k(split))
  name = 'sim10k_cycle_{}'.format(split)
  __sets[name] = (lambda split=split: sim10k_cycle(split))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_water_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc_water(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cycleclipart_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cyclewater_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cyclewater(split, year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'clipart_{}'.format(split)
    __sets[name] = (lambda split=split : clipart(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'gta_car_{}'.format(split)
    __sets[name] = (lambda split=split : gta_car(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'ucas_car_{}'.format(split)
    __sets[name] = (lambda split=split : ucas_car(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'sat_mtb_{}'.format(split)
    __sets[name] = (lambda split=split : sat_mtb(split,year))

for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'dlr3k_car_{}'.format(split)
    __sets[name] = (lambda split=split : dlr3k_car(split,year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'dior10_{}'.format(split)
    __sets[name] = (lambda split=split : dior10(split,year))
for year in ['2007']:
  for split in ['train', 'test','trainval']:
    name = 'water_{}'.format(split)
    __sets[name] = (lambda split=split : water(split,year))
def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
