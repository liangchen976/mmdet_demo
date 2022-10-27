from faster_rcnn import *
from faster_rcnn.modify_config import config
# Build dataset
datasets = [build_dataset(config.data.train)]

# Build the detector
model = build_detector(config.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
import os.path as osp

mmcv.mkdir_or_exist(osp.abspath(config.work_dir))
train_detector(model, datasets, config, distributed=False, validate=True)