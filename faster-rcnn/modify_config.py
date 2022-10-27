from mmcv import Config
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmdet.apis import set_random_seed

# 先引入基配置文件
config = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = '../checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = Config.fromfile(config)
print(f'以下是配置文件的{(len(config.keys()))}个部分')
for item in config.keys():
    print(item)
# model
# dataset_type
# data_root
# img_norm_cfg
# train_pipeline
# test_pipeline
# data
# evaluation
# optimizer
# optimizer_config
# lr_config
# runner
# checkpoint_config
# log_config
# custom_hooks
# dist_params
# log_level
# load_from
# resume_from
# workflow
# opencv_num_threads
# mp_start_method
# auto_scale_lr

# 下面可以按照字典形式个性化修改配置文件 reference:mmdet Doc MMDet_Tutorial.ipynb

config.data_root = 'F:/Obj_detec/coco/'
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
config.optimizer.lr /=8
config.load_from = checkpoint

# Set up working dir to save files and logs.
config.work_dir = './tutorial_exps'

# Set the device to be used for evaluation
device='cuda:0'
# Set seed thus the results are more reproducible
config.seed = 0
set_random_seed(0, deterministic=False)
config.gpu_ids = range(1)

# We can also use tensorboard to log the training process
config.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

print(f'Config:\n{config.pretty_text}') #使用官方api将配置文件打印出来



from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


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
