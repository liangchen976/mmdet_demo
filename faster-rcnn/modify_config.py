from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

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
