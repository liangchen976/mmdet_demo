# 从多个 base 文件中合并 Config 同时也支持多个 base 文件合并得到最终配置，用户只需要在非 base 配置文件中将类似
# _base_ = './base.py'改成 _base_ = ['./base.py',...]
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
