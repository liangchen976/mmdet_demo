from faster_rcnn import *
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from faster_rcnn.modify_config import config

# Build dataset
datasets = [build_dataset(config.data.train)]

# Build the detector
model = build_detector(config.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)

# test_img_dir = '../../test_img/test01.jpg'
test_img_dir = '../../test_img/demo.jpg'
img = mmcv.imread(test_img_dir)

model.cfg = config
result = inference_detector(model, img)
show_result_pyplot(model, img, result, 0.5)