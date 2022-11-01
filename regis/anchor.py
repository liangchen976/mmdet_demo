from mmdet.core import AnchorGenerator
# base_sizes: anchor的基本大小
# FPN 输出的多尺度信息可以帮助区分不同大小物体识别问题，每一层就不再需要不包括 FPN 的 Faster R-CNN 算法那么多 anchor 了
# 所以使用FPN时，无需定义多尺度的strides
self = AnchorGenerator(strides=[16], ratios=[0.5, 1.0, 2.0], scales=[2, 4, 8, 16, 32])
# 在 10 * 10 的特征图上生成对应anchor的左上右下坐标 shape = torch.Size([len(ratios) * len(scales) * 100, 4])
# 也就是每个特征图对应15个anchor
all_anchors = self.grid_priors([(10, 10)], device='cpu')
print(all_anchors, all_anchors[0].shape)


