import torch
# 此文件是验证mmdet里的forward部分 forward_dummy
outs = ()
for i in range(10):
    # 括弧里单元素加, 便是还是以元组形式存储，只不过是单个元素
    outs = outs + (3, )
    # outs = outs + (3, 4)
print(outs)
# dict的update方法 更新键值对 覆盖与添加
a = {'one': 1, 'two': 2, 'three': 3}
a.update({'one':4.5, 'four': 9.3})
print(a)
# 判断是否采样 返回真假
loss_cls = dict()
loss_cls.update({'type': 'None'})
sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
print(sampling)