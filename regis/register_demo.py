# 本demo是学习mmdet中的注册机制 通过类名来实现类的实例化
# 首先实例化一个注册器类
import mmcv
CATS = mmcv.Registry('cat')


@CATS.register_module()
class BritishShorthair:
    print('this is a instance of BritishShorthair')
# 类实例化
CATS.get('BritishShorthair')

@CATS.register_module(name='pig')
class SiameseCat:
    print('this is a instance of SiameseCat')
# 类实例化
CATS.get('pig')
