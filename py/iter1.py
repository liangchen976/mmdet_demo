from collections.abc import Iterable, Iterator
# __getitem__ 实际上是属于 iter和next` 方法的高级封装，也就是我们常说的语法糖，
# 只不过这个转化是通过编译器完成，内部自动转化，非常方便
class A(object):
    def __init__(self):
        self.a = [1, 2, 3]

    def __getitem__(self, item):
        return self.a[item]

cls_a = A()
print(isinstance(cls_a, Iterable))  # False
print(isinstance(cls_a, Iterator))  # False
print(dir(cls_a))  # 仅仅具备 __getitem__ 方法

cls_a = iter(cls_a)
print(dir(cls_a))  # 具备 __iter__ 和 __next__ 方法
# 在调用iter()方法时，自动具备__iter__ 和 __next__方法，从而可迭代了
print(isinstance(cls_a, Iterable))  # True
print(isinstance(cls_a, Iterator))  # True


print('-'*30)
# yield生成器
# 生成器是懒加载模式，特别适合解决内存占用大的集合问题
# 假设创建一个包含10万个元素的列表，如果用 list 返回不仅占用很大的存储空间，如果我们仅仅需要访问前面几个元素，
# 那后面绝大多数元素占用的空间都白白浪费了，这种场景就适合采用生成器，在迭代过程中推算出后续元素，而不需要一次性全部算出
def func():
    for a in [1, 2, 3]:
        yield a

cls_g = func()
print(isinstance(cls_g, Iterator))  # True
print(isinstance(cls_g, Iterable))  # True
print(dir(cls_g))  # 自动具备 __iter__ 和 __next__ 方法

for a in cls_g:
    print(a)

# 下面是dataloader的实现
class Dataset(object):
    # 只要实现了 __getitem__ 方法就可以变成迭代器
    def __getitem__(self, index):
        raise NotImplementedError
    # 用于获取数据集长度
    def __len__(self):
        raise NotImplementedError

# 基类
class Sampler(object):
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
class SequentialSampler(Sampler):

    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        # 返回迭代器，不然无法 for .. in ..
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
class RandomSampler(Sampler):

    def __init__(self, data_source):
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        # 返回迭代器，不然无法 for .. in ..
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        # 采样方式
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        # 调用 sampler 内部的迭代器对象
        for idx in self.sampler:
            batch.append(idx)
            # 如果已经得到了 batch 个 索引，则可以通过 yield
            # 关键字生成生成器返回，得到迭代器对象
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            # 如果最后的索引数不够一个 batch，则抛弃
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
import torch
def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy':
        return default_collate([torch.as_tensor(b) for b in batch])
    else:
        raise NotImplementedError
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, collate_fn=None, drop_last=False):
        self.dataset = dataset

        # 因为这两个功能是冲突的，假设 shuffle=True,
        # 但是 sampler 里面是 SequentialSampler，那么就违背设计思想了
        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # 一旦设置了 batch_sampler，那么 batch_size、shuffle、sampler
            # 和 drop_last 四个参数就不能传入
            # 因为这4个参数功能和 batch_sampler 功能冲突了
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False

        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        # 也就是说 batch_sampler 必须要存在，你如果没有设置，那么采用默认类
        if batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = iter(batch_sampler)

        if collate_fn is None:
            collate_fn = default_collate
        self.collate_fn = collate_fn

    # 核心代码
    def __next__(self):
        index = next(self.batch_sampler)
        data = [self.dataset[idx] for idx in index]
        data = self.collate_fn(data)
        return data

    # 返回自身，因为自身实现了 __next__
    def __iter__(self):
        return self


import numpy as np
class SimpleV1Dataset(Dataset):
    def __init__(self):
        # 伪造数据
        self.imgs = np.arange(0, 16).reshape(8, 2)

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return self.imgs.shape[0]

simple_dataset = SimpleV1Dataset()
dataloader = DataLoader(simple_dataset, batch_size=2, collate_fn=default_collate)
for data in dataloader:
    print(data)