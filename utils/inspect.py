import inspect
import torch

class A:
    def __init__(self):
        self.a = [1, 2, 3]

    def __iter__(self):
        pass

# 查看是否为类 注意参数不是实例化对象
a = A()
print(a, type(a))
print(inspect.isclass(A))
print(inspect.isclass(a))


# 关于self(x)
# 因为self指向的是类本身,()又是call方法的调用 所以

class B:
    def __init__(self, x):
        self.data = x
        print(f'初始化为{self.data}')

    def __call__(self, x):
        print(f'do the call method, input is {x}')
        return 1


    def forward(self, x):
        print('开始输出')
        out = self(x)
        print(out, type(out))



b = B(11)
print(b.forward(3))