import torch
a = torch.ones(5)
a.requires_grad = True

b = 2*a
# Since b is non-leaf and it's grad will be destroyed otherwise.
#
b.retain_grad()

c = b.mean()

c.backward()

print(a.grad, b.grad)

a = torch.ones(5)

a.requires_grad = True

b = 2*a

b.retain_grad()
# 为tensor注册hook 在反向传播时执行注册函数传入的参数
# 在执行backward时，自动调用触发器hook

b.register_hook(lambda x: print(x))

b.mean().backward()


print(a.grad, b.grad)