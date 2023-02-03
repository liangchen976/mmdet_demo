import torch
from torchvision.models import resnet34

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = resnet34(pretrained=True)
model = model.to(device)
# 有36个卷积
# print(model)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

save_output = SaveOutput()

hook_handles = []

for layer in model.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

from PIL import Image
from torchvision import transforms as T
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open(dir).convert('RGB')
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
X = transform(image).unsqueeze(dim=0).to(device)
out = model(X)
# print(len(save_output.outputs))

import matplotlib.pyplot as plt
def module_output_to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()
# 结果为0到35的卷积层
images = module_output_to_numpy(save_output.outputs[30])
print(f'save output len is {len(save_output.outputs)})')
#这里的0代表读取output里第一个卷积层的输出

with plt.style.context("seaborn-white"):
    plt.figure(figsize=(20, 20), frameon=False)
    for idx in range(16):
        plt.subplot(4, 4, idx+1)
        plt.imshow(images[0, idx])
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()
