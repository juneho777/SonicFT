import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sklearn
from sklearn import linear_model
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Linear, Sequential, ReLU
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from myresnet import resnet18
from torch.optim import Adam, SGD
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

root = "/home/xc429/datasets/ILSVRC12/train"

t1 = create_transform(
    input_size=224,
    is_training=True,
    color_jitter=0.4,
    auto_augment='rand-m9-mstd0.5-inc1',
    interpolation='bicubic',
    re_prob=0.25,
    re_mode='pixel',
    re_count=1,
    mean=[0, 0, 0],
    std=[1, 1, 1],
)


class FFTT(object):
    def __init__(self, kernel_size=8, crop_size=4):
        self.kernel_size = kernel_size
        self.crop_size = crop_size

    def __call__(self, x):
        x = x.permute(1, 2, 0)
        x = x.unfold(0, self.kernel_size, self.kernel_size).unfold(1, self.kernel_size, self.kernel_size)
        x = torch.fft.fft2(x)
        h, w = x.shape[:2]
        real = x.real[:, :, :, :self.crop_size, :self.crop_size].contiguous().view(h, w,
                                                                                   3 * self.crop_size * self.crop_size)
        imag = x.imag[:, :, :, :self.crop_size, :self.crop_size].contiguous().view(h, w,
                                                                                   3 * self.crop_size * self.crop_size)
        return torch.cat([real, imag], dim=2).permute(2, 0, 1)


transform = transforms.Compose([t1, FFTT()])

# x = Image.open("/home/xc429/datasets/ILSVRC12/train/n02058221/n02058221_13479.JPEG")
# y = transform(x)
# print(y.shape)


epochs = 300
batch_size = 64
lr = 4e-3
weight_decay = 0.05


dataset = ImageFolder(root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)


from myresnet import resnet18
import torch.optim as optim
import math


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

model = resnet18().cuda()
model = torch.nn.DataParallel(model)

optimizer = optim.AdamW(model.module.parameters(), lr=lr, weight_decay=weight_decay)
lr_schedule_values = cosine_scheduler(
    lr, 1e-6, epochs, len(dataset) // batch_size,
    warmup_epochs=20, warmup_steps=-1,
)
wd_schedule_values = cosine_scheduler(
    weight_decay, weight_decay, epochs, len(dataset) // batch_size)

criterion = torch.nn.CrossEntropyLoss()


for epoch in range(epochs):
    for it, (data, label) in enumerate(dataloader):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule_values[it]
            param_group["weight_decay"] = wd_schedule_values[it]
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (it + 1) % 10 == 0:
            print(f"epoch {epoch+1}, it {it+1}, acc {float(torch.mean((torch.argmax(output, dim=1) == label).float()))*100:0.2f}%, loss {float(loss.item()):0.4f}")
    torch.save(model.state_dict(), os.path.join("checkpoints", f"epoch{epoch}.pth"))
