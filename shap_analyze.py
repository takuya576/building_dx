import os
import pathlib
import shutil
import sys
import time
from datetime import datetime
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.const import model_mapping
from utils.load_save import load_config

plt.rcParams["font.size"] = 18
plt.tight_layout()

load_path = os.path.join(
    os.path.expanduser("~/dx"), "result/dataset0a/2024-07-22_19-55-46"
)

# configでCNN、ハイパーパラメータや使用するデータを指定
config = load_config(
    config_path=pathlib.Path(
        (os.path.join(os.path.expanduser("~/dx"), "config/config.toml"))
    )
)

# 開始時間を記録
start_time = time.time()

args = sys.argv
program_name = args[0].split(".")[0].split("/")[-1]
batch_size = config.batch_size
device = torch.device(
    f"cuda:{int(config.nvidia)}" if torch.cuda.is_available() else "cpu"
)

which_data = config.which_data

root_dir = os.getcwd()

train_dir = os.path.join(
    root_dir, "data", config.which_data, config.train_data
)
test_dir = os.path.join(root_dir, "data", config.which_data, config.test_data)

sample_dir = os.path.join(root_dir, "data", config.which_data, "sample2")

# Get the current date and time
now = datetime.now()
Date = now.strftime("%Y-%m-%d")
Time = now.strftime("%H-%M-%S")

# Create the directory name
when = f"{Date}_{Time}"

save_dir = os.path.join(
    os.path.join(os.path.expanduser("~/dx"), "result"), which_data, when
)
os.makedirs(save_dir, exist_ok=True)

# 実行時jsonを保存する
shutil.copy(
    src=os.path.join(os.path.expanduser("~/dx"), "config/config.toml"),
    dst=save_dir,
)


test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

# remove augumentation in train
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)


classes = sorted(os.listdir(train_dir))
test_classes = sorted(os.listdir(sample_dir))
print(f"classes: {classes}")
print(f"test_classes: {test_classes}")

train_data = datasets.ImageFolder(train_dir, transform=test_transform)

test_data = datasets.ImageFolder(test_dir, transform=test_transform)

sample_data = datasets.ImageFolder(sample_dir, transform=test_transform)

test_loader_for_check = DataLoader(
    test_data, batch_size=50, num_workers=2, pin_memory=True, shuffle=True
)

sample_loader = DataLoader(
    sample_data, batch_size=5, num_workers=2, pin_memory=True, shuffle=True
)

model_path = os.path.join(
    load_path,
    "epoch24.pth",
)

net = torch.load(model_path, map_location=device)

# ReLU レイヤーを in-place でない版に変更
for module in net.modules():
    if isinstance(module, nn.ReLU):
        module.inplace = False

for images, labels in test_loader_for_check:
    images = images.to(device)
    labels = labels.to(device)
    break

background = images[:47].requires_grad_(True)

# test_images = images[47:].requires_grad_(True)
# test_labels = labels[47:].cpu().numpy()

for test_images, test_labels in sample_loader:
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    break
print(test_labels)

test_labels = test_labels.cpu().numpy()

net.eval()
explainer = shap.DeepExplainer(net, background)
shap_values = explainer.shap_values(test_images)

shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))
# cpu()：cpu上に移す,　detach()：勾配情報を削除する
test_numpy = np.swapaxes(
    np.swapaxes(test_images.cpu().detach().numpy().copy(), 1, -1), 1, 2
)
test_numpy = (test_numpy + 1) / 2

# plot the feature attributions
true_labels = [test_classes[i] for i in test_labels]
outputs = net(test_images)
pred_labels = [classes[i] for i in torch.max(outputs, 1)[1]]
print(f"true_labels: {true_labels}, pred_labels: {pred_labels}")

true_pred_labels = []
for true_label, pred_label in zip(true_labels, pred_labels):
    true_pred_labels.append(true_label + ":" + pred_label)

all_labels = [classes for i in range(len(test_labels))]
all_labels = np.array(all_labels)
print(f"all_labels: {all_labels[0]}")

plt.rcParams["font.size"] = 8
# print(len(shap_numpy))
shap.image_plot(
    shap_values=shap_numpy,
    pixel_values=test_numpy,
    labels=all_labels,
    true_labels=true_pred_labels,
    show=False,
)

plt.savefig(os.path.join(save_dir, "shap.png"))
plt.close()
