import os
import pathlib
import shutil
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pythonlibs.my_torch_lib import (
    evaluate_history,
    fit,
    modify_network_head,
    save_history_to_csv,
    show_images_labels,
    show_incorrect_images_labels,
    torch_seed,
)
from utils.const import model_mapping
from utils.load_save import load_config

plt.rcParams["font.size"] = 18
plt.tight_layout()

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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

which_data = config.which_data

root_dir = os.getcwd()

train_dir = os.path.join(
    root_dir, "data", config.which_data, config.train_data
)
test_dir = os.path.join(root_dir, "data", config.which_data, config.test_data)

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

train_data = datasets.ImageFolder(train_dir, transform=test_transform)

test_data = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    shuffle=False,
)

test_loader_for_check = DataLoader(
    test_data, batch_size=50, num_workers=2, pin_memory=True, shuffle=True
)

net = model_mapping[config.net](weights=config.weights)


torch_seed()

if config.transfer:
    for param in net.parameters():
        param.requires_grad = False

modify_network_head(net, len(classes), config.net)


# vitを使うときはこれ
# fc_in_features = net.heads.head.in_features
# net.heads.head = nn.Linear(fc_in_features, len(classes))

# resnetなどを使うときはこっち
# fc_in_features = net.fc.in_features
# net.fc = nn.Linear(fc_in_features, len(classes))

# vggなどを使うときはこっち
# in_features = net.classifier[6].in_features
# net.classifier[6] = nn.Linear(in_features, len(classes))
# net.avgpool = nn.Identity()

net = net.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum)

history = np.zeros((0, 11))


num_epochs = config.num_epochs

history = fit(
    net,
    optimizer,
    criterion,
    num_epochs,
    classes,
    train_loader,
    test_loader,
    device,
    history,
    program_name,
    save_dir,
    which_data,
    True,
    True,
)

save_history_to_csv(history, save_dir)

evaluate_history(history, save_dir, config.train_data)

show_images_labels(
    test_loader_for_check,
    classes,
    net,
    device,
    program_name + config.train_data,
    save_dir,
)

show_incorrect_images_labels(
    test_loader_for_check,
    classes,
    net,
    device,
    save_dir,
)

# 終了時間を記録
end_time = time.time()

# 実行時間を計算して表示
execution_time = end_time - start_time
# ファイルを開く
with open(f"{save_dir}/abst.txt", "a") as f:
    # ファイルに出力する
    print("実行時間:", execution_time, "秒", file=f)
