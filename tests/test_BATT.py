'''
This is the test code of benign training and poisoned training under BadNets.
'''


import os
import sys
import os.path as osp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, Normalize, ToPILImage, Resize
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import core

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = './data'
'''
# ============== cifar10 ==============
# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    ToPILImage(),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_10', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
transform_test = Compose([
    ToPILImage(),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

batt = core.BATT(
    train_dataset=trainset,
    test_dataset=testset,
    # model=core.models.ResNet(50),
    model=core.models.vgg19_bn(num_classes=10),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = batt.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': './result/',
    'experiment_name': 'batt_cifar10_vgg19'
}

# batt.train(schedule)
# infected_model = batt.get_model()

'''

'''
# ============== GTSRB ==============
# Define Benign Training and Testing Dataset
transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

batt = core.BATT(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50,43),
    # model=core.models.vgg19_bn(num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    schedule=None,
    seed=666,
)

poisoned_train_dataset, poisoned_test_dataset = batt.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': './result/',
    'experiment_name': 'batt_gtsrb_resnet50'
}

batt.train(schedule)
infected_model = batt.get_model()
'''
# ========== ResNet-18_imagenet_BATT ==========
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToPILImage(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'All_datasets', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
transform_test = Compose([
    ToPILImage(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'All_datasets', 'test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
batt = core.BATTspec(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNetspec(50,6),
    # model=core.models.vgg19_bn(num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    schedule=None,
    seed=666,
)
poisoned_train_dataset, poisoned_test_dataset = batt.get_poisoned_dataset()
import os
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
import torch

# 依你的 Normalize 設定
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denormalize(img_t: torch.Tensor, mean=MEAN, std=STD):
    """
    img_t: shape [C, H, W], float32, 經過 Normalize 後的張量
    回傳：反標準化且 clamp 到 [0,1] 的張量
    """
    if img_t.ndim != 3:
        raise ValueError(f"Expect CHW, got shape {img_t.shape}")
    # 避免 in-place 破壞原 tensor
    x = img_t.clone()
    x = x * std + mean
    return x.clamp(0, 1)

def save_dataset(dataset, out_root: str, split_name: str, num_classes: int = 6):
    """
    dataset: 你的 (img, label) 資料集（例如 poisoned_train_dataset）
    out_root: 根輸出資料夾，例如 'poisoned_dump'
    split_name: 'train' 或 'test'
    num_classes: 類別數（你目前是 0~5）
    """
    out_root = Path(out_root) / split_name
    # 建資料夾
    for c in range(num_classes):
        (out_root / str(c)).mkdir(parents=True, exist_ok=True)

    # 每個類別的流水號
    counters = [0] * num_classes

    for idx in range(len(dataset)):
        img_t, y = dataset[idx]  # img_t: [C,H,W] float, normalized; y: int
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        if y < 0 or y >= num_classes:
            # 若你的資料集中有不在 0~5 的標籤，這裡可選擇跳過或建新資料夾
            continue

        # 反標準化 → [0,1]
        img_01 = denormalize(img_t)

        # 轉 PIL（RGB）
        pil_img = to_pil_image(img_01)

        # 檔名：000001.jpg 這種
        counters[y] += 1
        fname = f"{counters[y]:06d}.jpg"
        save_path = out_root / str(y) / fname

        # 存檔（品質可自行調整）
        pil_img.save(save_path, format="JPEG", quality=95)

    print(f"Saved {sum(counters)} images to {out_root}. Per-class counts: {counters}")

# 例：把兩個資料集各自存到 poisoned_dump/train 與 poisoned_dump/test
save_dataset(poisoned_train_dataset, out_root="/app/graduate/BackdoorBox/data", split_name="train", num_classes=6)
save_dataset(poisoned_test_dataset,  out_root="/app/graduate/BackdoorBox/data", split_name="test",  num_classes=6)

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 64,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [120,150,180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': './result/',
    'experiment_name': 'ResNet-50_SP4_BATT'
}
# batt.train(schedule)