'''
This is the test code of poisoned training under BadNets.
'''


import os.path as osp
import os
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, Normalize

import core


# ========== Set global settings ==========
global_seed = 666
deterministic = False
torch.manual_seed(global_seed)
# CUDA_SELECTED_DEVICES = '0,3'
CUDA_SELECTED_DEVICES = '0'
datasets_root_dir = './data'

# # ========== ResNet-50_CIFAR-10_BadNets ==========
# dataset = torchvision.datasets.DatasetFolder
# model_path = '/app/graduate/BackdoorBox/experiments/resnet18_CIFAR-10_BadNets_2025-02-19_12_08_37/ckpt_epoch_200.pth'
# badnet_model = core.models.ResNet(18, num_classes=10)
# badnet_model.load_state_dict(torch.load(model_path))
# transform_train = Compose([
#     ToPILImage(),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
# ])
# # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_10', 'train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('jpg',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# transform_test = Compose([
#     ToPILImage(),
#     ToTensor(),
#     Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
# ])
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('jpg',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)
# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-3:, -3:] = 255
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[-3:, -3:] = 1.0

# badnets = core.Blended(
#     train_dataset=trainset,
#     test_dataset=testset,
#     # model=core.models.vgg19_bn(num_classes=10),
#     model=badnet_model,
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.1,
#     pattern=pattern,
#     weight=weight,
#     seed=global_seed,
#     deterministic=deterministic
# )

# # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
# schedule = {
#     'device': 'GPU',
#     'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 8,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'resnet18_CIFAR-10_BadNets'
# }
# # badnets.train(schedule)

# badnets.test(schedule)


# # ========== ResNet-50_GTSRB_BadNets ==========
# transform_train = Compose([
#     ToPILImage(),
#     Resize((32, 32)),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)

# transform_test = Compose([
#     ToPILImage(),
#     Resize((32, 32)),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your test set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)


# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-3:, -3:] = 255
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[-3:, -3:] = 0.2

# badnets = core.Blended(
#     train_dataset=trainset,
#     test_dataset=testset,
#     # model=core.models.vgg19_bn(num_classes=43),
#     model=core.models.ResNet(50,43),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.1,
#     pattern=pattern,
#     weight=weight,
#     poisoned_transform_train_index=2,
#     poisoned_transform_test_index=2,
#     seed=global_seed,
#     deterministic=deterministic
# )

# # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_watermarked.py)
# schedule = {
#     'device': 'GPU',
#     'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 8,

#     'lr': 0.01,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150,180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'resnet50_GTSRB_Blended'
# }
# # badnets.train(schedule)
# # badnets.test(schedule)
# # ========== ResNet-18_imagenet_BadNets ==========
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([
#     ToPILImage(),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
# ])
# # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'imagenet', 'train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('jpeg',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# transform_test = Compose([
#     ToPILImage(),
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
# ])
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'imagenet', 'val'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('jpeg',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)
# pattern = torch.zeros((224, 224), dtype=torch.uint8)
# pattern[-16:, -16:] = 255
# weight = torch.zeros((224, 224), dtype=torch.float32)
# weight[-16:, -16:] = 0.2

# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     # model=core.models.vgg19_bn(num_classes=10),
#     model=core.models.ResNet224(18,num_classes=1000),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.1,
#     pattern=pattern,
#     weight=weight,
#     seed=global_seed,
#     deterministic=deterministic
# )

# # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
# schedule = {
#     'device': 'GPU',
#     'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

#     'benign_training': False,
#     'batch_size': 350,
#     'num_workers': 6,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [30, 60,90],

#     'epochs': 100,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'resnet18_imagenet_Blended'
# }
# badnets.train(schedule)
# badnets.test(schedule)
'''
# ========== ResNet-50_CIFAR-10_BadNets ==========
dataset = torchvision.datasets.DatasetFolder

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
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)

# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

    'benign_training': False,
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

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-50_CIFAR-10_BadNets'
}
badnets.train(schedule)
badnets.test(schedule)


# ========== ResNet-50_GTSRB_BadNets ==========
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


pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50,43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    seed=global_seed,
    deterministic=deterministic
)

# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_watermarked.py)
schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150,180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-50_GTSRB_BadNets'
}
badnets.train(schedule)
badnets.test(schedule)
'''

# ========== ResNet-50_SP3_BadNets ==========
dataset = torchvision.datasets.DatasetFolder
transform_train = Compose([
    ToPILImage(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])
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
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'All_datasets', 'test'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


pattern = torch.zeros((224, 224), dtype=torch.uint8)
# pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-16:, -16:] = 255
weight = torch.zeros((224, 224), dtype=torch.float32)
# weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-16:, -16:] = 1.0

badnets = core.BadNetsspec(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNetspec(50,6),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
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
save_dataset(poisoned_train_dataset, out_root="/app/graduate/BackdoorBox/data/SP4_badnets", split_name="train", num_classes=6)
save_dataset(poisoned_test_dataset,  out_root="/app/graduate/BackdoorBox/data/SP4_badnets", split_name="test",  num_classes=6)

# # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_watermarked.py)
# schedule = {
#     'device': 'GPU',
#     'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

#     'benign_training': False,
#     'batch_size': 64,
#     'num_workers': 8,

#     'lr': 0.001,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [120,150,180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'ResNet-50_SP4_BadNets'
# }
# badnets.train(schedule)
# badnets.test(schedule)

# ========== ResNet-18_CIFAR-10_BadNets ==========
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToTensor(),
    ToPILImage(),
    Resize((32, 32)),
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
    ToTensor(),
    ToPILImage(),
    Resize((32, 32)),
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

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0
model = core.models.ResNet(18)
print(model)
# model.load_state_dict(torch.load('/app/graduate/BackdoorBox/experiments/resnet18_CIFAR-100_Blended_2025-05-02_01_52_15/ckpt_epoch_200.pth'))
badnets = core.BadNetsadv(
    train_dataset=trainset,
    test_dataset=testset,
    # model=core.models.vgg19_bn(num_classes=10),
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)

# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [120,150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'resnet18_cifar10_badnets_lam100_all'
}
badnets.train(schedule)
badnets.test(schedule)