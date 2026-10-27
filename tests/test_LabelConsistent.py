'''
This is the test code of poisoned training under LabelConsistent.
'''


import os
import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize, Normalize
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder

import core


CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
datasets_root_dir = './data'
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

'''
# ============== Label Consistent attack BaselineMNISTNetwork on MNIST ==============
dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

adv_model = core.models.BaselineMNISTNetwork()
adv_ckpt = torch.load('/data/yamengxi/Backdoor/experiments/BaselineMNISTNetwork_MNIST_Benign_2022-03-29_16:18:02/ckpt_epoch_200.pth')
adv_model.load_state_dict(adv_ckpt)


pattern = torch.zeros((28, 28), dtype=torch.uint8)

k = 6

pattern[:k,:k] = 255
pattern[:k,-k:] = 255
pattern[-k:,:k] = 255
pattern[-k:,-k:] = 255

# pattern[-1, -1] = 255
# pattern[-1, -3] = 255
# pattern[-3, -1] = 255
# pattern[-2, -2] = 255

# pattern[0, -1] = 255
# pattern[1, -2] = 255
# pattern[2, -3] = 255
# pattern[2, -1] = 255

# pattern[0, 0] = 255
# pattern[1, 1] = 255
# pattern[2, 2] = 255
# pattern[2, 0] = 255

# pattern[-1, 0] = 255
# pattern[-1, 2] = 255
# pattern[-2, 1] = 255
# pattern[-3, 0] = 255

weight = torch.zeros((28, 28), dtype=torch.float32)
weight[:k,:k] = 1.0
weight[:k,-k:] = 1.0
weight[-k:,:k] = 1.0
weight[-k:,-k:] = 1.0


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 1024,
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
    'experiment_name': 'BaselineMNISTNetwork_MNIST_LabelConsistent'
}


eps = 64
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.5

label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/MNIST_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target=2,
    poisoned_rate=poisoned_rate,
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

label_consistent.train()
'''
'''
# ============== Label Consistent attack ResNet-50 on CIFAR-10 ==============
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
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

# adv_model = core.models.ResNet(50)
adv_model = core.models.vgg19_bn(num_classes=10)
# adv_ckpt = torch.load('experiments/ResNet-50_CIFAR-10_Benign_2024-11-28_21_00_00/ckpt_epoch_200.pth')
adv_ckpt = torch.load('/home/Bachi/graduate/BackdoorBox/experiments/vgg19_CIFAR-10_Benign_2024-12-10_10_03_19/ckpt_epoch_200.pth')
adv_model.load_state_dict(adv_ckpt)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-1, -1] = 255
pattern[-1, -3] = 255
pattern[-3, -1] = 255
pattern[-2, -2] = 255

pattern[0, -1] = 255
pattern[1, -2] = 255
pattern[2, -3] = 255
pattern[2, -1] = 255

pattern[0, 0] = 255
pattern[1, 1] = 255
pattern[2, 2] = 255
pattern[2, 0] = 255

pattern[-1, 0] = 255
pattern[-1, 2] = 255
pattern[-2, 1] = 255
pattern[-3, 0] = 255

weight = torch.zeros((32, 32), dtype=torch.float32)
weight[:3,:3] = 1.0
weight[:3,-3:] = 1.0
weight[-3:,:3] = 1.0
weight[-3:,-3:] = 1.0


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
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
    'experiment_name': 'vgg19_CIFAR-10_LabelConsistent'
}


eps = 8
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.25

label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    # model=core.models.ResNet(50),
    model=core.models.vgg19_bn(num_classes=10),
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target=2,
    poisoned_rate=poisoned_rate,
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

label_consistent.train()
'''
# # ============== Label Consistent attack ResNet-50 on CIFAR-10 ==============
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([
#     ToPILImage(),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(mean = [0.5071, 0.4867, 0.4408],std  = [0.2675, 0.2565, 0.2761])
# ])
# # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_100', 'train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# transform_test = Compose([
#     ToPILImage(),
#     ToTensor(),
#     Normalize(mean = [0.5071, 0.4867, 0.4408],std  = [0.2675, 0.2565, 0.2761])
# ])
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_100', 'test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)

# # adv_model = core.models.ResNet(50)
# adv_model = core.models.vgg19_bn(num_classes=10)
# # adv_ckpt = torch.load('experiments/ResNet-50_CIFAR-10_Benign_2024-11-28_21_00_00/ckpt_epoch_200.pth')
# adv_ckpt = torch.load('/home/Bachi/graduate/BackdoorBox/experiments/vgg19_CIFAR-10_Benign_2024-12-10_10_03_19/ckpt_epoch_200.pth')
# adv_model.load_state_dict(adv_ckpt)

# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-1, -1] = 255
# pattern[-1, -3] = 255
# pattern[-3, -1] = 255
# pattern[-2, -2] = 255

# pattern[0, -1] = 255
# pattern[1, -2] = 255
# pattern[2, -3] = 255
# pattern[2, -1] = 255

# pattern[0, 0] = 255
# pattern[1, 1] = 255
# pattern[2, 2] = 255
# pattern[2, 0] = 255

# pattern[-1, 0] = 255
# pattern[-1, 2] = 255
# pattern[-2, 1] = 255
# pattern[-3, 0] = 255

# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[:3,:3] = 1.0
# weight[:3,-3:] = 1.0
# weight[-3:,:3] = 1.0
# weight[-3:,-3:] = 1.0


# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
#     'GPU_num': 1,

#     'benign_training': False, # Train Attacked Model
#     'batch_size': 512,
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
#     'experiment_name': 'resnet18_CIFAR-100_LabelConsistent'
# }


# eps = 8
# alpha = 1.5
# steps = 100
# max_pixel = 255
# poisoned_rate = 0.1

# label_consistent = core.LabelConsistent(
#     train_dataset=trainset,
#     test_dataset=testset,
#     # model=core.models.ResNet(50),
#     model=core.models.vgg19_bn(num_classes=10),
#     adv_model=adv_model,
#     adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
#     loss=nn.CrossEntropyLoss(),
#     y_target=2,
#     poisoned_rate=poisoned_rate,
#     pattern=pattern,
#     weight=weight,
#     eps=eps,
#     alpha=alpha,
#     steps=steps,
#     max_pixel=max_pixel,
#     poisoned_transform_train_index=0,
#     poisoned_transform_test_index=0,
#     poisoned_target_transform_index=0,
#     schedule=schedule,
#     seed=global_seed,
#     deterministic=True
# )

# label_consistent.train()
'''
# ============== Label Consistent attack ResNet-50 on GTSRB ==============
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB', 'Train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB', 'Test'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


adv_model = core.models.ResNet(50, 43)
adv_ckpt = torch.load('experiments/ResNet-50_GTSRB_Benign_2024-11-28_23_49_23/ckpt_epoch_200.pth')
adv_model.load_state_dict(adv_ckpt)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-1, -1] = 255
pattern[-1, -3] = 255
pattern[-3, -1] = 255
pattern[-2, -2] = 255

pattern[0, -1] = 255
pattern[1, -2] = 255
pattern[2, -3] = 255
pattern[2, -1] = 255

pattern[0, 0] = 255
pattern[1, 1] = 255
pattern[2, 2] = 255
pattern[2, 0] = 255

pattern[-1, 0] = 255
pattern[-1, 2] = 255
pattern[-2, 1] = 255
pattern[-3, 0] = 255

weight = torch.zeros((32, 32), dtype=torch.float32)
weight[:3,:3] = 1.0
weight[:3,-3:] = 1.0
weight[-3:,:3] = 1.0
weight[-3:,-3:] = 1.0


# pattern = torch.zeros((32, 32), dtype=torch.uint8)

# k = 5

# pattern[:k,:k] = 255
# pattern[:k,-k:] = 255
# pattern[-k:,:k] = 255
# pattern[-k:,-k:] = 255

# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[:k,:k] = 1.0
# weight[:k,-k:] = 1.0
# weight[-k:,:k] = 1.0
# weight[-k:,-k:] = 1.0

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
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
    'experiment_name': 'ResNet-50_GTSRB_LabelConsistent'
}


eps = 16
alpha = 3.0
steps = 100
max_pixel = 255
poisoned_rate = 0.5

label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50, 43),
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/GTSRB_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target=2,
    poisoned_rate=poisoned_rate,
    adv_transform=Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), ToTensor(), Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])]),
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

# label_consistent.train()
'''
# ============== Label Consistent attack ResNet-18 on iamgenet ==============
transform_train = Compose([
    transforms.ToPILImage(),
    Resize((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'All_datasets', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    transforms.ToPILImage(),
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
])
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'All_datasets', 'test'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


adv_model = core.models.ResNetspec(50, 6)
adv_ckpt = torch.load('/app/graduate/BackdoorBox/experiments/ResNet-50_SP4_Benign_2025-08-05_12_18_10/ckpt_epoch_200.pth')
adv_model.load_state_dict(adv_ckpt)
#resize pattern to 224x224
pattern = torch.zeros((224,224), dtype=torch.uint8)
pattern[:5,:5] = 255
pattern[:5,-5:] = 255
pattern[-5:,:5] = 255
pattern[-5:,-5:] = 255

weight = torch.zeros((224,224), dtype=torch.float32)
weight[:5,:5] = 1.0
weight[:5,-5:] = 1.0
weight[-5:,:5] = 1.0
weight[-5:,-5:] = 1.0


# pattern = torch.zeros((32, 32), dtype=torch.uint8)

# k = 5

# pattern[:k,:k] = 255
# pattern[:k,-k:] = 255
# pattern[-k:,:k] = 255
# pattern[-k:,-k:] = 255

# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[:k,:k] = 1.0
# weight[:k,-k:] = 1.0
# weight[-k:,:k] = 1.0
# weight[-k:,-k:] = 1.0

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 64,
    'num_workers': 8,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [120,150,180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-50_SP4_LabelConsistent'
}


eps = 8
alpha = 3.0
steps = 100
max_pixel = 255
poisoned_rate = 0.1

label_consistent = core.LabelConsistentspec(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNetspec(50, 6),
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/imagenet_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=poisoned_rate,
    adv_transform=Compose([transforms.ToPILImage(), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])]),
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

label_consistent.train()

# # ============== Label Consistent attack ResNet-18 on CIFAR-100 ==============
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([
#     ToPILImage(),
#     Resize((32, 32)),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(mean = [0.5071, 0.4867, 0.4408],std  = [0.2675, 0.2565, 0.2761])
# ])
# # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_100', 'train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# transform_test = Compose([
#     ToPILImage(),
#     Resize((32, 32)),
#     ToTensor(),
#     Normalize(mean = [0.5071, 0.4867, 0.4408],std  = [0.2675, 0.2565, 0.2761])
# ])
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_100', 'test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)

# adv_model = core.models.ResNet(18,100)
# # adv_model = core.models.vgg19_bn(num_classes=10)
# # adv_ckpt = torch.load('experiments/ResNet-50_CIFAR-10_Benign_2024-11-28_21_00_00/ckpt_epoch_200.pth')
# adv_ckpt = torch.load('/app/graduate/BackdoorBox/experiments/resnet18_cifar100_Benign_2025-05-02_06_18_52/ckpt_epoch_200.pth')
# adv_model.load_state_dict(adv_ckpt)

# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-1, -1] = 255
# pattern[-1, -3] = 255
# pattern[-3, -1] = 255
# pattern[-2, -2] = 255

# pattern[0, -1] = 255
# pattern[1, -2] = 255
# pattern[2, -3] = 255
# pattern[2, -1] = 255

# pattern[0, 0] = 255
# pattern[1, 1] = 255
# pattern[2, 2] = 255
# pattern[2, 0] = 255

# pattern[-1, 0] = 255
# pattern[-1, 2] = 255
# pattern[-2, 1] = 255
# pattern[-3, 0] = 255

# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[:3,:3] = 1.0
# weight[:3,-3:] = 1.0
# weight[-3:,:3] = 1.0
# weight[-3:,-3:] = 1.0


# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
#     'GPU_num': 1,

#     'benign_training': False, # Train Attacked Model
#     'batch_size': 512,
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
#     'experiment_name': 'resnet18_CIFAR-100_LabelConsistent'
# }


# eps = 30
# alpha = 3
# steps = 100
# max_pixel = 255
# poisoned_rate = 0.3

# label_consistent = core.LabelConsistent(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.ResNet(18,100),
#     # model=core.models.vgg19_bn(num_classes=10),
#     adv_model=adv_model,
#     adv_dataset_dir=f'./adv_dataset/CIFAR-100_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
#     loss=nn.CrossEntropyLoss(),
#     y_target=2,
#     poisoned_rate=poisoned_rate,
#     pattern=pattern,
#     weight=weight,
#     eps=eps,
#     alpha=alpha,
#     steps=steps,
#     max_pixel=max_pixel,
#     poisoned_transform_train_index=0,
#     poisoned_transform_test_index=0,
#     poisoned_target_transform_index=0,
#     schedule=schedule,
#     seed=global_seed,
#     deterministic=True
# )

# label_consistent.train()