'''
This is the test code of SCALE-UP defense.
SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency [ICLR, 2023] (https://arxiv.org/abs/2302.03251)
'''


from copy import deepcopy
import os.path as osp

import numpy as np
import random
import cv2
import os

import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize, Normalize
from PIL import Image
import core


# ========== Set global settings ==========
global_seed = 666
# deterministic = True
deterministic = False
torch.manual_seed(global_seed)
# ========== Set global settings ==========
datasets_root_dir = './data'  # please replace this with the path to your datasets root directory
CUDA_VISIBLE_DEVICES = '0'
portion = 0.1
batch_size = 128
num_workers = 4


def test(model_name, dataset_name, attack_name, defense_name, benign_dataset, attacked_dataset, defense, y_target):
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments/SCALE-UP-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }
    defense.test_acc(benign_dataset, schedule)
    if not attack_name == 'Benign':
        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
            'GPU_num': 1,

            'batch_size': batch_size,
            'num_workers': num_workers,

            # 1. ASR: the attack success rate calculated on all poisoned samples
            # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
            # 3. BA: the accuracy on all benign samples
            # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
            # In other words, ASR or BA does not influence the computation of the metric.
            # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
            'metric': 'ASR_NoTarget',
            'y_target': y_target,

            'save_dir': 'experiments/SCALE-UP-defense',
            'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR'
        }
        defense.test_acc(attacked_dataset, schedule)


# ========== ResNet-18_CIFAR-10_Attack_SCALE-UP ==========
# ===================== Benign ======================
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'GTSRB', 'Benign', 'SCALE-UP'

# benign_model = core.models.ResNet(18, num_classes=10)
# model_path = '/home/ay2/data/experiments/List_ResNet-18_CIFAR-10_Benign/_2023-08-01_19:05:02/ckpt_999.pth'
# benign_model.load_state_dict(torch.load(model_path), strict=False)

# dataset = torchvision.datasets.CIFAR10
# transform_train = Compose([
#     RandomHorizontalFlip(),
#     ToTensor()
# ])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# # Construct Shift Set for Defensive Purpose
# num_img = len(testset)
# indices = list(range(0, num_img))
# random.shuffle(indices)
# val_budget = 2000
# val_indices = indices[:val_budget]
# test_indices = indices[val_budget:]
# # Construct Shift Set for Defensive Purpose
# sub_testset = Subset(testset, test_indices)
# val_set = Subset(testset, val_indices)

# defense = core.SCALE_UP(model=benign_model)
# test(model_name, dataset_name, attack_name, defense_name, testset, None, defense, None)

# # ========== ResNet-18_CIFAR-10_Attack_Defense SCALE-UP ==========
poisoning_rate = 0.1
# tar# target_label = 1
# = BadNets ======================
print('Benigns\n')
model_name, dataset_name, attack_name, defense_name = 'ResNet-50', 'GTSRB', 'Bening', 'SCALE-UP'
badnet_model = core.models.ResNet(50, num_classes=43)
model_path = '/app/graduate/BackdoorBox/experiments/resnet50_GTSRB_Benign_2025-03-24_10_54_04/ckpt_epoch_200.pth'
def load_dict(model_path):
    state_dict = torch.load(model_path)
    # print(state_dict)
    if 'model' in list(state_dict.keys()):
        return state_dict['model']
    else:
        return state_dict
badnet_model.load_state_dict(load_dict(model_path))
transform_train = Compose([ 
    ToPILImage(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
        ToPILImage(),
    transforms.Resize((32, 32)),
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
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]

val_set = Subset(testset, val_indices)
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

# Get BadNets poisoned dataset
attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50, num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=poisoning_rate,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
# poisoned_testset = Subset(poisoned_testset, test_indices)
defense = core.SCALE_UP(model=badnet_model, valset=val_set)
# print(f'the BA and ASR of the original BadNets model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1)
defense.test(testset, poisoned_testset,'ResNet-50_GTSRB_Benign_SCALE-UP')
# ===================== BadNets ======================
print('BadNets\n')
model_name, dataset_name, attack_name, defense_name = 'ResNet-50', 'GTSRB', 'BadNets', 'SCALE-UP'
badnet_model = core.models.ResNet(50, num_classes=43)
model_path = '/app/graduate/BackdoorBox/experiments/resnet50_GTSRB_BadNets_2025-03-22_07_26_08/ckpt_epoch_200.pth'
def load_dict(model_path):
    state_dict = torch.load(model_path)
    # print(state_dict)
    if 'model' in list(state_dict.keys()):
        return state_dict['model']
    else:
        return state_dict
badnet_model.load_state_dict(load_dict(model_path))
transform_train = Compose([ 
    ToPILImage(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
        ToPILImage(),
    transforms.Resize((32, 32)),
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
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]

val_set = Subset(testset, val_indices)
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

# Get BadNets poisoned dataset
attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50, num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=poisoning_rate,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
# poisoned_testset = Subset(poisoned_testset, test_indices)
defense = core.SCALE_UP(model=badnet_model, valset=val_set)
# print(f'the BA and ASR of the original BadNets model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1)
defense.test(testset, poisoned_testset,'ResNet-50_GTSRB_BadNet_SCALE-UP')
# preds_benign = defense.detect(testset)
# preds_poison = defense.detect(poisoned_testset)
# print(f'Is poisoned for real benign batch: {preds_benign}')
# print(f'Is poisoned for real poisoned batch: {preds_poison}')


# ===================== WaNet ======================
print('WaNet\n')
attack_name = 'WaNet'
wanet_model = core.models.ResNet(50, num_classes=43)
model_path = '/app/graduate/BackdoorBox/experiments/resnet50_GTSRB_WaNet_2025-03-22_12_02_12/ckpt_epoch_200.pth'
wanet_model.load_state_dict(torch.load(model_path))
identity_grid = torch.load('/app/graduate/BackdoorBox/resnet50_GTSRB_WaNet_identity_grid.pth')
noise_grid = torch.load('/app/graduate/BackdoorBox/resnet50_GTSRB_WaNet_noise_grid.pth')
# target_label = state_dict['target']
transform_train = Compose([ 
    ToTensor(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
        ToPILImage(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
    ToTensor(),
        ToPILImage(),
    transforms.Resize((32, 32)),
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

testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]

val_set = Subset(testset, val_indices)
attack = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50, num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=poisoning_rate,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
defense = core.SCALE_UP(model=wanet_model, valset=val_set)
# print(f'the BA and ASR of the original WaNet model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 0)
defense.test(testset, poisoned_testset, 'ResNet-50_GTSRB_WaNet_SCALE-UP')

preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
print(f'Is poisoned for real benign batch: {preds_benign}')
print(f'Is poisoned for real poisoned batch: {preds_poison}')

# ===================== Blended ======================
print('Blended\n')
attack_name = 'Blended'
blended_model = core.models.ResNet(50, num_classes=43)
model_path = '/app/graduate/BackdoorBox/experiments/resnet50_GTSRB_Blended_2025-03-22_09_52_47/ckpt_epoch_200.pth'
blended_model.load_state_dict(torch.load(model_path))
transform_train = Compose([ 
        ToPILImage(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
        ToPILImage(),
    transforms.Resize((32, 32)),
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
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 0.2
val_set = Subset(testset, val_indices)
attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50, num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=poisoning_rate,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
defense = core.SCALE_UP(model=blended_model, valset=val_set)
# print(f'the BA and ASR of the original Blended model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1)
defense.test(testset, poisoned_testset, 'ResNet-50_GTSRB_Blended_SCALE-UP')

preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
print(f'Is poisoned for real benign batch: {preds_benign}')
print(f'Is poisoned for real poisoned batch: {preds_poison}')
def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:        
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError
# ===================== Refool ======================
print('Refool\n')
attack_name = 'Refool'
refool_model = core.models.ResNet(50, num_classes=43)
model_path = '/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_GTSRB_Refool_resnet50_2025-03-26_12_02_39/ckpt_epoch_100.pth'
refool_model.load_state_dict(torch.load(model_path))
reflection_images = []
reflection_data_dir = "./data/VOCdevkit/VOC2012/JPEGImages/"
dataset = torchvision.datasets.DatasetFolder
reflection_image_path = os.listdir(reflection_data_dir)
reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
transform_train = Compose([ 
        ToPILImage(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
        ToPILImage(),
    transforms.Resize((32, 32)),
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
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]

val_set = Subset(testset, val_indices)
refool= core.Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model= refool_model,
    # model = core.models.vgg19_bn(num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
    reflection_candidates = reflection_images,
)
poisoned_trainset, poisoned_testset = refool.get_poisoned_dataset()
defense = core.SCALE_UP(model=refool_model, valset=val_set)
# print(f'the BA and ASR of the original Refool model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1)
defense.test(testset, poisoned_testset, 'ResNet-50_GTSRB_Refool_SCALE-UP')

preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
print(f'Is poisoned for real benign batch: {preds_benign}')
print(f'Is poisoned for real poisoned batch: {preds_poison}')
# ===================== BATT ======================
print('BATT\n')
attack_name = 'BATT'
batt_model = core.models.ResNet(50, num_classes=43)
model_path = '/app/graduate/BackdoorBox/result/batt_gtsrb_2025-01-31_19_18_58/ckpt_epoch_30.pth'
batt_model.load_state_dict(torch.load(model_path))
transform_train = Compose([ 
        ToPILImage(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
        ToPILImage(),
    transforms.Resize((32, 32)),
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
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]

val_set = Subset(testset, val_indices)
batt = core.BATT(
    train_dataset=trainset,
    test_dataset=testset,
    model= batt_model,
    # model=core.models.vgg19_bn(num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_trainset, poisoned_testset = batt.get_poisoned_dataset()

defense = core.SCALE_UP(model=batt_model, valset=val_set)
# print(f'the BA and ASR of the original BATT model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1)
defense.test(testset, poisoned_testset, 'ResNet-50_GTSRB_BATT_SCALE-UP')
preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
print(f'Is poisoned for real benign batch: {preds_benign}')
print(f'Is poisoned for real poisoned batch: {preds_poison}')
# ===================== ISSBA ======================
print('ISSBA\n')
attack_name = 'ISSBA'
issba_model = core.models.ResNet(18, num_classes=43)
model_path = '/app/graduate/BackdoorBox/experiments/train_poison_GTSRB_ISSBA_resnet18_Nonoramlized_2025-03-23_12:16:56/ckpt_epoch_200.pth'
issba_model.load_state_dict(torch.load(model_path))
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.RandomCrop((32, 32), padding=5),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
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
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]
# test_indices = indices[val_budget:]
# # Construct Shift Set for Defensive Purpose
# sub_testset = Subset(testset, test_indices)
val_set = Subset(testset, val_indices)
poisoned_trainset = torch.load('/app/graduate/BackdoorBox/ISSBA_resnet18_gtsrb_poisoned_trainset.pth')
poisoned_testset = torch.load('/app/graduate/BackdoorBox/ISSBA_resnet18_gtsrb_poisoned_testset.pth')
defense = core.SCALE_UP(model=issba_model, valset=val_set)
# print(f'the BA and ASR of the original ISSBA model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense,1)
defense.test(testset, poisoned_testset, 'ResNet-18_GTSRB_ISSBA_SCALE-UP')

preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
print(f'Is poisoned for real benign batch: {preds_benign}')
print(f'Is poisoned for real poisoned batch: {preds_poison}')
# ===================== IAD ======================
print('IAD\n')
attack_name = 'IAD'
iad_model = core.models.ResNet(18, num_classes=43)
model_path = '/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_GTSRB_IAD_resnet18_2025-03-24_05:41:36/ckpt_epoch_270.pth'
iad_model.load_state_dict(torch.load(model_path)['model'])

transform_train = Compose([ 
        ToPILImage(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((32, 32)),
    ToTensor(),
])
transform_test = Compose([
        ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]
# test_indices = indices[val_budget:]
# # Construct Shift Set for Defensive Purpose
# sub_testset = Subset(testset, test_indices)
val_set = Subset(testset, val_indices)
poisoned_trainset = torch.load('/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_GTSRB_IAD_resnet18_2025-03-24_05:41:36/train270.pth')
poisoned_testset = torch.load('/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_GTSRB_IAD_resnet18_2025-03-24_05:41:36/test270.pth')
defense = core.SCALE_UP(model=iad_model, valset=val_set)
# print(f'the BA and ASR of the original IAD model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1)
defense.test(testset, poisoned_testset, 'ResNet-18_GTSRB_IAD_SCALE-UP')

preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
print(f'Is poisoned for real benign batch: {preds_benign}')
print(f'Is poisoned for real poisoned batch: {preds_poison}')
