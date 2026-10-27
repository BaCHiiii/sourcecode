import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize, Normalize

from torch.utils.data import random_split
import os
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import core
import numpy as np

from copy import deepcopy
from torchvision.datasets import DatasetFolder
import cv2
from core.utils import test
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
normal_count50 = [12,16,20]
normal_count18 =[8,24]
# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '0'
batch_size = 128
num_workers = 8
type1 = 'median'
class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)
class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label
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
def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

# for draw distribution
def test_caculate1(model,p,trainset,testset,poisoned_testest,layers,layers_name,normal_count,y_target,experiment_name):
    model = model.to('cuda')
    cad = core.Caculate(
        train_dataset=trainset,
        test_dataset=testset,
        poison_test_dataset=poisoned_testest,
        model=model,
        layers=layers,
        layers_name=layers_name,
        normal_count=normal_count,
        experiment_name=experiment_name
    )
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,
        'type': type,
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    cad.cal1(schedule)

    del cad
from time import perf_counter
def test_caculate(model,p,trainset,testset,poisoned_testest,layers,layers_name,normal_count,y_target,experiment_name):
    num1 = int(len(trainset) * p)
    num2 = int(len(trainset) - num1)
    trainset1, trainset2 = random_split(trainset, [num1, num2])
    model = model.to('cuda')
    times = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    cad = core.Caculate(
        train_dataset=trainset1,
        test_dataset=testset,
        poison_test_dataset=poisoned_testest,
        model=model,
        layers=layers,
        layers_name=layers_name,
        normal_count=normal_count,
        experiment_name=experiment_name
    )
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,
        'type': type1,
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    cad.cal(schedule)
    work_dir = osp.join('/app/graduate/BackdoorBox/auroc', experiment_name+"_"+times)
    if not osp.exists(work_dir):
        os.makedirs(work_dir)
    log = Log(osp.join(work_dir, 'auroc.txt'))
    log(f'mean\n')
    log(f'PDF : 1e-5\n')
    log(f'trainset size : {len(trainset)}\n')
    inference_time = []
    start = perf_counter()
    detect_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'type': type1,
        'metric': 'PDF',
        'save_dir': 'experiments_CAD',
        'data': 'poison',
        'experiment_name': experiment_name+"_"+times,       
    }
    TP,FN = cad.detection2(detect_schedule)
    middle = perf_counter()
    detect_schedule2 = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'type': type1,
        'metric': 'PDF',
        'save_dir': 'experiments_CAD',
        'data': 'clean',
        'experiment_name': experiment_name+"_"+times, 
    }
    FP,TN = cad.detection2(detect_schedule2)
    end = perf_counter()
    inference_time.append((middle-start)/10000.0)
    inference_time.append((end-middle)/10000.0)
    log(f'poison inference time: {((middle-start)/10000.0)}\n')
    log(f'clean inference time: {((end-middle)/10000.0)}\n')
    log(f'TP: {TP}\n')
    log(f'FP: {FP}\n')
    log(f'TN: {TN}\n')
    log(f'FN: {FN}\n')
    log(f'inference_time : {np.mean(inference_time)}\n')

    del cad
def test_caculate3(model,p,trainset,testset,poisoned_testest,layers,layers_name,normal_count,y_target,experiment_name):
    num1 = int(len(trainset) * p)
    num2 = int(len(trainset) - num1)
    trainset1, trainset2 = random_split(trainset, [num1, num2])
    model = model.to('cuda')
    times = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    cad = core.Caculate(
        train_dataset=trainset1,
        test_dataset=testset,
        poison_test_dataset=poisoned_testest,
        model=model,
        layers=layers,
        layers_name=layers_name,
        normal_count=normal_count,
        experiment_name=experiment_name
    )
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,
        'type': type1,
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    cad.cal(schedule)
    from itertools import product
    from sklearn.metrics import roc_curve, auc
    ranges = [[x+i for i in range(-5,4,2)] for x in normal_count]
    combinations = list(product(*ranges))
    for i in combinations[:5]:
        print(i)
    print(len(combinations))
    fpr = [0]
    tpr = [0]
    work_dir = osp.join('/app/graduate/BackdoorBox/auroc', experiment_name+"_"+times)
    if not osp.exists(work_dir):
        os.makedirs(work_dir)
    log = Log(osp.join(work_dir, 'auroc.txt'))
    log(f'median\n')
    log(f'Z-score: 5\n')
    for i in combinations:
        cad.normal_count = i
        detect_schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
            'GPU_num': 1,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'type': type1,
            'metric': 'Z-score',
            'save_dir': 'experiments_CAD',
            'data': 'poison',
            'experiment_name': experiment_name+"_"+times,       
        }
        tpr1 = cad.detection1(detect_schedule)
        detect_schedule2 = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
            'GPU_num': 1,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'type': type1,
            'metric': 'Z-score',
            'save_dir': 'experiments_CAD',
            'data': 'clean',
            'experiment_name': experiment_name+"_"+times, 
        }
        fpr1 = cad.detection1(detect_schedule2)
        fpr.append(fpr1)
        tpr.append(tpr1)
        log(f'count: {i}\n')
        log(f'FPR: {fpr1}\n')
        log(f'TPR: {tpr1}\n')
    fpr.append(1)
    tpr.append(1)
    # sort fpr tpr by fpr
    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    log(f'FPR: {fpr}\n')
    log(f'TPR: {tpr}\n')
    roc_auc = auc(fpr, tpr)

    log(f'ROC AUC: {roc_auc:.2f}\n')
    print(f'ROC AUC: {roc_auc:.2f}')
    del cad
datasets_root_dir = './data'

# ============= ResNet-50_GTSRB_Benign_CAD =============
print('ResNet-50_GTSRB_Benign_CAD')
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),   
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])

# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
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

# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

model = core.models.ResNet(50,43)
model_path = '/app/graduate/BackdoorBox/experiments/resnet50_GTSRB_Benign_2025-03-24_10_54_04/ckpt_epoch_200.pth'
badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = badnets.get_poisoned_dataset()

model.load_state_dict(torch.load(model_path))

#choose the proportion of data to calculation
p = 0.1
layer = model.layer4[0].conv1
# print(layer)
layers = [model.layer4[0],model.layer4[1],model.layer4[2]]
layers_name = ['layer4.0','layer4.1','layer4.2']
normal_count = normal_count50
y_target = 1
# print(len(testset))
# print(len(poisoned_testset))
# for name, module in model.named_modules():
#     print(name)
    # print(module)
test_caculate(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-50_GTSRB_Benign_CAD')

# ============= ResNet-50_GTSRB_BadNet_CAD =============
# print('ResNet-50_GTSRB_BadNet_CAD')
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([
#     ToTensor(),
#     ToPILImage(),
# Resize((32, 32)),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# transform_test = Compose([
#     ToTensor(),
#     ToPILImage(),
# Resize((32, 32)),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)

# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-3:, -3:] = 255
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[-3:, -3:] = 1.0

# model = core.models.ResNet(50,43)
# model_path = '/app/graduate/BackdoorBox/experiments/resnet50_GTSRB_BadNets_2025-03-22_07_26_08/ckpt_epoch_200.pth'
# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.1,
#     pattern=pattern,
#     weight=weight,
#     seed=global_seed,
#     deterministic=deterministic
# )
# poisoned_trainset, poisoned_testset = badnets.get_poisoned_dataset()


# model.load_state_dict(torch.load(model_path))

# #choose the proportion of data to calculation
# p = 0.1
# layer = model.layer4[0].conv1
# # print(layer)
# layers = [model.layer4[0],model.layer4[1],model.layer4[2]]
# layers_name = ['layer4.0','layer4.1','layer4.2']
# normal_count = normal_count50
# y_target = 1
# # print(len(testset))
# # print(len(poisoned_testset))
# # for name, module in model.named_modules():
# #     print(name)
#     # print(module)
# test_caculate3(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-50_GTSRB_BadNet_CAD')



# # ============= ResNet_GTSRB_WaNet_CAD =============
# print('ResNet-50_GTSRB_WaNet_CAD')
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([
#     ToTensor(),
#     ToPILImage(),
# Resize((32, 32)),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# transform_test = Compose([
#     ToTensor(),
#     ToPILImage(),
# Resize((32, 32)),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)
# # identity_grid,noise_grid=gen_grid(32,4)
# identity_grid = torch.load('/app/graduate/BackdoorBox/resnet50_GTSRB_WaNet_identity_grid.pth')
# noise_grid = torch.load('/app/graduate/BackdoorBox/resnet50_GTSRB_WaNet_noise_grid.pth')

# model=core.models.ResNet(50,43)
# model_path = '/app/graduate/BackdoorBox/experiments/resnet50_GTSRB_WaNet_2025-03-22_12_02_12/ckpt_epoch_200.pth'
# wanet = core.WaNet(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.1,
#     identity_grid=identity_grid,
#     noise_grid=noise_grid,
#     noise=False,
#     seed=global_seed,
#     deterministic=deterministic
# )
# poisoned_trainset, poisoned_testset = wanet.get_poisoned_dataset()

# model.load_state_dict(torch.load(model_path))
# #choose the proportion of data to calculation
# p = 0.1
# layers = [model.layer4[0],model.layer4[1],model.layer4[2]]
# layers_name = ['layer4.0','layer4.1','layer4.2']
# normal_count = normal_count50
# y_target = 0
# test_caculate3(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-50_GTSRB_WaNet_CAD')

# # ========== ResNet-50_GTSRB_Blended_CAD ==========
# print('ResNet-50_GTSRB_Blended_CAD')
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([
#     ToTensor(),
#     ToPILImage(),
# Resize((32, 32)),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# transform_test = Compose([
#     ToTensor(),
#     ToPILImage(),
# Resize((32, 32)),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)

# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-3:, -3:] = 255
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[-3:, -3:] = 0.2

# model = core.models.ResNet(50,43)
# model_path = '/app/graduate/BackdoorBox/experiments/resnet50_GTSRB_Blended_2025-03-22_09_52_47/ckpt_epoch_200.pth'
# blended = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.1,
#     pattern=pattern,
#     weight=weight,
#     seed=global_seed,
#     deterministic=deterministic
# )
# poisoned_trainset, poisoned_testset = blended.get_poisoned_dataset()


# model.load_state_dict(torch.load(model_path))

# #choose the proportion of data to calculation
# p = 0.1
# layer = model.layer4[0].conv1
# # print(layer)
# layers = [model.layer4[0],model.layer4[1],model.layer4[2]]
# layers_name = ['layer4.0','layer4.1','layer4.2']
# normal_count = normal_count50
# y_target = 1
# # print(len(testset))
# # print(len(poisoned_testset))
# # for name, module in model.named_modules():
# #     print(name)
#     # print(module)
# test_caculate3(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-50_GTSRB_Blended_CAD')

# # ========== ResNet-50_GTSRB_Refool_CAD ==========
# print('ResNet-50_GTSRB_Refool_CAD')
# reflection_images = []
# reflection_data_dir = "./data/VOCdevkit/VOC2012/JPEGImages/"
# dataset = torchvision.datasets.DatasetFolder
# reflection_image_path = os.listdir(reflection_data_dir)
# reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
# # Prepare datasets
# transform_train = Compose([
#     ToTensor(), 
#         ToPILImage(),
# Resize((32, 32)),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# transform_test = Compose([
#     ToTensor(),
#         ToPILImage(),
# Resize((32, 32)),
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
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)

# model = core.models.ResNet(50,43)
# model_path = '/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_GTSRB_Refool_resnet50_2025-03-26_12_02_39/ckpt_epoch_100.pth'
# refool= core.Refool(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     # model = core.models.vgg19_bn(num_classes=43),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,
#     poisoned_transform_train_index=0,
#     poisoned_transform_test_index=0,
#     poisoned_target_transform_index=0,
#     schedule=None,
#     seed=global_seed,
#     deterministic=deterministic,
#     reflection_candidates = reflection_images,
# )
# poisoned_trainset, poisoned_testset = refool.get_poisoned_dataset()


# model.load_state_dict(torch.load(model_path))
# print(len(testset))
# print(len(poisoned_testset))
# # for name, module in model.named_modules():
# #     print(name)
# #     print(module)
# #choose the proportion of data to calculation
# p = 0.1
# layer = model.layer4[0].conv1
# # print(layer)
# layers = [model.layer4[0],model.layer4[1],model.layer4[2]]
# layers_name = ['layer4.0','layer4.1','layer4.2']
# normal_count = normal_count50
# y_target = 1
# # print(len(testset))
# # print(len(poisoned_testset))
# # for name, module in model.named_modules():
# #     print(name)
# #     print(module)
# test_caculate3(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-50_GTSRB_Refool_CAD')

# # ========== ResNet-50_GTSRB_BATT_CAD ==========
# print('ResNet-50_GTSRB_BATT_CAD')
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([ 
#         ToPILImage(),
# Resize((32, 32)),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# transform_test = Compose([
#         ToPILImage(),
# Resize((32, 32)),
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
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)
# model = core.models.ResNet(50,43)
# model_path = '/app/graduate/BackdoorBox/result/batt_gtsrb_2025-01-31_19_18_58/ckpt_epoch_30.pth'
# batt = core.BATT(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     # model=core.models.vgg19_bn(num_classes=10),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,
#     seed=global_seed,
#     deterministic=deterministic
# )

# poisoned_trainset, poisoned_testset = batt.get_poisoned_dataset()

# model.load_state_dict(torch.load(model_path))

# p = 0.1
# layer = model.layer4[0].conv1
# # print(layer)
# layers = [model.layer4[0],model.layer4[1],model.layer4[2]]
# layers_name = ['layer4.0','layer4.1','layer4.2']
# normal_count = normal_count50
# y_target = 1
# test_caculate3(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-50_GTSRB_BATT_CAD')

# # ========== ResNet-18_GTSRB_ISSBA_CAD ==========
# print('ResNet-18_GTSRB_ISSBA_CAD')
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([
#     transforms.ToPILImage(),
# Resize((32, 32)),
#     transforms.RandomCrop((32, 32), padding=5),
#     transforms.RandomRotation(10),
#     transforms.RandomHorizontalFlip(p=0.5),
#     ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                         (0.247, 0.243, 0.261))
# ])
# transform_test = Compose([
#     transforms.ToPILImage(),
# Resize((32, 32)),
#     ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                         (0.247, 0.243, 0.261))
# ])
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)

# model = core.models.ResNet(18,43)
# model_path = '/app/graduate/BackdoorBox/experiments/train_poison_GTSRB_ISSBA_resnet18_Nonoramlized_2025-03-23_12:16:56/ckpt_epoch_200.pth'

# poisoned_trainset = torch.load('/app/graduate/BackdoorBox/ISSBA_resnet18_gtsrb_poisoned_trainset.pth')
# poisoned_testset = torch.load('/app/graduate/BackdoorBox/ISSBA_resnet18_gtsrb_poisoned_testset.pth')
# print(len(testset))
# print(len(poisoned_testset))
# model.load_state_dict(torch.load(model_path))

# p = 0.1
# layer = model.layer4[0].conv1
# # print(layer)
# layers = [model.layer4[0],model.layer4[1]]
# layers_name = ['layer4.0','layer4.1']
# normal_count = normal_count18
# y_target = 1
# test_caculate3(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-18_GTSRB_ISSBA_CAD')

# # ========== ResNet-18_GTSRB_IAD_CAD ==========
# print('ResNet-18_GTSRB_IAD_CAD')
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([ 
#         ToPILImage(),
# Resize((32, 32)),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed

#     ToTensor(),
# ])
# transform_test = Compose([
#         ToPILImage(),
# Resize((32, 32)),
#     ToTensor(),
# ])
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)

# model = core.models.ResNet(18,43)
# model_path = '/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_GTSRB_IAD_resnet18_2025-03-24_05:41:36/ckpt_epoch_270.pth'
# poisoned_trainset = torch.load('/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_GTSRB_IAD_resnet18_2025-03-24_05:41:36/train270.pth')
# poisoned_testset = torch.load('/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_GTSRB_IAD_resnet18_2025-03-24_05:41:36/test270.pth')
# print(len(testset))
# print(len(poisoned_testset))
# model.load_state_dict(torch.load(model_path)['model'])

# p = 0.1
# layer = model.layer4[0].conv1
# # print(layer)
# layers = [model.layer4[0],model.layer4[1]]
# layers_name = ['layer4.0','layer4.1']
# normal_count = normal_count18
# y_target = 1
# test_caculate3(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-18_GTSRB_iAD_CAD')
# # ========== ResNet-50_GTSRB_Labelconsistent_CAD ==========
# print('ResNet-50_GTSRB_Labelconsistent_CAD')
# model = core.models.ResNet(50,43)

# attack_name = 'LabelConsistent'
# model_path = '/app/graduate/BackdoorBox/experiments/ResNet-50_GTSRB_LabelConsistent_2024-11-29_14_03_37/ckpt_epoch_200.pth'

# adv_model = deepcopy(model)
# adv_ckpt = torch.load('/app/graduate/BackdoorBox/experiments/ResNet-50_GTSRB_Benign_2024-11-28_21_00_00/ckpt_epoch_200.pth')
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
#     'experiment_name': 'ResNet-50_GTSRB_LabelConsistent'
# }

# eps = 8
# alpha = 1.5
# steps = 100
# max_pixel = 255
# poisoned_rate = 0.25

# labelconsistent = core.LabelConsistent(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     adv_model=adv_model,
#     adv_dataset_dir=f'./adv_dataset/GTSRB_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
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
# poisoned_trainset, poisoned_testset = labelconsistent.get_poisoned_dataset()


# model.load_state_dict(torch.load(model_path))

# #choose the proportion of data to calculation
# p=0.1
# #set the layer to prune
# layers = [model.layer4[0],model.layer4[1],model.layer4[2]]
# layers_name = ['layer4.0','layer4.1','layer4.2']
# normal_count = normal_count50
# y_target=2
# # test_caculate(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-50_GTSRB_LabelConsistent_CAD')

# # ========== ResNet-18_GTSRB_Sleeperagent_CAD ==========
# print('ResNet-18_GTSRB_Sleeperagent_CAD')
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([ 
#         ToPILImage(),
# Resize((32, 32)),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
#     ToTensor(),
# ])
# transform_test = Compose([
#         ToPILImage(),
# Resize((32, 32)),
#     ToTensor(),
# ])
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)

# model = core.models.ResNet(18,43)
# model_path = '/app/graduate/BackdoorBox/experiments/train_poison_GTSRB_SleeperAgent_resnet18_NoNormalization_2025-03-04_07:06:21/ckpt_epoch_100.pth'

# poisoned_trainset = torch.load('/app/graduate/BackdoorBox/Sleeperagent_resnet18_poisoned_trainset.pth')
# poisoned_testset = torch.load('/app/graduate/BackdoorBox/Sleeperagent_resnet18_poisoned_testset.pth')
# print(len(testset))
# print(len(poisoned_testset))
# model.load_state_dict(torch.load(model_path))

# p = 0.1
# layer = model.layer4[0].conv1
# # print(layer)
# layers = [model.layer4[0],model.layer4[1]]
# layers_name = ['layer4.0','layer4.1']
# normal_count = normal_count18
# y_target = 1
# test_caculate(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-18_GTSRB_Sleeperagent_CAD')


'''
# ===============================================================vgg============================================================
# ============= vgg_GTSRB_BadNet_CAD =============
print('vgg_GTSRB_BadNet_CAD')
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToTensor(),
    ToPILImage(),
Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
transform_test = Compose([
    ToTensor(),
    ToPILImage(),
Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

model = core.models.vgg19_bn(num_classes=10)
model_path = '/app/graduate/BackdoorBox/experiments/vgg19_GTSRB_BadNets_2024-12-10_17_10_57/ckpt_epoch_200.pth'
badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = badnets.get_poisoned_dataset()


model.load_state_dict(torch.load(model_path))

#choose the proportion of data to calculation
layer = model.features
# print(layer) 
p = 0.1
layers_name = ['layer38','layer42','layer45','layer48','layer51']
layers = [model.features[38],model.features[42],model.features[45],model.features[48],model.features[51]]
normal_count = [0.1]*5
y_target = 1
print(len(testset))
print(len(poisoned_testset))
# for name, module in model.named_modules():
#     print(name)
#     print(module)
# test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'vgg_GTSRB_BadNet_CAD')



# ============= vgg_GTSRB_WaNet_CAD =============
print('vgg_GTSRB_WaNet_CAD')
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToTensor(),
    ToPILImage(),
Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
transform_test = Compose([
    ToTensor(),
    ToPILImage(),
Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
identity_grid,noise_grid=gen_grid(32,4)
torch.load('vgg19_GTSRB_WaNet_identity_grid.pth')
torch.load('vgg19_GTSRB_WaNet_noise_grid.pth')

model=core.models.vgg19_bn(num_classes=10)
model_path = '/app/graduate/BackdoorBox/experiments/vgg19_GTSRB_WaNet_2025-02-25_16_31_58/ckpt_epoch_200.pth'
wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = wanet.get_poisoned_dataset()

model.load_state_dict(torch.load(model_path))
#choose the proportion of data to calculation
p = 0.1
layers_name = ['layer38','layer42','layer45','layer48','layer51']
layers = [model.features[38],model.features[42],model.features[45],model.features[48],model.features[51]]
normal_count = [0.1]*5
y_target = 1
print(len(testset))
print(len(poisoned_testset))
# test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'vgg_GTSRB_WaNet_CAD')

# ========== vgg_GTSRB_Labelconsistent_CAD ==========
print('vgg_GTSRB_Labelconsistent_CAD')
model = core.models.vgg19_bn(num_classes=10)

attack_name = 'LabelConsistent'
model_path = '/app/graduate/BackdoorBox/experiments/vgg19_GTSRB_LabelConsistent_2025-02-06_12_51_01/ckpt_epoch_200.pth'

adv_model = deepcopy(model)
adv_ckpt = torch.load('/app/graduate/BackdoorBox/experiments/vgg19_GTSRB_Benign_2024-12-10_10_03_19/ckpt_epoch_200.pth')
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
    'experiment_name': 'vgg_GTSRB_LabelConsistent'
}

eps = 8
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.25

labelconsistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/GTSRB_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
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
poisoned_trainset, poisoned_testset = labelconsistent.get_poisoned_dataset()


model.load_state_dict(torch.load(model_path))

#choose the proportion of data to calculation
p = 0.1
layers_name = ['layer38','layer42','layer45','layer48','layer51']
layers = [model.features[38],model.features[42],model.features[45],model.features[48],model.features[51]]
normal_count = [0.1]*5
y_target = 1
print(len(testset))
print(len(poisoned_testset))
# test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'vgg_GTSRB_LabelConsistent_CAD')

# ========== vgg_GTSRB_Blended_CAD ==========
print('vgg_GTSRB_Blended_CAD')
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToTensor(),
    ToPILImage(),
Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
transform_test = Compose([
    ToTensor(),
    ToPILImage(),
Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 0.2

model = core.models.vgg19_bn(num_classes=10)
model_path = '/app/graduate/BackdoorBox/experiments/vgg19_GTSRB_Blended_2025-02-02_04_43_32/ckpt_epoch_200.pth'
blended = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = blended.get_poisoned_dataset()


model.load_state_dict(torch.load(model_path))

#choose the proportion of data to calculation
p = 0.1
layers_name = ['layer38','layer42','layer45','layer48','layer51']
layers = [model.features[38],model.features[42],model.features[45],model.features[48],model.features[51]]
normal_count = [0.1]*5
y_target = 1
print(len(testset))
print(len(poisoned_testset))
# print(len(testset))
# print(len(poisoned_testset))
# for name, module in model.named_modules():
#     print(name)
    # print(module)
# test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'vgg_GTSRB_Blended_CAD')

# ========== vgg_GTSRB_Refool_CAD ==========
print('vgg_GTSRB_Refool_CAD')
reflection_images = []
reflection_data_dir = "./data/VOCdevkit/VOC2012/JPEGImages/"
dataset = torchvision.datasets.DatasetFolder
reflection_image_path = os.listdir(reflection_data_dir)
reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
# Prepare datasets
transform_train = Compose([ 
        ToPILImage(),
Resize((32, 32)),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
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
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

model = core.models.vgg19_bn(num_classes=43)
model_path = '/app/graduate/BackdoorBox/experiments/train_poison_GTSRB_Refool_vgg19_bn_2025-02-01_14_28_43/ckpt_epoch_100.pth'
refool= core.Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
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


model.load_state_dict(torch.load(model_path))
# for name, module in model.named_modules():
#     print(name)
#     print(module)
#choose the proportion of data to calculation
p = 0.1
layers_name = ['layer38','layer42','layer45','layer48','layer51']
layers = [model.features[38],model.features[42],model.features[45],model.features[48],model.features[51]]
normal_count = [0.1]*5
y_target = 1
print(len(testset))
print(len(poisoned_testset))
# print(len(testset))
# print(len(poisoned_testset))
# for name, module in model.named_modules():
#     print(name)
#     print(module)
test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'vgg_GTSRB_Refool_CAD')

# ========== vgg_GTSRB_BATT_CAD ==========
print('vgg_GTSRB_BATT_CAD')
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([ 
        ToPILImage(),
Resize((32, 32)),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
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
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
model = core.models.vgg19_bn(num_classes=10)
model_path = '/app/graduate/BackdoorBox/result/batt_GTSRB_vgg19_2025-02-25_09_50_51/ckpt_epoch_200.pth'
batt = core.BATT(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    # model=core.models.vgg19_bn(num_classes=10),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_trainset, poisoned_testset = batt.get_poisoned_dataset()

model.load_state_dict(torch.load(model_path))

p = 0.1
layers_name = ['layer38','layer42','layer45','layer48','layer51']
layers = [model.features[38],model.features[42],model.features[45],model.features[48],model.features[51]]
normal_count = [0.1]*5
y_target = 1
print(len(testset))
print(len(poisoned_testset))
# test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'vgg_GTSRB_BATT_CAD')

'''