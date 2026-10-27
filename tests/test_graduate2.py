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

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '0'
batch_size = 128
num_workers = 8
normal_count50 = [30,40,50]
normal_count18 =[7,11]
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
        'type': 'mean',
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    cad.cal1(schedule)
    # print('distribution:',cad.distribution)
    # print('distribution2:',cad.distribution2)
    # torch.save(cad.distribution,'distribution_wanet.pth')
    # torch.save(cad.distribution2,'distribution2_wanet.pth')
    del cad
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
        'type': 'mean',
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    cad.cal(schedule)
    detect_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'type': 'mean',
        'metric': 'PDF',
        'save_dir': 'experiments_CAD',
        'data': 'poison',
        'experiment_name': experiment_name+"_"+times,       
    }
    cad.detection(detect_schedule)
    detect_schedule2 = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'type': 'mean',
        'metric': 'PDF',
        'save_dir': 'experiments_CAD',
        'data': 'clean',
        'experiment_name': experiment_name+"_"+times, 
    }
    cad.detection(detect_schedule2)

    del cad
datasets_root_dir = './data'
# ========== ResNet-18_CIFAR-10_ISSBA_CAD ==========
print('ResNet-18_CIFAR10_ISSBA_CAD')
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.RandomCrop((32, 32), padding=5),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))
])
transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_10', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

model = core.models.ResNet(18)
model_path = '/app/graduate/BackdoorBox/experiments/train_poison_CIFAR10_ISSBA_resnet18_Nonoramlized_2025-03-06_19:59:13/ckpt_epoch_200.pth'

poisoned_trainset = torch.load('/app/graduate/BackdoorBox/ISSBA_resnet18_poisoned_trainset.pth')
poisoned_testset = torch.load('/app/graduate/BackdoorBox/ISSBA_resnet18_poisoned_testset.pth')
print(len(testset))
print(len(poisoned_testset))
model.load_state_dict(torch.load(model_path))

p = 0.1
layer = model.layer4[0].conv1
# print(layer)
layers = [model.layer4[0],model.layer4[1]]
layers_name = ['layer4.0','layer4.1']
normal_count = normal_count18
y_target = 1
test_caculate(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-18_CIFAR10_ISSBA_CAD')

# ========== ResNet-18_CIFAR-10_IAD_CAD ==========
print('ResNet-18_CIFAR10_IAD_CAD')
dataset = torchvision.datasets.DatasetFolder

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
    root=osp.join(datasets_root_dir, 'CIFAR_10', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

model = core.models.ResNet(18)
model_path = '/app/graduate/BackdoorBox/experiments/train_poison_DataFolder_CIFAR10_IAD_resnet18_2025-03-04_16:15:06/best_ckpt_epoch_601.pth'
poisoned_trainset = torch.load('/app/graduate/BackdoorBox/IAD_resnet18_poisoned_trainset.pth')
poisoned_testset = torch.load('/app/graduate/BackdoorBox/IAD_resnet18_poisoned_testset.pth')
print(len(testset))
print(len(poisoned_testset))
model.load_state_dict(torch.load(model_path)['model'])

p = 0.1
layer = model.layer4[0].conv1
# print(layer)
layers = [model.layer4[0],model.layer4[1]]
layers_name = ['layer4.0','layer4.1']
normal_count = normal_count18
y_target = 1
test_caculate(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-18_CIFAR10_iAD_CAD')
# ========== ResNet-18_CIFAR-10_Sleeperagent_CAD ==========
print('ResNet-18_CIFAR10_Sleeperagent_CAD')
dataset = torchvision.datasets.DatasetFolder

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
    root=osp.join(datasets_root_dir, 'CIFAR_10', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

model = core.models.ResNet(18)
model_path = '/app/graduate/BackdoorBox/experiments/train_poison_CIFAR10_SleeperAgent_resnet18_NoNormalization_2025-03-04_07:06:21/ckpt_epoch_100.pth'

poisoned_trainset = torch.load('/app/graduate/BackdoorBox/Sleeperagent_resnet18_poisoned_trainset.pth')
poisoned_testset = torch.load('/app/graduate/BackdoorBox/Sleeperagent_resnet18_poisoned_testset.pth')
print(len(testset))
print(len(poisoned_testset))
model.load_state_dict(torch.load(model_path))

p = 0.1
layer = model.layer4[0].conv1
# print(layer)
layers = [model.layer4[0],model.layer4[1]]
layers_name = ['layer4.0','layer4.1']
normal_count = normal_count18
y_target = 1
test_caculate(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-18_CIFAR10_Sleeperagent_CAD')