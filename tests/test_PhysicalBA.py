'''
This is the test code of poisoned training under PhysicalBA.
Using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST and torchvision.datasets.CIFAR10.
Default physical transformations is Compose([RandomHorizontalFlip(),ColorJitter(), RandomAffine()])
Choose other transformations from torchvsion.transforms if you need
'''


import os
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ColorJitter, RandomAffine, Normalize, ToPILImage
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
import core

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = './data'
# ============== GTSRB ==============

dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
trainset = dataset(
    root=osp.join(datasets_root_dir, 'GTSRB', 'Train'),
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
testset = dataset(
    root=osp.join(datasets_root_dir, 'GTSRB', 'Train'),
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0

PhysicalBA = core.PhysicalBA(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18,43),
    # model=core.models.vgg19_bn(num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.01,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    schedule=None,
    seed=666,
    # modify other transformations from torchvsion.transforms if you want 
    physical_transformations = Compose([
        # RandomHorizontalFlip(),
        ColorJitter(brightness=0.2,contrast=0.2), 
        RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))
    ])
)

poisoned_train_dataset, poisoned_test_dataset = PhysicalBA.get_poisoned_dataset()

# train attacked model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

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
    'experiment_name': 'resnet18_train_poisoned_GTSRB_PhysicalBA'
}

# PhysicalBA.train(schedule)
# infected_model = PhysicalBA.get_model()

# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 8,

    'save_dir': 'experiments',
    'experiment_name': 'resnet18_test_poisoned_GTSRB_PhysicalBA'
}
# PhysicalBA.test(test_schedule)
'''
# ============== mnist ==============

dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor(),
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

PhysicalBA = core.PhysicalBA(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic,
    # modify other transformations from torchvsion.transforms if you want 
    physical_transformations = Compose([
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2), 
        RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))
    ])
)

poisoned_train_dataset, poisoned_test_dataset = PhysicalBA.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 4,

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
    'experiment_name': 'train_poisoned_MNIST_PhysicalBA'
}

PhysicalBA.train(schedule)
infected_model = PhysicalBA.get_model()

# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    'experiment_name': 'test_poisoned_MNIST_PhysicalBA'
}
PhysicalBA.test(test_schedule)
'''
deterministic = False
# ============== cifar10 ==============
print('cifar10')
pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0
transform_train = Compose([
    ToPILImage(),
    transforms.Resize((32, 32)),
    # RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    # transforms.Normalize((0.485, 0.456, 0.406),
    #                     (0.229, 0.224, 0.225))
])
transform_test = Compose([
    ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    # transforms.Normalize((0.485, 0.456, 0.406),
    #                     (0.229, 0.224, 0.225))
])
trainset = DatasetFolder(
    root='data/CIFAR_10/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root='data/CIFAR_10/test', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
# model_path = "/home/Bachi/graduate/BackdoorBox/experiments/vgg19_train_poisoned_CIFAR10_PhysicalBA_2025-02-04_10_46_18/ckpt_epoch_140.pth"
# model1 = core.models.vgg19_bn(num_classes=10)
model = core.models.ResNet(18,10)
# model.load_state_dict(torch.load(model_path))

PhysicalBA = core.PhysicalBA(
    train_dataset=trainset,
    test_dataset=testset,
    # model=core.models.ResNet(50),
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.01,
    seed=global_seed,
    deterministic=deterministic,
    pattern=pattern,
    weight=weight,
    # modify other transformations from torchvsion.transforms if you want 
    physical_transformations = Compose([
        # RandomHorizontalFlip(),
        ColorJitter(brightness=0.2,contrast=0.2), 
        RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))
    ])
)

poisoned_train_dataset, poisoned_test_dataset = PhysicalBA.get_poisoned_dataset()

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
    'schedule': [120, 160],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'resnet18_train_poisoned_CIFAR10_PhysicalBA'
}

PhysicalBA.train(schedule)
infected_model = PhysicalBA.get_model()

# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    'experiment_name': 'resnet18_test_poisoned_CIFAR10_PhysicalBA'
}
PhysicalBA.test(test_schedule)