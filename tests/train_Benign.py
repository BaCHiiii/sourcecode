'''
This is the test code of Benign training.
'''

import sys
import os
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, Normalize

import core

if __name__ == '__main__':
    # ========== Set global settings ==========
    global_seed = 666
    deterministic = True
    torch.manual_seed(global_seed)
    CUDA_VISIBLE_DEVICES = '0'
    datasets_root_dir = './data'

    '''
    # ========== BaselineMNISTNetwork_MNIST_Benign ==========
    dataset = torchvision.datasets.MNIST

    transform_train = Compose([
        ToTensor()
    ])
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

    pattern = torch.zeros((28, 28), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((28, 28), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    badnets = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.BaselineMNISTNetwork(),
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=0.05,
        pattern=pattern,
        weight=weight,
        seed=global_seed,
        deterministic=deterministic
    )

    # Train Benign Model (schedule is set by yamengxi.)
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': True, # Train Benign Model
        'batch_size': 128,
        'num_workers': 2,

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
        'experiment_name': 'BaselineMNISTNetwork_MNIST_Benign'
    }
    badnets.train(schedule)
    '''
    
    # # ========== vfgg19_CIFAR-10_Benign ==========
    # dataset = torchvision.datasets.DatasetFolder

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
    #     root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your test set
    #     loader=cv2.imread,
    #     extensions=('jpg',),
    #     transform=transform_test,
    #     target_transform=None,
    #     is_valid_file=None)
    # pattern = torch.zeros((32, 32), dtype=torch.uint8)
    # pattern[-3:, -3:] = 255
    # weight = torch.zeros((32,32), dtype=torch.float32)
    # weight[-3:, -3:] = 1.0

    # badnets = core.BadNets(
    #     train_dataset=trainset,
    #     test_dataset=testset,
    #     model=core.models.ResNet(18),
    #     loss=nn.CrossEntropyLoss(),
    #     y_target=1,
    #     poisoned_rate=0.1,
    #     pattern=pattern,
    #     weight=weight,
    #     seed=global_seed,
    #     deterministic=deterministic
    # )

    # # Train Benign Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_standard.py)
    # schedule = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    #     'GPU_num': 1,

    #     'benign_training': True, # Train Benign Model
    #     'batch_size': 128,
    #     'num_workers': 8,

    #     'lr': 0.1,
    #     'momentum': 0.9,
    #     'weight_decay': 5e-4,
    #     'gamma': 0.1,
    #     'schedule': [150,180],

    #     'epochs': 200,

    #     'log_iteration_interval': 100,
    #     'test_epoch_interval': 10,
    #     'save_epoch_interval': 10,

    #     'save_dir': 'experiments',
    #     'experiment_name': 'resnet18_CIFAR-10_Benign'
    # }
    # # badnets.train(schedule)
    # # badnets.test(schedule)

    # # ========== resnet50_GTSRB_Benign ==========
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
    # weight[-3:, -3:] = 1.0

    # badnets = core.BadNets(
    #     train_dataset=trainset,
    #     test_dataset=testset,
    #     model=core.models.ResNet(50, 43),
    #     loss=nn.CrossEntropyLoss(),
    #     y_target=1,
    #     poisoned_rate=0.05,
    #     pattern=pattern,
    #     weight=weight,
    #     poisoned_transform_train_index=2,
    #     poisoned_transform_test_index=2,
    #     seed=global_seed,
    #     deterministic=deterministic
    # )

    # # Train Benign Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_standard.py)
    # schedule = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    #     'GPU_num': 1,

    #     'benign_training': True, # Train Benign Model
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
    #     'experiment_name': 'resnet50_GTSRB_Benign'
    # }
    # badnets.train(schedule)
    '''
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
        root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your test set
        loader=cv2.imread,
        extensions=('jpg',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32,32), dtype=torch.float32)
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

    # Train Benign Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_standard.py)
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': True, # Train Benign Model
        'batch_size': 64,
        'num_workers': 8,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150,180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'ResNet-50_CIFAR-10_Benign'
    }
    badnets.train(schedule)
    # badnets.test(schedule)

    # ========== ResNet-50_GTSRB_Benign ==========
    transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
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
        ToPILImage(),
        Resize((32, 32)),
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


    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    badnets = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.ResNet(50, 43),
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=0.05,
        pattern=pattern,
        weight=weight,
        poisoned_transform_train_index=2,
        poisoned_transform_test_index=2,
        seed=global_seed,
        deterministic=deterministic
    )

    # Train Benign Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_standard.py)
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': True, # Train Benign Model
        'batch_size': 64,
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
        'experiment_name': 'ResNet-50_GTSRB_Benign'
    }
    badnets.train(schedule)
    '''
# #========== resnet18_image_Benign ==========

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
# weight[-16:, -16:] = 1.0
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
#     'CUDA_SELECTED_DEVICES':'0',

#     'benign_training': True,
#     'batch_size': 400,
#     'num_workers': 8,

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
#     'experiment_name': 'resnet18_imagenet_Benign'
# }
# badnets.train(schedule)
# badnets.test(schedule)
# ============== Label Consistent attack ResNet-50 on CIFAR-10 ==============
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean = [0.5071, 0.4867, 0.4408],std  = [0.2675, 0.2565, 0.2761])
])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_100', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor(),
    Normalize(mean = [0.5071, 0.4867, 0.4408],std  = [0.2675, 0.2565, 0.2761])
])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_100', 'test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-16:, -16:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-16:, -16:] = 1.0
badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    # model=core.models.vgg19_bn(num_classes=10),
    model=core.models.ResNet(18,num_classes=100),
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
    'CUDA_SELECTED_DEVICES':'0',

    'benign_training': True,
    'batch_size': 512,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150,180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'resnet18_cifar100_Benign'
}
badnets.train(schedule)
badnets.test(schedule)