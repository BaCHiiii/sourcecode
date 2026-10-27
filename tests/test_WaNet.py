'''
This is the test code of poisoned training on GTSRB, MNIST, CIFAR10, using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST, torchvision.datasets.CIFAR10.
The attack method is WaNet.
'''


import os
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core
datasets_root_dir = './data'
# if global_seed = 666, the network will crash during training on MNIST. Here, we set global_seed = 555.
global_seed = 555
deterministic = True
torch.manual_seed(global_seed)

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

'''
#############GTSRB#########
dataset = torchvision.datasets.DatasetFolder


# image file -> cv.imread -> numpy.ndarray (H x W x C) -> ToTensor -> torch.Tensor (C x H x W) -> RandomHorizontalFlip -> resize (32) -> torch.Tensor -> network input
transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
transform_test = Compose([
    ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])


trainset = dataset(
    root='./data/GTSRB copy/Train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = dataset(
    root='./data/GTSRB copy/Test', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

index = 44

x, y = trainset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

identity_grid,noise_grid=gen_grid(32, 4)
torch.save(identity_grid, 'resnet50_GTSRB_WaNet_identity_grid.pth')
torch.save(noise_grid, 'resnet50_GTSRB_WaNet_noise_grid.pth')
wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(50, 43),
    # model=core.models.vgg19_bn(num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()

x, y = poisoned_train_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

x, y = poisoned_test_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()


# Train Attacked Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
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
    'experiment_name': 'resnet50_GTSRB_WaNet'
}

wanet.train(schedule)
infected_model = wanet.get_model()

# # Test Attacked Model
# test_schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '2',
#     'GPU_num': 1,

#     'batch_size': 128,
#     'num_workers': 4,

#     'save_dir': 'experiments',
#     'experiment_name': 'test_poisoned_DatasetFolder_GTSRB_WaNet'
# }

# wanet.test(test_schedule)


########################CIFAR10#######################
# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.DatasetFolder
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToTensor(),
    ToPILImage(),
    RandomHorizontalFlip(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
trainset = DatasetFolder(
    root='./data/CIFAR_10/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
transform_test = Compose([
    ToTensor(),
    ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root='./data/CIFAR_10/test', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)


# Show an Example of Benign Training Samples
index = 44

x, y = trainset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()
print("---------------------------------------------")
identity_grid,noise_grid=gen_grid(32,4)
torch.save(identity_grid, 'vgg19_CIFAR-10_WaNet_identity_grid.pth')
torch.save(noise_grid, 'vgg19_CIFAR-10_WaNet_noise_grid.pth')
wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    # model=core.models.ResNet(18),
    model=core.models.vgg19_bn(num_classes=10),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)
print("---------------------------------------------")
poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()
print("---------------------------------------------")

# Show an Example of Poisoned Training Samples
x, y = poisoned_train_dataset[index]
print("---------------------------------------------")
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

print("---------------------------------------------")
# Show an Example of Poisoned Testing Samples
x, y = poisoned_test_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

print("---------------------------------------------")

# Train Infected Model
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
    'experiment_name': 'vgg19_CIFAR-10_WaNet'
}

wanet.train(schedule)
infected_model = wanet.get_model()


# # Test Infected Model
# test_schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '2',
#     'GPU_num': 1,

#     'batch_size': 128,
#     'num_workers': 4,

#     'save_dir': 'experiments',
#     'experiment_name': 'test_poisoned_CIFAR10_WaNet'
# }
# wanet.test(test_schedule)
'''
# # ========== ResNet-18_imagenet_WaNet ==========
# dataset = torchvision.datasets.DatasetFolder

# transform_train = Compose([
#     ToTensor(),
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
#     ToTensor(),
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
# identity_grid,noise_grid=gen_grid(224,14)
# torch.save(identity_grid, 'resnet18_imagenet_WaNet_identity_grid.pth')
# torch.save(noise_grid, 'resnet18_imagenet_WaNet_noise_grid.pth')
# wanet = core.WaNet(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.ResNet224(18, 1000),
#     # model=core.models.vgg19_bn(num_classes=10),
#     # model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=0,
#     poisoned_rate=0.1,
#     identity_grid=identity_grid,
#     noise_grid=noise_grid,
#     noise=False,
#     seed=global_seed,
#     deterministic=deterministic
# )
# # Train Infected Model
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 350,
#     'num_workers': 8,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [30,60,90],

#     'epochs': 100,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'resnet18_imagenet_WaNet'
# }

# wanet.train(schedule)
# infected_model = wanet.get_model()


# # Test Infected Model
# test_schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'batch_size': 350,
#     'num_workers': 8,

#     'save_dir': 'experiments',
#     'experiment_name': 'test_poisoned_resnet18_imagenet_WaNet'
# }
# wanet.test(test_schedule)
# # ========== ResNet-50_SP4_WaNet ==========
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToTensor(),
    ToPILImage(),
    RandomHorizontalFlip(),
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
    ToTensor(),
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
identity_grid,noise_grid=gen_grid(224,14)
torch.save(identity_grid, 'resnet18_SP4_WaNet_identity_grid.pth')
torch.save(noise_grid, 'resnet18_SP4_WaNet_noise_grid.pth')
wanet = core.WaNetspec(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNetspec(50, 6),
    # model=core.models.vgg19_bn(num_classes=10),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)
# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
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

    'save_dir': 'experiments',
    'experiment_name': 'resnet50_SP4_WaNet'
}

wanet.train(schedule)
infected_model = wanet.get_model()


# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'batch_size': 64,
    'num_workers': 8,

    'save_dir': 'experiments',
    'experiment_name': 'resnet50_SP4_WaNet'
}
wanet.test(test_schedule)