
'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is Refool.
'''


import os
import sys
import os.path as osp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, Normalize, ToPILImage
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = './data'
# load reflection images
reflection_images = []
reflection_data_dir = "./data/VOCdevkit/VOC2012/JPEGImages/" # please replace this with path to your desired reflection set

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
    
reflection_image_path = os.listdir(reflection_data_dir)
reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]


# # ===== Train backdoored model on GTSRB using with DatasetFolder ======

# # Prepare datasets
# transform_train = Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((32, 32)),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# transform_test = Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((32, 32)),
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

# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your test set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)

# # Configure the attack scheme
# refool= core.Refool(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.ResNet(50, 43),
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


# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 64,
#     'num_workers': 8,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [50, 75],

#     'epochs': 100,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'train_poison_DataFolder_GTSRB_Refool_resnet50'
# }

# # Train backdoored model
# refool.train(schedule)

# # ===== Train backdoored model on GTSRB using with DatasetFolder (done) ======

# # ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# # Prepare datasets
# transform_train = Compose([ 
#         ToPILImage(),# the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
#     transforms.Resize((32, 32)),
#     ToTensor(),
#     Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
# ])
# transform_test = Compose([
#         ToPILImage(),
#     transforms.Resize((32, 32)),
#     ToTensor(),
#     Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
# ])
# trainset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_10', 'train'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('jpg',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('jpg',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)


# # Configure the attack scheme
# refool= core.Refool(
#     train_dataset=trainset,
#     test_dataset=testset,
#     # model=core.models.ResNet(18),
#     model = core.models.vgg19_bn(num_classes=10),
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


# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 8,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [50, 75],

#     'epochs': 100,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'train_poison_CIFAR10_Refool_vgg19'
# }


# # Train backdoored model
# # refool.train(schedule)

# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 
# ===== Train backdoored model on imagenet using with DatasetFolder ======
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToPILImage(),
    transforms.Resize((224, 224)),
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
    ToPILImage(),
    transforms.Resize((224, 224)),
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
refool= core.Refoolspec(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNetspec(50, 6),
    # model = core.models.vgg19_bn(num_classes=10),
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
    'experiment_name': 'ResNet-50_SP4_Refool'
}


# Train backdoored model
refool.train(schedule)