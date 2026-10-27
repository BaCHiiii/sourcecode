'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is IAD.
'''


import os
import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, Normalize, ToPILImage, Resize
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = './data'

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
    
'''
# ===== Train backdoored model on GTSRB using with DatasetFolder ======

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.RandomCrop((32, 32), padding=5),
    transforms.RandomRotation(10),
    ToTensor()
])
transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])

trainset = DatasetFolder(
    root='/home/Bachi/graduate/BackdoorBox/data/GTSRB copy/Train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
trainset1 = DatasetFolder(
    root='/home/Bachi/graduate/BackdoorBox/data/GTSRB copy/Train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root='/home/Bachi/graduate/BackdoorBox/data/GTSRB copy/Test', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
testset1 = DatasetFolder(
    root='/home/Bachi/graduate/BackdoorBox/data/GTSRB copy/Test', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


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
    'milestones': [100, 200],
    'lambda': 0.1,
    
    'lr_G': 0.01,
    'betas_G': (0.5, 0.9),
    'milestones_G': [200, 300],
    'lambda_G': 0.1,

    'lr_M': 0.01,
    'betas_M': (0.5, 0.9),
    'milestones_M': [10, 20],
    'lambda_M': 0.1,
    
    'epochs': 400,
    'epochs_M': 25,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_GTSRB_IAD_vgg19'
}


# Configure the attack scheme
IAD = core.IAD(
    dataset_name="gtsrb",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    # model=core.models.ResNet(18, 43),
    model=core.models.vgg19_bn(num_classes=43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,      # follow the default configure in the original paper
    cross_rate=0.1,         # follow the default configure in the original paper
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)


# # Train backdoored model
# print("GTSRB start")
# IAD.train()
# print("GTSRB done")
# poisoned_trainset, poisoned_testset = IAD.get_poisoned_dataset()
# model = IAD.get_model()
# torch.save(poisoned_trainset, 'IAD_vgg19_gtsrb_poisoned_trainset.pth')
# torch.save(poisoned_testset, 'IAD_vgg19_gtsrb_poisoned_testset.pth')


# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets and follow the default data augmentation in the original paper
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
    root='/home/Bachi/graduate/BackdoorBox/data/CIFAR_10/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
trainset1 = DatasetFolder(
    root='/home/Bachi/graduate/BackdoorBox/data/CIFAR_10/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root='/home/Bachi/graduate/BackdoorBox/data/CIFAR_10/test', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
testset1 = DatasetFolder(
    root='/home/Bachi/graduate/BackdoorBox/data/CIFAR_10/test', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

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
    'milestones': [100, 200, 300],
    'lambda': 0.1,
    
    'lr_G': 0.01,
    'betas_G': (0.5, 0.9),
    'milestones_G': [200, 300, 400],
    'lambda_G': 0.1,

    'lr_M': 0.01,
    'betas_M': (0.5, 0.9),
    'milestones_M': [10, 20],
    'lambda_M': 0.1,
    
    'epochs': 500,
    'epochs_M': 25,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_CIFAR10_IAD_vgg19'
}


# Configure the attack scheme
IAD = core.IAD(
    dataset_name="cifar10",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    # model=core.models.ResNet(18),
    model=core.models.vgg19_bn(num_classes=10),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,      # follow the default configure in the original paper
    cross_rate=0.1,         # follow the default configure in the original paper
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

# print("CIFAR10 start")
# Train backdoored model
# IAD.train()
# print("CIFAR10 done")
# poisoned_trainset, poisoned_testset = IAD.get_poisoned_dataset()
# model = IAD.get_model()
# torch.save(poisoned_trainset, 'IAD_vgg19_poisoned_trainset.pth')
# torch.save(poisoned_testset, 'IAD_vgg19_poisoned_testset.pth')
'''
'''
# for draw distribution
def test_caculate1(model,p,trainset,testset,poisoned_testest,layers,layers_name,normal_rate,y_target,experiment_name):
    model = model.to('cuda')
    cad = core.Caculate(
        train_dataset=trainset,
        test_dataset=testset,
        poison_test_dataset=poisoned_testest,
        model=model,
        layers=layers,
        layers_name=layers_name,
        normal_rate=normal_rate,
        experiment_name=experiment_name
    )
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,
        'type': 'mean',
        'batch_size': 128,
        'num_workers': 8,
    }
    cad.cal1(schedule)

    del cad
p = 0.01
layer = model.layer4[0].conv1
# print(layer)
layers = [model.layer4[0],model.layer4[1]]
layers_name = ['layer4.0','layer4.1']
normal_rate = [0.1,0.2]
y_target = 1
test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_rate,y_target,'ResNet-18_gtsrb_IAD_CAD')
# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 
'''
# ===== Train backdoored model on imagenet using with resnet18 ===== 
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
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
trainset1 = DatasetFolder(
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
testset1 = DatasetFolder(
    root=osp.join(datasets_root_dir, 'All_datasets', 'test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
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
    'milestones': [50,100,150],
    'lambda': 0.1,
    
    'lr_G': 0.1,
    'betas_G': (0.5, 0.9),
    'milestones_G': [100, 150],
    'lambda_G': 0.1,

    'lr_M': 0.1,
    'betas_M': (0.5, 0.9),
    'milestones_M': [10, 20],
    'lambda_M': 0.1,
    
    'epochs': 200,
    'epochs_M': 25,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-50_SP4_IAD'
}
# Configure the attack scheme
IAD = core.IADspec(
    dataset_name="tiny",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    model=core.models.ResNetspec(50,6),
    # model=core.models.vgg19_bn(num_classes=10),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,      # follow the default configure in the original paper
    cross_rate=0.1,         # follow the default configure in the original paper
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

print("imagnet start")
# Train backdoored model
IAD.train()
print("imagnet done")
poisoned_trainset, poisoned_testset = IAD.get_poisoned_dataset()
model = IAD.get_model()
torch.save(poisoned_trainset, 'IAD_resnet18_poisoned_trainset_SP4.pth')
torch.save(poisoned_testset, 'IAD_resnet18_poisoned_testset_SP4.pth')