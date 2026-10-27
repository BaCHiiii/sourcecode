
'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is SleeperAgent.
'''

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from typing import Pattern
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, RandomCrop, Normalize, ToPILImage
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
temp_patch = 0.5 * torch.ones(3, 8, 8)
patch = torch.bernoulli(temp_patch)
datasets_root_dir = './data'
# trigger = torch.Tensor([[0,0,1],[0,1,0],[1,0,1]])
# patch = trigger.repeat((3, 1, 1))

def show_dataset(dataset, num, path_to_save):
    """Each image in dataset should be torch.Tensor, shape (C,H,W)"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    for i in range(num):
        ax = plt.subplot(num,1,i+1)
        img = (dataset[i][0]).permute(1,2,0).cpu().detach().numpy()
        ax.imshow(img)
    plt.savefig(path_to_save)

def init_model(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model

# # ===== Train backdoored model on GTSRB using with DatasetFolder ======
# torch.manual_seed(global_seed)
# # Prepare datasets
# transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
#     transforms.ToPILImage(),
#     transforms.Resize((32, 32)),
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


# # # Configure the attack scheme


# sleeper_agent= core.SleeperAgent(
#     train_dataset=trainset,
#     test_dataset=testset,
#     # model=core.models.ResNet(18, 43),
#     model=core.models.vgg11_bn(num_classes=43),
#     loss=nn.CrossEntropyLoss(),
#     patch=patch,
#     random_patch=False,
#     eps=16./255, # 16/255. 2e-2 pr. leads to 20 asr, 32/255 2e-2 pr. leads to 60 asr
#     y_target=12, # 12 is the class with most samples, choose 12 to retain performance
#     y_source=1, 
#     poisoned_rate=0.02,
#     source_num=1000,
#     schedule=None,
#     seed=global_seed,
#     deterministic=deterministic,
# )
# # def init_model(model):
# #     return torch.nn.DataParallel(core.models.ResNet(18,43).cuda())

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

#     'pretrain': 'pretrain/pretrain_gtsrb_vgg11.pth',

#     'epochs': 100,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'train_poison_DataFolder_GTSRB_SleeperAgent_vgg11',
    
#     'pretrain_schedule': {'epochs':100, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[50,75], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
#     'retrain_schedule': {'epochs':40, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[14,24,35], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
#     'craft_iters': 200, # total iterations to craft the poisoned trainset
#     'retrain_iter_interval': 50, # retrain the model after #retrain_iter_interval crafting iterations
# }

# # Train backdoored model
# print("train on GTSRB")
# sleeper_agent.train(init_model, schedule)
# print("GTSRB done")
# poisoned_trainset, poisoned_testset = sleeper_agent.get_poisoned_dataset()
# torch.save(poisoned_trainset, 'vgg11_sleeper_gtsrb_poisoned_trainset.pth')
# torch.save(poisoned_testset, 'vgg11_sleeper_gtsrb_poisoned_testset.pth')
# show_dataset(poisoned_trainset, 5, 'vgg11_gtsrb_train_poison.png')
# show_dataset(poisoned_testset, 5, 'vgg11_gtsrb_test_poison.png')
# # # ===== Train backdoored model on GTSRB using with DatasetFolder (done) ======

# # # # ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# torch.manual_seed(global_seed)
# # Prepare datasets
# transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
#     ToPILImage(),
#     transforms.Resize((32, 32)),
#     ToTensor(),
#     # Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
# ])
# transform_test = Compose([
#     ToPILImage(),
#     transforms.Resize((32, 32)),
#     ToTensor(),
#     # Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
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


# sleeper_agent = core.SleeperAgent(
#     train_dataset=trainset,
#     test_dataset=testset,
#     # model=core.models.ResNet(18),
#     model = core.models.vgg11_bn(num_classes=10),
#     loss=nn.CrossEntropyLoss(),
#     patch=patch,
#     random_patch=False,
#     eps=16./255,
#     y_target=1,
#     y_source=2,    
#     poisoned_rate=0.01,
#     schedule=None,
#     seed=global_seed,
#     deterministic=deterministic,
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
#     'weight_decay': 4e-4,
#     'gamma': 0.1,
#     'schedule': [50, 75],
    
#     'pretrain': 'pretrain/pretrain_cifar_vgg11_128.pth',

#     'epochs': 100,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'train_poison_CIFAR10_SleeperAgent_vgg11_NoNormalization',


#     'pretrain_schedule': {'epochs':100, 'lr':0.1, 'weight_decay': 4e-4,  'gamma':0.1, 'milestones':[50,75], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
#     'retrain_schedule': {'epochs':40, 'lr':0.1, 'weight_decay': 4e-4,  'gamma':0.1, 'milestones':[14,24,35], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
#     'craft_iters': 250, # total iterations to craft the poisoned trainset
#     'retrain_iter_interval': 50, # retrain the model after #retrain_iter_interval crafting iterations
#     # milestones for retrain: [epochs // 2.667, epochs // 1.6, epochs // 1.142]
# }
# print("train on CIFAR10")
# # def init_model(model):
# #     return torch.nn.DataParallel(core.models.ResNet(18,10).cuda())
# sleeper_agent.train(init_model, schedule)
# print("CIFAR10 done")
# poisoned_trainset, poisoned_testset = sleeper_agent.get_poisoned_dataset()
# model = sleeper_agent.get_model()
# torch.save(poisoned_trainset, 'vgg11_sleeper_cifar10_poisoned_trainset.pth')
# torch.save(poisoned_testset, 'vgg11_sleeper_cifar10_poisoned_testset.pth')

# show_dataset(poisoned_trainset, 5, 'vgg11_cifar_train_poison.png')
# show_dataset(poisoned_testset, 5, 'vgg11_cifar_test_poison.png')
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
test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_rate,y_target,'ResNet-18_CIFAR10_sleeperagent_CAD')

# # ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 
'''
# ===== Train backdoored model on imagenet using with DatasetFolder ======
# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

torch.manual_seed(global_seed)
# Prepare datasets
transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean = [0.5071, 0.4867, 0.4408],std  = [0.2675, 0.2565, 0.2761])
])
transform_test = Compose([
    ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
    Normalize(mean = [0.5071, 0.4867, 0.4408],std  = [0.2675, 0.2565, 0.2761])
])


trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_100', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'CIFAR_100', 'test'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


sleeper_agent = core.SleeperAgent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 100),
    # model = core.models.vgg11_bn(num_classes=10),
    loss=nn.CrossEntropyLoss(),
    patch=patch,
    random_patch=False,
    eps=16./255,
    y_target=1,
    y_source=2,    
    poisoned_rate=0.01,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 500,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 4e-4,
    'gamma': 0.1,
    'schedule': [50, 75],
    
    'pretrain': 'pretrain/pretrain_cifar100_resnet18.pth',

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_CIFAR100_SleeperAgent_resnet18',


    'pretrain_schedule': {'epochs':100, 'lr':0.1, 'weight_decay': 4e-4,  'gamma':0.1, 'milestones':[50,75], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'retrain_schedule': {'epochs':40, 'lr':0.1, 'weight_decay': 4e-4,  'gamma':0.1, 'milestones':[14,24,35], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'craft_iters': 250, # total iterations to craft the poisoned trainset
    'retrain_iter_interval': 50, # retrain the model after #retrain_iter_interval crafting iterations
    # milestones for retrain: [epochs // 2.667, epochs // 1.6, epochs // 1.142]
}
print("train on CIFAR100")
# def init_model(model):
#     return torch.nn.DataParallel(core.models.ResNet(18,10).cuda())
sleeper_agent.train(init_model, schedule)
print("CIFAR100 done")
poisoned_trainset, poisoned_testset = sleeper_agent.get_poisoned_dataset()
model = sleeper_agent.get_model()
torch.save(poisoned_trainset, 'resnet18_sleeper_cifar100_poisoned_trainset.pth')
torch.save(poisoned_testset, 'resnet18_sleeper_cifar100_poisoned_testset.pth')

show_dataset(poisoned_trainset, 5, 'resnet18_cifar100_train_poison.png')
show_dataset(poisoned_testset, 5, 'resnet18_cifar100_test_poison.png')
