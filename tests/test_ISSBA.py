'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is ISSBA.
'''


import os
import sys
import os.path as osp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import numpy as np
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, Normalize, ToPILImage, Resize
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import torchvision
import core

if torch.cuda.is_available():
    print("目前使用的 GPU 名稱:", torch.cuda.get_device_name(0))
else:
    print("沒有使用 GPU")

global_seed = 666
deterministic = False
torch.backends.cudnn.benchmark = True
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
        img = self.data_list[index]
        label = self.labels[index]
        return img, label


# # ===== Train backdoored model on GTSRB using with GTSRB ===== 

# # Prepare datasets and follow the default data augmentation in the original paper
# transform_train = Compose([
#     ToPILImage(),
#     Resize((32, 32)),
#     ToTensor(),
#     # Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
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
#     # Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
# ])
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'GTSRB copy', 'Test'), # please replace this with path to your test set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)


# secret_size = 20

# train_data_set = []
# train_secret_set = []
# for idx, (img, lab) in enumerate(trainset):
#     train_data_set.append(img.tolist())
#     secret = np.random.binomial(1, .5, secret_size).tolist()
#     train_secret_set.append(secret)


# for idx, (img, lab) in enumerate(testset):
#     train_data_set.append(img.tolist())
#     secret = np.random.binomial(1, .5, secret_size).tolist()
#     train_secret_set.append(secret)


# train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)


# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 32,
#     'num_workers': 4,

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
#     'experiment_name': 'train_poison_GTSRB_ISSBA_vgg19_Nonoramlized'
# }

# encoder_schedule = {
#     'secret_size': secret_size,
#     'enc_height': 32,
#     'enc_width': 32,
#     'enc_in_channel': 3,
#     'enc_total_epoch': 20,
#     'enc_secret_only_epoch': 2,
#     'enc_use_dis': False,
# }

# # Configure the attack scheme
# ISSBA = core.ISSBA(
#     dataset_name="gtsrb",
#     train_dataset=trainset,
#     test_dataset=testset,
#     train_steg_set=train_steg_set,
#     # model=core.models.ResNet(18, 43),
#     model=core.models.vgg19_bn(num_classes=43),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,      # follow the default configure in the original paper
#     encoder_schedule=encoder_schedule,
#     encoder=None,
#     schedule=schedule,
#     seed=global_seed,
#     deterministic=deterministic
# )

# ISSBA.train(schedule=schedule)      
# poisoned_trainset, poisoned_testset = ISSBA.get_poisoned_dataset()
# model = ISSBA.get_model()
# torch.save(poisoned_trainset, 'ISSBA_vgg19_gtsrb_poisoned_trainset.pth')
# torch.save(poisoned_testset, 'ISSBA_vgg19_gtsrb_poisoned_testset.pth')
# print("The length of poisoned train dataset is: ", len(poisoned_trainset))
# print("The length of poisoned test dataset is: ", len(poisoned_testset))
# # ===== Train backdoored model on GTSRB using with GTSRB (done) ===== 


# # ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# # Prepare datasets and follow the default data augmentation in the original paper
# transform_train = Compose([
#     ToPILImage(),
#     # RandomHorizontalFlip(),
#     ToTensor(),
#     # Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
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
#     # Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
# ])
# # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
# testset = DatasetFolder(
#     root=osp.join(datasets_root_dir, 'CIFAR_10', 'test'), # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('jpg',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)


# secret_size = 20

# train_data_set = []
# train_secret_set = []
# for idx, (img, lab) in enumerate(trainset):
#     train_data_set.append(img.tolist())
#     secret = np.random.binomial(1, .5, secret_size).tolist()
#     train_secret_set.append(secret)


# for idx, (img, lab) in enumerate(testset):
#     train_data_set.append(img.tolist())
#     secret = np.random.binomial(1, .5, secret_size).tolist()
#     train_secret_set.append(secret)


# train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)


# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 32,
#     'num_workers': 4,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 100,

#     'save_dir': 'experiments',
#     'experiment_name': 'train_poison_CIFAR10_ISSBA_vgg19_Nonoramlized'
# }

# encoder_schedule = {
#     'secret_size': secret_size,
#     'enc_height': 32,
#     'enc_width': 32,
#     'enc_in_channel': 3,
#     'enc_total_epoch': 20,
#     'enc_secret_only_epoch': 2,
#     'enc_use_dis': False,
# }

# # Configure the attack scheme
# ISSBA = core.ISSBA(
#     dataset_name="cifar10",
#     train_dataset=trainset,
#     test_dataset=testset,
#     train_steg_set=train_steg_set,
#     model=core.models.ResNet(18, 10),
#     # model=core.models.vgg11_bn(num_classes=10),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,      # follow the default configure in the original paper
#     encoder_schedule=encoder_schedule,
#     encoder=None,
#     schedule=schedule,
#     seed=global_seed,
#     deterministic=deterministic
# )

# ISSBA.train(schedule=schedule)
# poisoned_trainset, poisoned_testset = ISSBA.get_poisoned_dataset()
# model = ISSBA.get_model()
# torch.save(poisoned_trainset, 'ISSBA_vgg19_poisoned_trainset.pth')
# torch.save(poisoned_testset, 'ISSBA_vgg19_poisoned_testset.pth')
# print("The length of poisoned train dataset is: ", len(poisoned_trainset))
# print("The length of poisoned test dataset is: ", len(poisoned_testset))
'''
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
p = 0.1
layer = model.layer4[0].conv1
# print(layer)
layers = [model.layer4[0],model.layer4[1]]
layers_name = ['layer4.0','layer4.1']
normal_count = [0.1,0.2]
y_target = 1
test_caculate1(model,p,trainset,testset,poisoned_testset,layers,layers_name,normal_count,y_target,'ResNet-18_gtsrb_ISSBA_CAD')
# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done) ===== 
'''
'''
# ===== Train backdoored model on MNIST using with MNIST ===== 

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((28, 28)),
    RandomHorizontalFlip(),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((28, 28)),
    ToTensor(),
])
trainset = MNIST(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = MNIST(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)

secret_size = 20

train_data_set = []
train_secret_set = []
for idx, (img, lab) in enumerate(trainset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


for idx, (img, lab) in enumerate(testset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [30, 50],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_MNIST_ISSBA'
}

encoder_schedule = {
    'secret_size': secret_size,
    'enc_height': 28,
    'enc_width': 28,
    'enc_in_channel': 1,
    'enc_total_epoch': 20,
    'enc_secret_only_epoch': 2,
    'enc_use_dis': False,
}

# Configure the attack scheme
ISSBA = core.ISSBA(
    dataset_name="mnist",
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=train_steg_set,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.2,      # follow the default configure in the original paper
    encoder_schedule=encoder_schedule,
    encoder=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

ISSBA.train(schedule=schedule)

# Get the poisoned dataset
poisoned_train_dataset, poisoned_test_dataset = ISSBA.get_poisoned_dataset()

print("The length of poisoned train dataset is: ", len(poisoned_train_dataset))
print("The length of poisoned test dataset is: ", len(poisoned_test_dataset))

# ===== Train backdoored model on MNIST using with MNIST (done) ===== 


# ===== Train encoder and decoder model on 224 * 224 *3 ===== 

transform_train = Compose([
    transforms.Resize((224, 224)),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((224, 224)),
    ToTensor(),
])

trainset = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)

secret_size = 20
train_data_set = np.random.random((1, 3, 224, 224)).tolist()
train_secret_set = np.random.random((1, 20)).tolist()
train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
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
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_Other_ISSBA'
}

encoder_schedule = {
    'secret_size': secret_size,
    'enc_height': 224,
    'enc_width': 224,
    'enc_in_channel': 3,
    'enc_total_epoch': 20,
    'enc_secret_only_epoch': 2,
    'enc_use_dis': False,
}

# Configure the attack scheme
ISSBA = core.ISSBA(
    dataset_name="other",
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=train_steg_set,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,      # follow the default configure in the original paper
    encoder=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

ISSBA.train_encoder_decoder(train_only=True)

# ===== Train encoder and decoder model on 224 * 224 *3 (done) ===== 
'''

# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# ========== ResNet-18_tiny_BadNets ==========
print("tiny")
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    ToPILImage(),
    Resize((224, 224)),
    # RandomHorizontalFlip(),
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
    Resize((224, 224)),
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
print("test1")

secret_size = 20

# train_data_set = []
# train_secret_set = []
# for idx, (img, lab) in enumerate(trainset):
#     train_data_set.append(img.tolist())
#     secret = np.random.binomial(1, .5, secret_size).tolist()
#     train_secret_set.append(secret)


# for idx, (img, lab) in enumerate(testset):
#     train_data_set.append(img.tolist())
#     secret = np.random.binomial(1, .5, secret_size).tolist()
#     train_secret_set.append(secret)
train_data_set = []
train_secret_set = []

for img, lab in trainset:
    train_data_set.append(img)  # 不要 .tolist()
    # print("1")
    secret = torch.from_numpy(np.random.binomial(1, 0.5, secret_size)).float()
    train_secret_set.append(secret)
    print(len(train_secret_set))

for img, lab in testset:
    train_data_set.append(img)
    # print("2")
    secret = torch.from_numpy(np.random.binomial(1, 0.5, secret_size)).float()
    train_secret_set.append(secret)
    print(len(train_secret_set))
torch.save(train_secret_set, 'SP4_train_secret_set.pth')
torch.save(train_data_set, 'SP4_train_data_set.pth')
# train_data_set = torch.load('/home/Bachi/graduate/BackdoorBox/tiny_train_data_set.pth')
# train_secret_set = torch.load('/home/Bachi/graduate/BackdoorBox/tiny_train_secret_set.pth')
print("test2")

train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)
print("test")

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 32,
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
    'experiment_name': 'ResNet-50_SP4_ISSBA'
}

encoder_schedule = {
    'secret_size': secret_size,
    'enc_height': 224,
    'enc_width': 224,
    'enc_in_channel': 3,
    'enc_total_epoch': 20,
    'enc_secret_only_epoch': 2,
    'enc_use_dis': False,
}

# Configure the attack scheme
ISSBA = core.ISSBAspec(
    dataset_name="tiny",
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=train_steg_set,
    model=core.models.ResNetspec(50, 6),
    # model=core.models.vgg11_bn(num_classes=10),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,      # follow the default configure in the original paper
    encoder_schedule=encoder_schedule,
    encoder=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)
print("ISSBA")
ISSBA.train(schedule=schedule)
poisoned_trainset, poisoned_testset = ISSBA.get_poisoned_dataset()
model = ISSBA.get_model()
torch.save(poisoned_trainset, 'ISSBA_SP4_poisoned_trainset.pth')
torch.save(poisoned_testset, 'ISSBA_SP4_poisoned_testset.pth')
print("The length of poisoned train dataset is: ", len(poisoned_trainset))
print("The length of poisoned test dataset is: ", len(poisoned_testset))