'''
This is the test code of Pruning.
'''

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


from copy import deepcopy
from torchvision.datasets import DatasetFolder
import cv2
from core.utils import test

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '0'
batch_size = 128
num_workers = 4


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



def test_pruning(model, p, trainset, testset, poisoned_testset, layer, prune_rate, y_target,experiment_name):
    num1 = int(len(trainset) * p)
    num2 = int(len(trainset) - num1)
    pretrainset1, pretrainset2 = random_split(trainset, [num1, num2])
    mytestset = deepcopy(testset)
    mypoisoned_testset = deepcopy(poisoned_testset)
    pruning = core.Pruningspec(
        train_dataset=pretrainset1,
        test_dataset=mytestset,
        model=model,
        layer=layer,
        prune_rate=prune_rate,
        seed=global_seed,
        deterministic=deterministic
    )
    print("without defense")
    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'batch_size': 64,
        'num_workers': 0,
        'metric': 'BA',

        'save_dir': 'experiments',
        'experiment_name': experiment_name + '_clean'
    }
    pruning.test(test_schedule)
    # pruning.test_dataset = mypoisoned_testset
    # test_schedule2 = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': '0',
    #     'GPU_num': 1,

    #     'batch_size': 64,
    #     'num_workers': 0,
    #     'metric': 'ASR_NoTarget',
    #     'y_target': y_target,

    #     'save_dir': 'experiments',
    #     'experiment_name': experiment_name + '_poisoned_no_defense'
    # }
    # pruning.test(test_schedule2)

    print("with defense")
    pruning.test_dataset = mytestset
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'batch_size': 64,
        'num_workers': 0,
    }
    pruning.repair(schedule)

    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'batch_size': 64,
        'num_workers': 0,
        'metric': 'BA',

        'save_dir': 'experiments',
        'experiment_name': experiment_name + '_clean'
    }
    pruning.test(test_schedule)
    # pruning.test_dataset = mypoisoned_testset
    # test_schedule2 = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': '0',
    #     'GPU_num': 1,

    #     'batch_size': 64,
    #     'num_workers': 0,
    #     'metric': 'ASR_NoTarget',
    #     'y_target': y_target,

    #     'save_dir': 'experiments',
    #     'experiment_name': experiment_name + '_poisoned'
    # }
    # pruning.test(test_schedule2)


    # # change the set
    # pruning.test_dataset = mypoisoned_testset

    # test_schedule2 = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': '1',
    #     'GPU_num': 1,

    #     'batch_size': 128,
    #     'num_workers': 4,
    #     'metric': 'ASR_NoTarget',
    #     'y_target': y_target,

    #     'save_dir': 'experiments',
    #     'experiment_name': 'pruning_CIFAR10_BadNets'
    # }
    # pruning.test(test_schedule2)

    # # test the get_model
    repair_model = pruning.get_model()
    print(type(repair_model))
    #print layer name
    for name, module in repair_model.named_modules():
        print(name)
    torch.save(repair_model.state_dict(),'model_Noisepruning2.pth')
    # test(repair_model, mypoisoned_testset, test_schedule2)

    del pruning



datasets_root_dir = './data'

# ========== ResNet-50_SP4_Badnets_Pruning ==========
print("badnets")
model = core.models.ResNetdct(50,6)
# model = torch.load('/home/Bachi/graduate/BackdoorBox/model_DCTpruning.pth')
print(type(model))
attack_name = 'BadNets'
# model_path = '/home/Bachi/graduate/BackdoorBox/experiments/dct/model_DCT3.pth'
model_path = '/home/Bachi/graduate/BackdoorBox/experiments/noise/model_lastest.pth'

pattern = torch.zeros((224, 224), dtype=torch.uint8)
pattern[-16:, -16:] = 255
weight = torch.zeros((224, 224), dtype=torch.float32)
weight[-16:, -16:] = 1.0
transform_train = Compose([
    # ToPILImage(),
    # Resize((224, 224)),
    ToTensor()
])
trainset = DatasetFolder(
    root='./data/All_datasets/train',  # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    # ToPILImage(),
    # Resize((224, 224)),
    ToTensor()
])
testset = DatasetFolder(
    root='./data/All_datasets/test',  # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('jpg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
attack = core.BadNetsspec(
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
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()


model.load_state_dict(torch.load(model_path))
# model = torch.load(model_path)
#choose the proportion of data to calculation
p=0.8
#set the layer to prune
#print layer name
for name, module in model.named_modules():
    print(name)
layer='layer4'
#set the prune rate
prune_rate=0.95
y_target=1
test_pruning(model, p, trainset, testset, poisoned_testset, layer, prune_rate, y_target,experiment_name='Pruning_ResNet-50_SP4_noise')