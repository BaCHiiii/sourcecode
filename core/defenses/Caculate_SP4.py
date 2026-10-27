import os
# from SP4.Spectrum2 import SpectrumTransfer
import torch
import torch.nn as nn
import time

from .base import Base
from ..utils import test
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
import os.path as osp
import time
from .log import Log
from .Spectrum_tensor import SpectrumTransfer
class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)

class Caculate_SP4(Base):
    """Caculate process.
    Args:
        train_dataset (types in support_list): forward dataset.
        test_dataset (types in support_list): testing dataset.
        model (torch.nn.Module): Network.
        layers(list): The layers to caculate
        normal_count (list(double)): the normal rate
        experiment_name (str): the name of experiment
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                train_dataset=None,
                test_dataset=None,
                poison_test_dataset=None,
                model=None,
                layers=None,
                layers_name=None,
                normal_count=None,
                experiment_name=None,
                schedule=None,
                seed=0,
                deterministic=False):
        super(Caculate_SP4, self).__init__(seed=seed, deterministic=deterministic)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.poison_test_dataset = poison_test_dataset
        self.model = model
        self.layers = layers
        self.layers_name = layers_name
        self.normal_count = normal_count
        self.experiment_name = experiment_name
        self.schedule = schedule
        self.distribution = None
        self.correct = []
        self.distribution2 = None
    def cal(self, schedule=None):
        """Caculate process.
        """

        if schedule == None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        schedule = schedule


        # Use GPU
        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            # print(
            #     f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

            if schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")
        model = self.model.to(device)
        layer_to_prune = self.layers
        # print(layer_to_prune)
        tr_loader = DataLoader(self.train_dataset, batch_size=schedule['batch_size'],
                               num_workers=schedule['num_workers'],
                               drop_last=True, pin_memory=True)
        print('caculating') 
        with torch.no_grad():
            container = {}
            def create_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    # print(layer_name)

                    #output shape [batch_size, channel, height, width]
                    # get height and width mean use torch.mean
                    if schedule['type']== 'mean':
                        # print(output.shape)
                        output = torch.mean(output, dim=(2, 3)) 
                        # print(output.shape)
                    elif schedule['type']== 'max':
                        # print(output.shape)
                        output = output.view(output.shape[0], output.shape[1], -1)

                        output = torch.max(output, dim=2).values

                    elif schedule['type'] == 'median':
                        flatten = output.view(output.shape[0], output.shape[1], -1)
                        sorted_flatten, _ = flatten.sort(dim=-1)
                        median_index = sorted_flatten.size(-1) // 2
                        output = sorted_flatten[..., median_index] 
                    # print(output.shape)
                    if layer_name not in container:
                        container[layer_name] = []  # 初始化为列表以存储多次前向传播的结果
                    container[layer_name].append(output)  # 添加输出到 container
                return forward_hook
            hooks = []
            for layer,layer_name in zip(self.layers,self.layers_name):
                # hooks.append(getattr(model,layer_name).register_forward_hook(create_forward_hook(layer_name)))
                hooks.append(layer.register_forward_hook(create_forward_hook(layer_name)))
            model.eval()
            for image, label in tr_loader:
                image=image.to(device)
                transfer = SpectrumTransfer()
                #load the weight of the spectrum transfer at current directory
                transfer.load('/app/graduate/BackdoorBox/core/attacks/weight.npz')
                batch_img = transfer.transfer_select_batch(image,interval=40,band = [0,1,2,3,4,5,6,7,8,9,10]).to(device)
                y = model(batch_img)

            for hook in hooks:
                hook.remove()

            distribution = {}
            for layer_name in self.layers_name:
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                container[layer_name] = torch.transpose(container[layer_name], 0, 1)
                mean = torch.mean(container[layer_name], dim=1)
                std = torch.std(container[layer_name], dim=1)
                distribution[layer_name] = (mean, std)

            self.distribution = distribution
            print('finish caculating')
    def detection(self,schedule=None):
        if schedule == None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        if self.test_dataset == None:
            raise AttributeError("Test set is None, please check your setting.")
        if self.distribution == None:
            raise AttributeError("Distribution is None, please check your setting.")
        
        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'])
        os.makedirs(work_dir, exist_ok=True)
        if schedule['data'] == 'clean':
            log = Log(osp.join(work_dir, 'log_clean.txt'))
        elif schedule['data'] == 'poison':
            log = Log(osp.join(work_dir, 'log_poison.txt'))
        
        # Use GPU
        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            # print(
            #     f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

            if schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        model = self.model.to(device)
        layer_to_prune = self.layers
        if schedule['data'] == 'clean':
            test_loader = DataLoader(self.test_dataset, batch_size=schedule['batch_size'],
                                num_workers=schedule['num_workers'],
                                drop_last=False, pin_memory=True)
        elif schedule['data'] == 'poison':
            test_loader = DataLoader(self.poison_test_dataset, batch_size=schedule['batch_size'],
                                    num_workers=schedule['num_workers'],
                                    drop_last=False, pin_memory=True)
        log("layer name : " + str(self.layers_name) + "\n")
        log("normal count : " + str(self.normal_count) + "\n")
        with torch.no_grad():
            container = {}
            if schedule['type'] == 'mean':
                log("Distribution setting : mean\n")
            elif schedule['type'] == 'max':
                log("Distribution setting : max\n")
            elif schedule['type'] == 'median':
                log("Distribution setting : median\n")
            def create_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    #output shape [batch_size, channel, height, width]
                    # get height and width mean use torch.mean
                    if schedule['type']== 'mean':
                        output = torch.mean(output, dim=(2, 3)) 
                    elif schedule['type']== 'max':
                        output = output.view(output.shape[0], output.shape[1], -1)
                        output = torch.max(output, dim=2).values
                    elif schedule['type']== 'median':
                        flatten = output.view(output.shape[0], output.shape[1], -1)
                        sorted_flatten, _ = flatten.sort(dim=-1)
                        median_index = sorted_flatten.size(-1) // 2
                        output = sorted_flatten[..., median_index] 
                    # print(output.shape)
                    if layer_name not in container:
                        container[layer_name] = []  
                    container[layer_name].append(output)  
                return forward_hook
            hooks = []
            for layer,layer_name in zip(self.layers,self.layers_name):
                # hooks.append(getattr(model,layer_name).register_forward_hook(create_forward_hook(layer_name)))
                hooks.append(layer.register_forward_hook(create_forward_hook(layer_name)))
            model.eval()
            self.correct = []
            for image, label in test_loader:
                image=image.to(device)
                transfer = SpectrumTransfer()
                #load the weight of the spectrum transfer at current directory
                transfer.load('/app/graduate/BackdoorBox/core/attacks/weight.npz')
                batch_img = transfer.transfer_select_batch(image,interval=40,band = [0,1,2,3,4,5,6,7,8,9,10]).to(device)
                y = model(batch_img)
                y = torch.argmax(y, dim=1)
                for i in range(len(label)):
                    if y[i] == label[i]:
                        self.correct.append(1)
                    else:
                        self.correct.append(0)

            # print(len(self.correct))
            # print(self.correct)
            for hook in hooks:
                hook.remove()
            if schedule['data'] == 'clean':
                log("===============Clean images misdetection count===============\n")
            elif schedule['data'] == 'poison':
                log("===============Backdoor images detection count===============\n")
        # caculate the score
            score = {}
            normal_count_dic = dict(zip(self.layers_name, self.normal_count))
        # if(schedule['metric']=='PDF'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                # print(distribution[1].min(), distribution[1].max())  # scale 是否有 0 或負數
                # print(np.isnan(distribution[1].cpu().numpy()).any())               # 是否包含 NaN
                # print(np.isnan(datapoint.cpu().numpy()).any())                  # input 是否有 NaN
                # print(datapoint.shape, distribution[0].cpu().numpy().shape, distribution[1].cpu().numpy().shape)
                # print(datapoint.shape)
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    safe_scale = np.clip(distribution[1].cpu().numpy(), a_min=1e-10, a_max=None)
                    pdf = norm.pdf(datapoint[i].cpu().numpy(), loc=distribution[0].cpu().numpy(), scale=safe_scale)
                    
                    # pdf = norm.pdf(datapoint[i].cpu().numpy(), loc=distribution[0].cpu().numpy(), scale=distribution[1].cpu().numpy())
                    # print(np.isnan(pdf).any())  # 是否包含 NaN
                    score[layer_name].append(pdf)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            for layer_name in self.layers_name:
                # print(len(score[layer_name]))
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True)  # 確保是 numpy 陣列
                    # print(activations)
                    # print(activations.shape)
                    count = np.sum(activations < 1e-3)# 計算小於 1e-4 的元素個數 1e-3
                    # print(count)
                    # print(normal_count_dic[layer_name])
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            # find wrong[i]==1 and checkpoisonornot[i]==1
            log("PDF threshold : %f\n" % 1e-3)
            log("PDF detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                log("model predict correct but misdetection : %d\n" % c1)
                log("model predict wrong and misdetection : %d\n" % c2)
                log("model misdetection rate : %f\n"%(c1/len(self.correct)))
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                    if checkpoisonornot[i] == 0 and self.correct[i] == 1:
                        c3 += 1
                log("model predict correct and detection : %d\n" % c1)
                log("model predict wrong but detection : %d\n" % c2)
                log("model predict correct but misdetection : %d\n" % c3)
                log("poison data detection rate : %f\n"%(c1/(c1+c3)))
            score = {}
        # elif(schedule['metric']=='CDF'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                # container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    cdf = norm.cdf(datapoint[i].cpu().numpy(), loc=distribution[0].cpu().numpy(), scale=distribution[1].cpu().numpy())
                    score[layer_name].append(cdf)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            for layer_name in self.layers_name:
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True)  # 確保是 numpy 陣列
                    count = np.sum((activations < 0.00005) | (activations > 0.99995))# 計算小於 1e-4 的元素個數 0.00003
                    # print(count)
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            log("CDF threshold : %f,%f\n" % (0.00005, 0.99995))
            log("CDF detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                log("model predict correct but misdetection : %d\n" % c1)
                log("model predict wrong and misdetection : %d\n" % c2)
                log("model misdetection rate : %f\n"%(c1/len(self.correct)))
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                    if checkpoisonornot[i] == 0 and self.correct[i] == 1:
                        c3 += 1
                log("model predict correct and detection : %d\n" % c1)
                log("model predict wrong but detection : %d\n" % c2)
                log("model predict correct but misdetection : %d\n" % c3)
                log("poison data detection rate : %f\n"%(c1/(c1+c3)))
            score = {}
        # elif(schedule['metric']=='Z-score'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                # container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    z_sc = (datapoint[i].cpu().numpy()-distribution[0].cpu().numpy())/distribution[1].cpu().numpy()
                    score[layer_name].append(z_sc)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            # zip self.normal_count and self.layers_name to dic
            
            for layer_name in self.layers_name:
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True)  # 確保是 numpy 陣列
                    
                    count = np.sum(activations>4)# 計算小於 1e-4 的元素個數 3
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            log("Z-score threshold : %f\n" % 4)
            log("Z-score detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                log("model predict correct but misdetection : %d\n" % c1)
                log("model predict wrong and misdetection : %d\n" % c2)
                log("model misdetection rate : %f\n"%(c1/len(self.correct)))
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                    if checkpoisonornot[i] == 0 and self.correct[i] == 1:
                        c3 += 1
                log("model predict correct and detection : %d\n" % c1)
                log("model predict wrong but detection : %d\n" % c2)
                log("model predict correct but misdetection : %d\n" % c3)
                log("poison data detection rate : %f\n"%(c1/(c1+c3)))
            log("==========================================================\n")
    # for draw the distribution
    def cal1(self, schedule=None):
        """Caculate process.
        """

        if schedule == None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        schedule = schedule


        # Use GPU
        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            # print(
            #     f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

            if schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")
        model = self.model.to(device)
        layer_to_prune = self.layers
        # print(layer_to_prune)
        tr_loader = DataLoader(self.test_dataset, batch_size=schedule['batch_size'],
                               num_workers=schedule['num_workers'],
                               drop_last=True, pin_memory=True)
        print('caculating') 
        with torch.no_grad():
            container = {}
            def create_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    # print(layer_name)

                    #output shape [batch_size, channel, height, width]
                    # get height and width mean use torch.mean
                    if schedule['type']== 'mean':
                        output = torch.mean(output, dim=(2, 3)) 
                    elif schedule['type']== 'max':
                        output = output.view(output.shape[0], output.shape[1], -1)
                        output = torch.max(output, dim=2).values
                    elif schedule['type'] == 'median':
                        flatten = output.view(output.shape[0], output.shape[1], -1)
                        flatten = flatten.cpu()
                        output = torch.median(flatten, dim=-1).values
                    # print(output.shape)
                    if layer_name not in container:
                        container[layer_name] = []  # 初始化为列表以存储多次前向传播的结果
                    container[layer_name].append(output)  # 添加输出到 container
                return forward_hook
            hooks = []
            for layer,layer_name in zip(self.layers,self.layers_name):
                # hooks.append(getattr(model,layer_name).register_forward_hook(create_forward_hook(layer_name)))
                hooks.append(layer.register_forward_hook(create_forward_hook(layer_name)))

            model.eval()
            for image, _ in tr_loader:
                image=image.to(device)
                transfer = SpectrumTransfer()
                #load the weight of the spectrum transfer at current directory
                transfer.load('/app/graduate/BackdoorBox/core/attacks/weight.npz')
                batch_img = transfer.transfer_select_batch(image,interval=40,band = [0,1,2,3,4,5,6,7,8,9,10]).to(device)
                model(batch_img)
            for hook in hooks:
                hook.remove()
            for layer_name in self.layers_name:
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                container[layer_name] = torch.transpose(container[layer_name], 0, 1)
            print('finish caculating')
            self.distribution = container
            print('poison caculating')
            # test poison data
            container2 = {}
            def create_forward_hook2(layer_name):
                def forward_hook(module, input, output):
                    # print(layer_name)
                    #output shape [batch_size, channel, height, width]
                    # get height and width mean use torch.mean
                    if schedule['type']== 'mean':
                        output = torch.mean(output, dim=(2, 3)) 
                    elif schedule['type']== 'max':
                        output = output.view(output.shape[0], output.shape[1], -1)
                        output = torch.max(output, dim=2).values
                    elif schedule['type'] == 'median':
                        flatten = output.view(output.shape[0], output.shape[1], -1)
                        flatten = flatten.cpu()
                        output = torch.median(flatten, dim=-1).values
                    # print(output.shape)
                    if layer_name not in container2:
                        container2[layer_name] = []
                    container2[layer_name].append(output)
                return forward_hook
            hooks2 = []
            for layer,layer_name in zip(self.layers,self.layers_name):
                hooks2.append(layer.register_forward_hook(create_forward_hook2(layer_name)))
            tr_loader_poison = DataLoader(self.poison_test_dataset, batch_size=schedule['batch_size'],
                                 num_workers=schedule['num_workers'],
                               drop_last=True, pin_memory=True)
            model.eval()
            for image, _ in tr_loader_poison:
                image = image.to(device)
                transfer = SpectrumTransfer()
                #load the weight of the spectrum transfer at current directory
                transfer.load('/app/graduate/BackdoorBox/core/attacks/weight.npz')
                batch_img = transfer.transfer_select_batch(image,interval=40,band = [0,1,2,3,4,5,6,7,8,9,10]).to(device)
                # print(image.shape)
                label=model(batch_img)
            for hook in hooks2:
                hook.remove()
            for layer_name in self.layers_name:
                container2[layer_name] = torch.cat(container2[layer_name], dim=0)
                container2[layer_name] = torch.transpose(container2[layer_name], 0, 1)
            # memory clean
            self.distribution2 = container2
            print('finish caculating')
            # self.distribution2 = distribution2
            os.makedirs(f'./distribution/{self.experiment_name}', exist_ok=True)
            # draw distribution
            for layer_name in self.layers_name:
                for i in range(0, container[layer_name].shape[0]):
                    data = container[layer_name][i].cpu().numpy()
                    data2 = container2[layer_name][i].cpu().numpy()
                    # print("data")
                    # print(type(data))
                    # print("data2")
                    # print(type(data2))
                    plt.hist(data, bins=30, alpha=0.5, color='gray', edgecolor='black', label='clean data histogram')
                    plt.hist(data2, bins=30, alpha=0.5, color='red', edgecolor='black', label='poison data histogram')
                    plt.title('Data Distribution')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.savefig(f'./distribution/{self.experiment_name}/{layer_name}_{i}.png')
                    print(f'./distribution/{self.experiment_name}/{layer_name}_{i}.png')
                    plt.clf()
                    plt.close()
                    # time.sleep(0.01)
    def detection1(self,schedule=None):
        if schedule == None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        if self.test_dataset == None:
            raise AttributeError("Test set is None, please check your setting.")
        if self.distribution == None:
            raise AttributeError("Distribution is None, please check your setting.")
        
        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'])
        os.makedirs(work_dir, exist_ok=True)
        if schedule['data'] == 'clean':
            log = Log(osp.join(work_dir, 'log_clean.txt'))
        elif schedule['data'] == 'poison':
            log = Log(osp.join(work_dir, 'log_poison.txt'))
        
        # Use GPU
        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            # print(
            #     f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

            if schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        model = self.model.to(device)
        layer_to_prune = self.layers
        if schedule['data'] == 'clean':
            test_loader = DataLoader(self.test_dataset, batch_size=schedule['batch_size'],
                                num_workers=schedule['num_workers'],
                                drop_last=False, pin_memory=True)
        elif schedule['data'] == 'poison':
            test_loader = DataLoader(self.poison_test_dataset, batch_size=schedule['batch_size'],
                                    num_workers=schedule['num_workers'],
                                    drop_last=False, pin_memory=True)
        log("layer name : " + str(self.layers_name) + "\n")
        log("normal count : " + str(self.normal_count) + "\n")
        with torch.no_grad():
            container = {}
            if schedule['type'] == 'mean':
                log("Distribution setting : mean\n")
            elif schedule['type'] == 'max':
                log("Distribution setting : max\n")
            elif schedule['type'] == 'median':
                log("Distribution setting : median\n")
            def create_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    #output shape [batch_size, channel, height, width]
                    # get height and width mean use torch.mean
                    if schedule['type']== 'mean':
                        output = torch.mean(output, dim=(2, 3)) 
                    elif schedule['type']== 'max':
                        output = output.view(output.shape[0], output.shape[1], -1)
                        output = torch.max(output, dim=2).values
                    elif schedule['type']== 'median':
                        flatten = output.view(output.shape[0], output.shape[1], -1)
                        sorted_flatten, _ = flatten.sort(dim=-1)
                        median_index = sorted_flatten.size(-1) // 2
                        output = sorted_flatten[..., median_index] 
                    # print(output.shape)
                    if layer_name not in container:
                        container[layer_name] = []  
                    container[layer_name].append(output)  
                return forward_hook
            hooks = []
            for layer,layer_name in zip(self.layers,self.layers_name):
                # hooks.append(getattr(model,layer_name).register_forward_hook(create_forward_hook(layer_name)))
                hooks.append(layer.register_forward_hook(create_forward_hook(layer_name)))
            model.eval()
            self.correct = []
            for image, label in test_loader:
                image=image.to(device)
                transfer = SpectrumTransfer()
                #load the weight of the spectrum transfer at current directory
                transfer.load('/app/graduate/BackdoorBox/core/attacks/weight.npz')
                batch_img = transfer.transfer_select_batch(image,interval=40,band = [0,1,2,3,4,5,6,7,8,9,10]).to(device)
                y = model(batch_img)

                y = torch.argmax(y, dim=1)
                for i in range(len(label)):
                    if y[i] == label[i]:
                        self.correct.append(1)
                    else:
                        self.correct.append(0)

            print(len(self.correct))
            # print(self.correct)
            for hook in hooks:
                hook.remove()
            if schedule['data'] == 'clean':
                log("===============Clean images misdetection count===============\n")
            elif schedule['data'] == 'poison':
                log("===============Backdoor images detection count===============\n")
        # caculate the score
            score = {}
            normal_count_dic = dict(zip(self.layers_name, self.normal_count))
        if(schedule['metric']=='PDF'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    safe_scale = np.clip(distribution[1].cpu().numpy(), a_min=1e-10, a_max=None)
                    pdf = norm.pdf(datapoint[i].cpu().numpy(), loc=distribution[0].cpu().numpy(), scale=safe_scale)
                    score[layer_name].append(pdf)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            for layer_name in self.layers_name:
                # print(len(score[layer_name]))
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True).astype(np.float32)
                    # print("dtype:", activations.dtype)
                    # print("sample:", activations.ravel()[:5])
                    # print("shape:", activations.shape)
                    # print("min:", activations.min(), "max:", activations.max())
                    print("start sum test")
                    count = np.sum(activations < 1e-4)
                    # print("count =", count)
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            # find wrong[i]==1 and checkpoisonornot[i]==1
            log("PDF threshold : %f\n" % 1e-4)
            log("PDF detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                log("model predict correct but misdetection : %d\n" % c1)
                log("model predict wrong and misdetection : %d\n" % c2)
                log("model misdetection rate : %f\n"%(c1/len(self.correct)))
                return c1/len(self.correct)
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                    if checkpoisonornot[i] == 0 and self.correct[i] == 1:
                        c3 += 1
                log("model predict correct and detection : %d\n" % c1)
                log("model predict wrong but detection : %d\n" % c2)
                log("model predict correct but misdetection : %d\n" % c3)
                log("poison data detection rate : %f\n"%(c1/(c1+c3)))
                return c1/(c1+c3)
            score = {}
        elif(schedule['metric']=='CDF'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    cdf = norm.cdf(datapoint[i].cpu().numpy(), loc=distribution[0].cpu().numpy(), scale=distribution[1].cpu().numpy())
                    score[layer_name].append(cdf)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            for layer_name in self.layers_name:
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True) 
                    print("start sum test") # 確保是 numpy 陣列
                    count = np.sum((activations < 0.00001) | (activations > 0.99999))# 計算小於 1e-4 的元素個數 0.00003
                    # print(count)
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            log("CDF threshold : %f,%f\n" % (0.00001, 0.99999))
            log("CDF detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                log("model predict correct but misdetection : %d\n" % c1)
                log("model predict wrong and misdetection : %d\n" % c2)
                log("model misdetection rate : %f\n"%(c1/len(self.correct)))
                return c1/len(self.correct)
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                    if checkpoisonornot[i] == 0 and self.correct[i] == 1:
                        c3 += 1
                log("model predict correct and detection : %d\n" % c1)
                log("model predict wrong but detection : %d\n" % c2)
                log("model predict correct but misdetection : %d\n" % c3)
                log("poison data detection rate : %f\n"%(c1/(c1+c3)))
                return c1/(c1+c3)
        elif(schedule['metric']=='Z-score'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    z_sc = (datapoint[i].cpu().numpy()-distribution[0].cpu().numpy())/distribution[1].cpu().numpy()
                    score[layer_name].append(z_sc)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            # zip self.normal_count and self.layers_name to dic
            
            for layer_name in self.layers_name:
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True)  # 確保是 numpy 陣列
                    print("start sum test") # 確保是 numpy 陣列
                    count = np.sum(activations>5)# 計算小於 1e-4 的元素個數 3
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            log("Z-score threshold : %f\n" % 5)
            log("Z-score detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                log("model predict correct but misdetection : %d\n" % c1)
                log("model predict wrong and misdetection : %d\n" % c2)
                log("model misdetection rate : %f\n"%(c1/len(self.correct)))
                return c1/len(self.correct)
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 and self.correct[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 1 and self.correct[i] == 0:
                        c2 += 1
                    if checkpoisonornot[i] == 0 and self.correct[i] == 1:
                        c3 += 1
                log("model predict correct and detection : %d\n" % c1)
                log("model predict wrong but detection : %d\n" % c2)
                log("model predict correct but misdetection : %d\n" % c3)
                log("poison data detection rate : %f\n"%(c1/(c1+c3)))
                return c1/(c1+c3)
            log("==========================================================\n")
    def detection2(self,schedule=None):
        if schedule == None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        if self.test_dataset == None:
            raise AttributeError("Test set is None, please check your setting.")
        if self.distribution == None:
            raise AttributeError("Distribution is None, please check your setting.")
        
        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'])
        os.makedirs(work_dir, exist_ok=True)
        if schedule['data'] == 'clean':
            log = Log(osp.join(work_dir, 'log_clean.txt'))
        elif schedule['data'] == 'poison':
            log = Log(osp.join(work_dir, 'log_poison.txt'))
        
        # Use GPU
        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            # print(
            #     f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

            if schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        model = self.model.to(device)
        layer_to_prune = self.layers
        if schedule['data'] == 'clean':
            test_loader = DataLoader(self.test_dataset, batch_size=schedule['batch_size'],
                                num_workers=schedule['num_workers'],
                                drop_last=False, pin_memory=True)
        elif schedule['data'] == 'poison':
            test_loader = DataLoader(self.poison_test_dataset, batch_size=schedule['batch_size'],
                                    num_workers=schedule['num_workers'],
                                    drop_last=False, pin_memory=True)
        log("layer name : " + str(self.layers_name) + "\n")
        log("normal count : " + str(self.normal_count) + "\n")
        with torch.no_grad():
            container = {}
            if schedule['type'] == 'mean':
                log("Distribution setting : mean\n")
            elif schedule['type'] == 'max':
                log("Distribution setting : max\n")
            elif schedule['type'] == 'median':
                log("Distribution setting : median\n")
            def create_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    #output shape [batch_size, channel, height, width]
                    # get height and width mean use torch.mean
                    if schedule['type']== 'mean':
                        output = torch.mean(output, dim=(2, 3)) 
                    elif schedule['type']== 'max':
                        output = output.view(output.shape[0], output.shape[1], -1)
                        output = torch.max(output, dim=2).values
                    elif schedule['type']== 'median':
                        flatten = output.view(output.shape[0], output.shape[1], -1)
                        sorted_flatten, _ = flatten.sort(dim=-1)
                        median_index = sorted_flatten.size(-1) // 2
                        output = sorted_flatten[..., median_index] 
                    # print(output.shape)
                    if layer_name not in container:
                        container[layer_name] = []  
                    container[layer_name].append(output)  
                return forward_hook
            hooks = []
            for layer,layer_name in zip(self.layers,self.layers_name):
                # hooks.append(getattr(model,layer_name).register_forward_hook(create_forward_hook(layer_name)))
                hooks.append(layer.register_forward_hook(create_forward_hook(layer_name)))
            model.eval()
            self.correct = []
            for image, label in test_loader:
                image=image.to(device)
                transfer = SpectrumTransfer()
                #load the weight of the spectrum transfer at current directory
                transfer.load('/app/graduate/BackdoorBox/core/attacks/weight.npz')
                batch_img = transfer.transfer_select_batch(image,interval=40,band = [0,1,2,3,4,5,6,7,8,9,10]).to(device)
                y = model(batch_img)

                y = torch.argmax(y, dim=1)
                for i in range(len(label)):
                    if y[i] == label[i]:
                        self.correct.append(1)
                    else:
                        self.correct.append(0)

            # print(len(self.correct))
            # print(self.correct)
            for hook in hooks:
                hook.remove()
            if schedule['data'] == 'clean':
                log("===============Clean images misdetection count===============\n")
            elif schedule['data'] == 'poison':
                log("===============Backdoor images detection count===============\n")
        # caculate the score
            score = {}
            normal_count_dic = dict(zip(self.layers_name, self.normal_count))
        if(schedule['metric']=='PDF'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    safe_scale = np.clip(distribution[1].cpu().numpy(), a_min=1e-10, a_max=None)
                    pdf = norm.pdf(datapoint[i].cpu().numpy(), loc=distribution[0].cpu().numpy(), scale=safe_scale)
                    score[layer_name].append(pdf)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            for layer_name in self.layers_name:
                # print(len(score[layer_name]))
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True).astype(np.float32)

                    # print("start sum test")
                    count = np.sum(activations < 1e-3)
                    # print("count =", count)
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            # find wrong[i]==1 and checkpoisonornot[i]==1
            log("PDF threshold : %f\n" % 1e-3)
            log("PDF detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 0:
                        c2 += 1
                log("FP : %d\n" % c1)
                log("TN : %d\n" % c2)
                return c1, c2
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 0:
                        c2 += 1
                log("TP : %d\n" % c1)
                log("FN : %d\n" % c2)
                return c1, c2
            score = {}
        elif(schedule['metric']=='CDF'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    cdf = norm.cdf(datapoint[i].cpu().numpy(), loc=distribution[0].cpu().numpy(), scale=distribution[1].cpu().numpy())
                    score[layer_name].append(cdf)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            for layer_name in self.layers_name:
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True) 
                    # print("start sum test") # 確保是 numpy 陣列
                    count = np.sum((activations < 0.00001) | (activations > 0.99999))# 計算小於 1e-4 的元素個數 0.00003
                    # print(count)
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            log("CDF threshold : %f,%f\n" % (0.00001, 0.99999))
            log("CDF detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1: 
                        c1 += 1
                    if checkpoisonornot[i] == 0:
                        c2 += 1
                log("FP : %d\n" % c1)
                log("TN : %d\n" % c2)
                return c1, c2
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 0 :
                        c2 += 1
                log("TP : %d\n" % c1)
                log("FN : %d\n" % c2)
                return c1, c2
        elif(schedule['metric']=='Z-score'):
            for layer_name in self.layers_name:
                score[layer_name] = []
                container[layer_name] = torch.cat(container[layer_name], dim=0)
                distribution = self.distribution[layer_name]
                datapoint = container[layer_name]
                #calculate pdf score
                for i in range(0, datapoint.shape[0]):
                    z_sc = (datapoint[i].cpu().numpy()-distribution[0].cpu().numpy())/distribution[1].cpu().numpy()
                    score[layer_name].append(z_sc)
                # print(len(score[layer_name]))
            checkpoisonornot = np.zeros(len(score[layer_name]))
            # zip self.normal_count and self.layers_name to dic
            
            for layer_name in self.layers_name:
                for i in range(len(score[layer_name])):
                    activations = np.array(score[layer_name][i], copy=True)  # 確保是 numpy 陣列
                    # print("start sum test") # 確保是 numpy 陣列
                    count = np.sum(activations>5)# 計算小於 1e-4 的元素個數 3
                    if count > normal_count_dic[layer_name]:
                        checkpoisonornot[i] = 1
            c = np.sum(checkpoisonornot==1)
            log("Z-score threshold : %f\n" % 5)
            log("Z-score detect count : %d\n" % c)
            c1 =0
            c2 =0
            c3 =0
            if schedule['data'] == 'clean':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1 :
                        c1 += 1
                    if checkpoisonornot[i] == 0:
                        c2 += 1
                log("FP : %d\n" % c1)
                log("TN : %d\n" % c2)
                return c1, c2
            elif schedule['data'] == 'poison':
                for i in range(len(checkpoisonornot)):
                    if checkpoisonornot[i] == 1:
                        c1 += 1
                    if checkpoisonornot[i] == 0 :
                        c2 += 1
                log("TP : %d\n" % c1)
                log("FN : %d\n" % c2)
                return c1, c2
            log("==========================================================\n")