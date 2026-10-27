# This is the test code of IBD-PSC defense.
# IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency [ICML, 2024] (https://arxiv.org/abs/2405.09786) 

import os
import pdb
import torch
from torchvision import transforms
from sklearn import metrics
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import time
from ..utils import test

from .base import Base
from time import perf_counter
class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)
class IBD_PSC(Base):
    """Identify and filter malicious testing samples (IBD-PSC).

    Args:
        model (nn.Module): The original backdoored model.
        n (int): The hyper-parameter for the number of parameter-amplified versions of the original backdoored model by scaling up of its different BN layers.
        xi (float): The hyper-parameter for the error rate.
        T (float):  The hyper-parameter for defender-specified threshold T. If PSC(x) > T , we deem it as a backdoor sample.
        scale (float): The hyper-parameter for amplyfying the parameters of selected BN layers.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.

        
    """
    def __init__(self, model, n=5, xi=0.6, T = 0.9, scale=1.5, valset=None, seed=666, deterministic=False):
        super(IBD_PSC, self).__init__(seed, deterministic)
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.n = n
        self.xi = xi
        self.T = T
        self.scale = scale
        self.valset = valset

        layer_num = self.count_BN_layers()
        sorted_indices = list(range(layer_num))
        sorted_indices = list(reversed(sorted_indices))
        self.sorted_indices = sorted_indices
        self.start_index = self.prob_start(self.scale, self.sorted_indices, valset=self.valset)

    
    def count_BN_layers(self):
        layer_num = 0
        for (name1, module1) in self.model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
            # if isinstance(module1, torch.nn.Conv2d):
                layer_num += 1
        return layer_num
    
    # test accuracy on the dataset 
    def test_acc(self, dataset, schedule):
        """Test repaired curve model on dataset

        Args:
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
        """
        model = self.model
        test(model, dataset, schedule)

    def scale_var_index(self, index_bn, scale=1.5):
        copy_model = copy.deepcopy(self.model)
        index  = -1
        for (name1, module1) in copy_model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                index += 1
                if index in index_bn:
                    module1.weight.data *= scale
                    module1.bias.data *= scale
        return copy_model  
    def prob_start(self, scale, sorted_indices, valset):
        val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
        layer_num = len(sorted_indices)
        # layer_index: k
        for layer_index in range(1, layer_num):            
            layers = sorted_indices[:layer_index]
            # print(layers)
            smodel = self.scale_var_index(layers, scale=scale)
            smodel.cuda()
            smodel.eval()
            
            total_num = 0 
            clean_wrong = 0
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    clean_img = batch[0]
                    labels = batch[1]
                    clean_img = clean_img.cuda()  # batch * channels * hight * width
                    # labels = labels.cuda()  # batch
                    clean_logits = smodel(clean_img).detach().cpu()
                    clean_pred = torch.argmax(clean_logits, dim=1)# model prediction
                    
                    clean_wrong += torch.sum(labels != clean_pred)
                    total_num += labels.shape[0]
                wrong_acc = clean_wrong / total_num
                # print(f'wrong_acc: {wrong_acc}')
                if wrong_acc > self.xi:
                    return layer_index

    def _test(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        self.model.eval()
        total_num = 0
        all_psc_score = []
        pred_correct_mask = []

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                labels = batch[1]
                total_num += labels.shape[0]
                # print('total_num:', total_num)
                imgs = imgs.cuda()  # batch * channels * hight * width
                labels = labels.cuda()  # batch
                original_pred = torch.argmax(self.model(imgs), dim=1) # model prediction
                mask = torch.eq(labels, original_pred) # only look at those samples that successfully attack the DNN
                pred_correct_mask.append(mask)

                psc_score = torch.zeros(labels.shape)
                scale_count = 0
                for layer_index in range(self.start_index, self.start_index + self.n):
                    layers = self.sorted_indices[:layer_index+1]
                    # print(f'layers: {layers}')
                    smodel = self.scale_var_index(layers, scale=self.scale)
                    scale_count += 1
                    smodel.eval()
                    logits = smodel(imgs).detach().cpu()
                    softmax_logits = torch.nn.functional.softmax(logits, dim=1)
                    psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred]

                psc_score /= scale_count
                all_psc_score.append(psc_score)
        
        all_psc_score = torch.cat(all_psc_score, dim=0)
        pred_correct_mask = torch.cat(pred_correct_mask, dim=0)
        # all_psc_score = all_psc_score[pred_correct_mask]
        return all_psc_score
    def test(self, testset, poisoned_testset,experiment_name=None):
        print(f'start_index: {self.start_index}')
        print(f'len(testset): {len(testset)}')
        print(f'len(poisoned_testset): {len(poisoned_testset)}')
        start = perf_counter()
        benign_psc = self._test(testset)
        middle = perf_counter()
        poison_psc = self._test(poisoned_testset)
        end = perf_counter()
        num_benign = benign_psc.size(0)
        num_poison = poison_psc.size(0)
        print(num_benign)
        print(num_poison)
        y_true = torch.cat((torch.zeros_like(benign_psc), torch.ones_like(poison_psc)))
        y_score = torch.cat((benign_psc, poison_psc), dim=0)
        y_pred = (y_score >= self.T)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        myf1 = metrics.f1_score(y_true, y_pred)
        times = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        work_dir = osp.join('/app/graduate/BackdoorBox/auroc/IBD_PSC', experiment_name+"_"+times)
        if not osp.exists(work_dir):
            os.makedirs(work_dir)
        log = Log(osp.join(work_dir, 'result.txt'))
        log(f'TP: {tp}\n')
        log(f'FP: {fp}\n')
        log(f'TN: {tn}\n')
        log(f'FN: {fn}\n')

        log(f'TPR: {tp / (tp + fn) * 100:.2f}\n')
        log(f'FPR: {fp / (tn + fp) * 100:.2f}\n')
        log(f'AUC: {auc:.4f}\n')
        log(f'F1 score: {myf1:.4f}\n')
        log(f'clean inference time: {((middle-start)/10000.0)}\n')
        log(f'poison inference time: {((end-middle)/10000.0)}\n')
        print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
        print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
        print("AUC: {:.4f}".format(auc))
        print(f"f1 score: {myf1}")

    def _detect(self, inputs):
        inputs = inputs.cuda()
        self.model.eval()
        self.model.cuda()
        original_pred = torch.argmax(self.model(inputs), dim=1) # model prediction

        psc_score = torch.zeros(inputs.size(0))
        scale_count = 0
        for layer_index in range(self.start_index, self.start_index + self.n):
            layers = self.sorted_indices[:layer_index+1]
            # print(f'layers: {layers}')
            smodel = self.scale_var_index(layers, scale=self.scale)
            scale_count += 1
            smodel.eval()
            logits = smodel(inputs).detach().cpu()
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred]

        psc_score /= scale_count
        
        y_pred = psc_score >= self.T
        return y_pred
    
    def detect(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                return self._detect(imgs)
