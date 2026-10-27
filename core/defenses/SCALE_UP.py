
import os
import pdb
import torch
from torchvision import transforms
from sklearn import metrics
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
import umap
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from ..utils import test

from .base import Base
import os.path as osp
import time
from time import perf_counter
class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)
class SCALE_UP(Base):
    """Identify and filter malicious testing samples (SCALE-UP).

    Args:
        model (nn.Module): The original backdoored model.
        scale_set (List):  The hyper-parameter for a set of scaling factors. Each integer n in the set scales the pixel values of an input image "x" by a factor of n.
        T (float): The hyper-parameter for defender-specified threshold T. If SPC(x) > T , we deem it as a backdoor sample.
        valset (Dataset): In data-limited scaled prediction consistency analysis, we assume that defenders have a few benign samples from each class.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
        
    """
    def __init__(self, model, scale_set=None, threshold=None, valset=None, seed=666, deterministic=False):
        super(SCALE_UP, self).__init__(seed, deterministic)
        self.model = model
        self.model.cuda()
        self.model.eval()
        if scale_set is None:
            scale_set = [3, 5, 7, 9, 11]
        if threshold is None:
            self.T = 0.5
        self.scale_set = scale_set
        # The statics for SPC values on samples from different classes
        self.valset = valset
        if self.valset:
            self.mean = None
            self.std = None
            self.init_spc_norm(self.valset)

    # test accuracy on the dataset 
    def test_acc(self, dataset, schedule):
        """Test repaired curve model on dataset

        Args:
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
        """
        model = self.model
        test(model, dataset, schedule)

    def init_spc_norm(self, valset):
        val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
        total_spc = []
        for idx, batch in enumerate(val_loader):
            clean_img = batch[0]
            labels = batch[1]
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                # If normalize:
                # scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
                # No normalize
                scaled_imgs.append(torch.clip(clean_img * scale, 0.0, 1.0))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc += scale_label == labels
            spc /= len(self.scale_set)
            total_spc.append(spc)
        total_spc = torch.cat(total_spc)

        self.mean = torch.mean(total_spc).item()
        self.std = torch.std(total_spc).item()

    def _test(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        self.model.eval()
        all_spc_score = []
        pred_correct_mask = []

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                labels = batch[1]
                imgs = imgs.cuda()  # batch * channels * hight * width
                labels = labels.cuda()  # batch
                original_pred = torch.argmax(self.model(imgs), dim=1) # model prediction
                mask = torch.eq(labels, original_pred) # only look at those samples that successfully attack the DNN
                pred_correct_mask.append(mask)

                scaled_imgs = []
                scaled_labels = []
                for scale in self.scale_set:
                    scaled_imgs.append(torch.clip(imgs * scale, 0.0, 1.0))
                    # normalized
                    # scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
                for scale_img in scaled_imgs:
                    scale_label = torch.argmax(self.model(scale_img), dim=1)
                    scaled_labels.append(scale_label)
                
                spc_score = torch.zeros(labels.shape).cuda()
                for scale_label in scaled_labels:
                    spc_score += scale_label == original_pred
                spc_score /= len(self.scale_set)

                if self.valset:
                    spc_score = (spc_score - self.mean) / self.std
                all_spc_score.append(spc_score)
        
        all_spc_score = torch.cat(all_spc_score, dim=0).cpu()
        pred_correct_mask = torch.cat(pred_correct_mask, dim=0)
        # all_spc_score = all_spc_score[pred_correct_mask]
        return all_spc_score

    def test(self, testset, poisoned_testset, experiment_name=None):
        start = perf_counter()
        benign_spc = self._test(testset)
        middle = perf_counter()
        poison_spc = self._test(poisoned_testset)
        end = perf_counter()
        num_benign = benign_spc.size(0)
        num_poison = poison_spc.size(0)
        print(f'num_benign: {num_benign}, num_poison: {num_poison}')
        y_true = torch.cat((torch.zeros_like(benign_spc), torch.ones_like(poison_spc)))
        y_score = torch.cat((benign_spc, poison_spc), dim=0)
        y_pred = (y_score >= self.T)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        myf1 = metrics.f1_score(y_true, y_pred)
        times = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        work_dir = osp.join('/app/graduate/BackdoorBox/auroc/SCALE_UP', experiment_name+"_"+times)
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

    # inputs: Tensor List , e.g., tensor([tensor_img1, tensor_img2, ..., tensor_imgn])
    def _detect(self, inputs):
        inputs = inputs.cuda()
        self.model.eval()
        self.model.cuda()
        original_pred = torch.argmax(self.model(inputs), dim=1) # model prediction
        
        scaled_imgs = []
        scaled_labels = []
        for scale in self.scale_set:
            scaled_imgs.append(torch.clip(inputs * scale, 0.0, 1.0))
            # normalized
            # scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
        for scale_img in scaled_imgs:
            scale_label = torch.argmax(self.model(scale_img), dim=1)
            scaled_labels.append(scale_label)
        
        spc_score = torch.zeros(inputs.size(0)).cuda()
        for scale_label in scaled_labels:
            spc_score += scale_label == original_pred
        spc_score /= len(self.scale_set)

        if self.valset:
            spc_score = (spc_score - self.mean) / self.std
        
        y_pred = (spc_score >= self.T)
        return y_pred

    
    def detect(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                return self._detect(imgs)

