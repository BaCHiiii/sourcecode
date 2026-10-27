import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import os
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
class STRIP:
    def __init__(self, model, valset, batch_size=512, num_perturbations=100, threshold=None,use_normalize=True,dataset_name='cifar10'):
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.valset = valset
        self.batch_size = batch_size
        self.num_perturbations = num_perturbations
        self.val_loader = DataLoader(valset, batch_size=1, shuffle=True)
        self.use_normalize = use_normalize
        self.dataset_name = dataset_name
        if self.use_normalize:
            if self.dataset_name == 'cifar10':
                self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device='cuda').view(1, 3, 1, 1)
                self.std = torch.tensor([0.2470, 0.2435, 0.2616], device='cuda').view(1, 3, 1, 1)
            if self.dataset_name == 'gtsrb':
                self.mean = torch.tensor([0.3337, 0.3064, 0.3171], device='cuda').view(1, 3, 1, 1)
                self.std = torch.tensor([0.2672, 0.2564, 0.2629], device='cuda').view(1, 3, 1, 1)
            if self.dataset_name == 'cifar100':
                self.mean = torch.tensor([0.5071, 0.4867, 0.4408], device='cuda').view(1, 3, 1, 1)
                self.std = torch.tensor([0.2675, 0.2565, 0.2761], device='cuda').view(1, 3, 1, 1)
            if self.dataset_name == 'tiny':
                self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
                self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
        self.threshold = threshold if threshold is not None else self._init_threshold()
    def _denormalize(self, x):
        return x * self.std + self.mean

    def _renormalize(self, x):
        return (x - self.mean) / self.std
    def _calculate_entropy(self, probs):
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return entropy

    def _get_entropy_scores(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        entropy_scores = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.cuda()
                batch_entropy = []

                for _ in range(self.num_perturbations):
                    perturb_x, _ = next(iter(self.val_loader))
                    perturb_x = perturb_x.expand(x.size(0), -1, -1, -1).cuda()

                    if self.use_normalize:
                        x_mixed = (self._denormalize(x) + self._denormalize(perturb_x)) / 2.0
                        x_mixed = torch.clamp(x_mixed, 0.0, 1.0)
                        x_mixed = self._renormalize(x_mixed)
                    else:
                        x_mixed = (x + perturb_x) / 2.0

                    logits = self.model(x_mixed)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    entropy = self._calculate_entropy(probs)
                    batch_entropy.append(entropy)

                batch_entropy = np.stack(batch_entropy, axis=0)
                mean_entropy = np.mean(batch_entropy, axis=0)
                entropy_scores.append(mean_entropy)

        entropy_scores = np.concatenate(entropy_scores, axis=0)
        return entropy_scores


    def _init_threshold(self):
        entropy_scores = self._get_entropy_scores(self.valset)
        print(np.percentile(entropy_scores, 1))
        return np.percentile(entropy_scores, 1)  # Using 5th percentile as threshold

    def detect(self, dataset):
        entropy_scores = self._get_entropy_scores(dataset)
        predictions = entropy_scores < self.threshold
        return predictions

    def test(self, testset, poisontestset,experiment_name):
        # print(1)
        start = perf_counter()
        test_scores = self._get_entropy_scores(testset)
        # print(2)
        middle = perf_counter()
        poison_scores = self._get_entropy_scores(poisontestset)
        end = perf_counter()
        # print(3)

        y_true = np.concatenate([np.zeros(len(test_scores)), np.ones(len(poison_scores))])
        # print(y_true)
        y_scores = np.concatenate([test_scores, poison_scores])
        # print(y_scores)
        y_pred = y_scores < self.threshold

        fpr, tpr, _ = metrics.roc_curve(y_true, -y_scores)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        myf1 = metrics.f1_score(y_true, y_pred)
        times = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        work_dir = osp.join('/app/graduate/BackdoorBox/auroc/STRIP', experiment_name+"_"+times)
        if not osp.exists(work_dir):
            os.makedirs(work_dir)
        log = Log(osp.join(work_dir, 'result.txt'))
        log(f'TP: {tp}\n') 
        log(f'FP: {fp}\n')
        log(f'TN: {tn}\n')
        log(f'FN: {fn}\n')
        log(f'TPR: {tp / (tp + fn) * 100:.2f}%\n')
        log(f'FPR: {fp / (tn + fp) * 100:.2f}%\n')
        log(f'AUC: {auc:.4f}\n')
        log(f'F1 score: {myf1:.4f}\n')
        log(f'clean inference time: {((middle-start)/10000.0)}\n')
        log(f'poison inference time: {((end-middle)/10000.0)}\n')
        print(f"TPR: {tp / (tp + fn) * 100:.2f}%")
        print(f"FPR: {fp / (tn + fp) * 100:.2f}%")
        print(f"AUC: {auc:.4f}")
        print(f"F1 Score: {myf1:.4f}")
