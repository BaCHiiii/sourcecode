import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import torchvision.transforms as transforms
from imagecorruptions import corrupt
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
class TeCo:
    def __init__(self, model, batch_size=512, max_severity=5,use_normalize=False,dataset_name='cifar10'):
        self.model = model.cuda()
        self.batch_size = batch_size
        self.max_severity = max_severity

        self.corruption_names = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
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
    def _denormalize(self, x):
        return x * self.std + self.mean

    def _renormalize(self, x):
        return (x - self.mean) / self.std

    def _predict_labels(self, loader):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.cuda()
                out = self.model(x)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.extend(pred)
        return preds
    def _corrupt_dataset(self, dataset, corruption_name, severity):
        corrupted = []
        for img, label in dataset:
            img = img.cuda()
            if isinstance(img, torch.Tensor):
                if self.use_normalize:
                    img = self._denormalize(img.unsqueeze(0)).squeeze(0)
                img = transforms.ToPILImage()(img)

            np_img = np.array(img)
            if np_img.dtype != np.uint8:
                np_img = (np_img * 255).clip(0, 255).astype(np.uint8)

            corrupted_img = corrupt(np_img, corruption_name=corruption_name, severity=severity)
            corrupted_img = transforms.ToTensor()(corrupted_img)

            if self.use_normalize:
                corrupted_img = self._renormalize(corrupted_img.unsqueeze(0).cuda()).squeeze(0)

            corrupted.append((corrupted_img, label))
        return corrupted

    def _build_prediction_dict(self, dataset):
        from tqdm import tqdm
        pred_dict = {}
        # print(dataset[0][0].shape)
        base_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        base_preds = self._predict_labels(base_loader)
        # print('base_preds',len(base_preds))
        for i in range(len(base_preds)):
            pred_dict[str(i)] = {}
            pred_dict[str(i)]['original'] = base_preds[i]

        for cname in tqdm(self.corruption_names, desc="corruptions"):
            for severity in range(1, self.max_severity + 1):
                # print(len(dataset))
                corrupted_dataset = self._corrupt_dataset(dataset, cname, severity)
                # print('corrupted_dataset',len(corrupted_dataset))
                loader = DataLoader(corrupted_dataset, batch_size=self.batch_size, shuffle=False)
                preds = self._predict_labels(loader)
                # print('preds',len(preds))
                for i, p in enumerate(preds):
                    name = str(i)
                    if cname not in pred_dict[name]:
                        pred_dict[name][cname] = []
                        pred_dict[name][cname].append(pred_dict[name]['original'])
                    pred_dict[name][cname].append(p)
            # print(pred_dict)
        return pred_dict

    def evaluate_consistency(self, preds_dict):
        mads = []
        for key in preds_dict:
            original = preds_dict[key]['original']
            indexs = []
            for name in self.corruption_names:
                preds = preds_dict[key].get(name, [])
                flag = False
                for i in range(1, self.max_severity + 1):
                    if i >= len(preds):
                        break
                    if preds[i] != original:
                        indexs.append(i)
                        flag = True
                        break
                if not flag:
                    indexs.append(self.max_severity)
            indexs = np.asarray(indexs)
            mad = np.std(indexs)
            mads.append(mad)
        return np.array(mads)

    def test(self, clean_dataset, poison_dataset,experiment_name):
        start = perf_counter()
        print(1)
        poison_preds_dict = self._build_prediction_dict(poison_dataset)
        print(2)
        poison_scores = self.evaluate_consistency(poison_preds_dict)
        print(3)
        middle = perf_counter()
        clean_preds_dict = self._build_prediction_dict(clean_dataset)
        print(4)
        clean_scores = self.evaluate_consistency(clean_preds_dict)
        print(5)
        end = perf_counter()
        y_true = np.concatenate([np.zeros_like(clean_scores), np.ones_like(poison_scores)])
        y_scores = np.concatenate([clean_scores, poison_scores])

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
        auc = metrics.auc(fpr, tpr)
        best_f1 = 0
        for th in thresholds:
            pred = (y_scores > th).astype(int)
            f1 = metrics.f1_score(y_true, pred)
            best_f1 = max(best_f1, f1)

        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_scores > thresholds[np.argmax(tpr - fpr)]).ravel()

        times = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        work_dir = osp.join('/app/graduate/BackdoorBox/auroc/TeCo', experiment_name+"_"+times)
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
        log(f'F1 score: {best_f1:.4f}\n')
        log(f'clean inference time: {((middle-start)/10000.0)}\n')
        log(f'poison inference time: {((end-middle)/10000.0)}\n')
        # print(f"TPR: {tp / (tp + fn) * 100:.2f}%")
        # print(f"FPR: {fp / (tn + fp) * 100:.2f}%")
        # print(f"AUC: {auc:.4f}")
        # print(f"F1 Score: {myf1:.4f}")

