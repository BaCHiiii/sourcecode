from copy import deepcopy
import math
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10

from ..utils import Log


support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10
)


def check(dataset):
    return isinstance(dataset, support_list)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Base(object):
    """Base class for backdoor training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None, seed=0, deterministic=False):
        assert isinstance(train_dataset, support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset, support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None
        self._set_seed(seed, deterministic)

    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            # torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_model(self):
        return self.model

    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch, step, len_epoch):
        factor = (torch.tensor(self.current_schedule['schedule']) <= epoch).sum()

        lr = self.current_schedule['lr']*(self.current_schedule['gamma']**factor)

        """Warmup"""
        if 'warmup_epoch' in self.current_schedule and epoch < self.current_schedule['warmup_epoch']:
            lr = lr*float(1 + step + epoch*len_epoch)/(self.current_schedule['warmup_epoch']*len_epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # === utils: layer name 解析（支援 endswith 模糊匹配）===
    def _resolve_layers(self, model, layer_names):
        all_names = [n for n, _ in model.named_modules()]
        resolved = []
        for name in layer_names:
            if name in all_names:
                resolved.append(name); continue
            cand = [n for n in all_names if n.endswith(name)]
            if len(cand) == 1:
                resolved.append(cand[0])
            elif len(cand) == 0:
                raise ValueError(f"[align] layer '{name}' not found. Try one of: {all_names[-20:]}")
            else:
                raise ValueError(f"[align] layer '{name}' is ambiguous: {cand}")
        return resolved


    def train(self, schedule=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        log('==========Schedule parameters==========\n')
        log(str(self.current_schedule)+'\n')

        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain'], map_location='cpu'), strict=False)
            log(f"Load pretrained parameters: {self.current_schedule['pretrain']}\n")

        # Device
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========Use GPUs to train==========\n')
            CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join([str(i) for i in range(torch.cuda.device_count())]))
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')
            if CUDA_VISIBLE_DEVICES == '':
                raise ValueError(f'This machine has no visible cuda devices!')
            CUDA_SELECTED_DEVICES = self.current_schedule.get('CUDA_SELECTED_DEVICES', CUDA_VISIBLE_DEVICES)
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')
            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))
            if not (set(CUDA_SELECTED_DEVICES_LIST) <= set(CUDA_VISIBLE_DEVICES_LIST)):
                raise ValueError(f'CUDA_SELECTED_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(x) for x in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)
        else:
            device = torch.device("cpu")

        # DataLoader（(img, label, poisoned)）
        if self.current_schedule['benign_training'] is True:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
        elif self.current_schedule['benign_training'] is False:
            train_loader = DataLoader(
                self.poisoned_train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
        else:
            raise AttributeError("self.current_schedule['benign_training'] should be True or False.")

        self.model.train()

        # === 設定 ===
        layer_names   = self.current_schedule.get('align_layer_names', ['layer1.0','layer1.1','layer2.0','layer2.1','layer3.0','layer3.1','layer4.0','layer4.1'])
        alpha_poison  = self.current_schedule.get('alpha_poison', 1.0)
        poison_label_is_target = self.current_schedule.get('poison_label_is_target', True)
        target_label  = self.current_schedule.get('target_label', 1)

        # 對齊（預設關閉；要開就設 True）
        use_align     = bool(self.current_schedule.get('use_align', True))
        lam           = float(self.current_schedule.get('align_lambda', 100))

        # 解析層名 + 單趟 forward 的 hook
        resolved_layers = self._resolve_layers(self.model, layer_names)
        name_to_module  = dict(self.model.named_modules())
        def _forward_with_hooks(x_in):
            feats = {ln: None for ln in resolved_layers}
            hs = []
            def _mk(ln):
                def h(m, i, o):
                    if isinstance(o, (tuple, list)): o = o[0]
                    feats[ln] = o
                return h
            for ln in resolved_layers:
                hs.append(name_to_module[ln].register_forward_hook(_mk(ln)))
            logits = self.model(x_in)
            for h in hs: h.remove()
            return logits, feats

        iteration_total = len(train_loader)
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.current_schedule['lr'],
                                    momentum=self.current_schedule['momentum'],
                                    weight_decay=self.current_schedule['weight_decay'])

        iteration = 0
        last_time = time.time()
        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        for i in range(self.current_schedule['epochs']):
            for batch_id, batch in enumerate(train_loader):
                self.adjust_learning_rate(optimizer, i, batch_id, int(math.ceil(len(self.train_dataset) / self.current_schedule['batch_size'])))

                # 讀 batch（支援 (img, label, poisoned)）
                if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                    batch_img, batch_label, poisoned_flag = batch[0], batch[1], batch[2]
                    pmask = poisoned_flag if isinstance(poisoned_flag, torch.Tensor) else torch.as_tensor(poisoned_flag)
                else:
                    batch_img, batch_label = batch[0], batch[1]
                    pmask = None

                x = batch_img.to(device)
                y = batch_label.to(device)
                B = x.size(0)

                optimizer.zero_grad()

                # 一次 forward：拿 logits + 全批特徵
                logits_full, feats_full = _forward_with_hooks(x)

                # 用 pmask 切子集
                if pmask is not None:
                    pmask = pmask.to(device=device, dtype=torch.bool)
                    idx_p = torch.nonzero(pmask).squeeze(1)      # poisoned
                    idx_c = torch.nonzero(~pmask).squeeze(1)     # clean
                    n_p, n_c = int(idx_p.numel()), int(idx_c.numel())
                else:
                    idx_p = idx_c = None
                    n_p, n_c = 0, B  # 全部當 clean

                # ===== Cross-Entropy（比例加權）=====
                L_clean  = torch.tensor(0.0, device=device)
                L_poison = torch.tensor(0.0, device=device)

                if pmask is None:
                    L_clean = self.loss(logits_full, y)
                else:
                    if n_c > 0:
                        L_clean = self.loss(logits_full.index_select(0, idx_c),
                                            y.index_select(0, idx_c))
                    if n_p > 0:
                        if poison_label_is_target:
                            y_t = y.index_select(0, idx_p)
                        else:
                            y_t = torch.full_like(y.index_select(0, idx_p), fill_value=target_label)
                        L_poison = self.loss(logits_full.index_select(0, idx_p), y_t)

                # 子集佔比權重（等價於全批 mean CE）
                wc = (n_c / float(B)) if (pmask is not None) else 1.0
                wp = (n_p / float(B)) if (pmask is not None) else 0.0

                # ===== （可選）對齊：單趟特徵、批次均值 L1（串所有層 channel）=====
                L_align = torch.tensor(0.0, device=device)
                if use_align and (pmask is not None) and (n_c > 0) and (n_p > 0):
                    mu_c_list, mu_p_list = [], []
                    for ln in resolved_layers:
                        a = feats_full[ln]                                # [B, C_l, H, W]
                        mu_c = a.index_select(0, idx_c).mean((0, 2, 3)).detach()  # [C_l]
                        mu_p = a.index_select(0, idx_p).mean((0, 2, 3))           # [C_l]
                        mu_c_list.append(mu_c); mu_p_list.append(mu_p)
                    mu_c_all = torch.cat(mu_c_list, dim=0)               # [sum C_l]
                    mu_p_all = torch.cat(mu_p_list, dim=0)               # [sum C_l]
                    L_align  = torch.nn.functional.l1_loss(mu_p_all, mu_c_all, reduction='mean')

                # 總損失
                if use_align:
                    loss = wc * L_clean + alpha_poison * wp * L_poison + lam * L_align
                else:
                    loss = wc * L_clean + alpha_poison * wp * L_poison

                loss.backward()
                optimizer.step()

                # logging
                iteration += 1
                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    n_clean_log = (n_c if pmask is not None else B)
                    n_poisoned_log = (n_p if pmask is not None else 0)
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{iteration_total}, " \
                        f"lr:{optimizer.param_groups[0]['lr']}, " \
                        f"Lc:{float(L_clean):.4f} Lp:{float(L_poison):.4f} La:{float(L_align):.4f} " \
                        f"wc:{wc:.3f} wp:{wp:.3f} lam:{(lam if use_align else 0.0):.3f} loss:{float(loss):.4f} " \
                        f"Bc:{n_clean_log} Bp:{n_poisoned_log}\n"
                    last_time = time.time()
                    log(msg)

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # benign test
                predict_digits, labels, mean_loss = self._test(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}\n"
                log(msg)

                # poisoned test
                predict_digits, labels, mean_loss = self._test(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}\n"
                log(msg)

                self.model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                self.model.eval()
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model.train()

        self.model.eval()
        self.model = self.model.cpu()






    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None, test_loss=None):
        if model is None:
            model = self.model
        else:
            model = model

        if test_loss is None:
            test_loss = self.loss
        else:
            test_loss = test_loss

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model.eval()

            predict_digits = []
            labels = []
            losses = []
            for batch in test_loader:
                # 讀 batch（支援 (img, label, poisoned)）
                if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                    # print(len(batch))
                    batch_img, batch_label, poisoned_flag = batch[0], batch[1], batch[2]
                    pmask = poisoned_flag if isinstance(poisoned_flag, torch.Tensor) else torch.as_tensor(poisoned_flag)
                else:
                    batch_img, batch_label = batch[0], batch[1]
                    pmask = None
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                batch_img = model(batch_img)
                loss = test_loss(batch_img, batch_label)

                predict_digits.append(batch_img.cpu()) # (B, self.num_classes)
                labels.append(batch_label.cpu()) # (B)
                if loss.ndim == 0: # scalar
                    loss = torch.tensor([loss])
                losses.append(loss.cpu()) # (B) or (1)

            predict_digits = torch.cat(predict_digits, dim=0) # (N, self.num_classes)
            labels = torch.cat(labels, dim=0) # (N)
            losses = torch.cat(losses, dim=0) # (N)
            return predict_digits, labels, losses.mean().item()

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None, test_loss=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========Use GPUs to train==========\n')

            CUDA_VISIBLE_DEVICES = ''
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

            if CUDA_VISIBLE_DEVICES == '':
                raise ValueError(f'This machine has no visible cuda devices!')

            CUDA_SELECTED_DEVICES = ''
            if 'CUDA_SELECTED_DEVICES' in self.current_schedule:
                CUDA_SELECTED_DEVICES = self.current_schedule['CUDA_SELECTED_DEVICES']
            else:
                CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

            CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
            CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
            if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
                raise ValueError(f'CUDA_VISIBLE_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')

            GPU_num = len(CUDA_SELECTED_DEVICES_SET)
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)

            if GPU_num > 1:
                self.model = nn.DataParallel(self.model, device_ids=device_ids, output_device=device_ids[0])
        # Use CPU
        else:
            device = torch.device("cpu")
        from time import perf_counter
        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            start = perf_counter()
            predict_digits, labels, mean_loss = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model, test_loss)
            end = perf_counter()
            log(f'inference time: {((end-start)/10000.0)}\n')
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            start = perf_counter()
            predict_digits, labels, mean_loss = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model, test_loss)
            end = perf_counter()
            log(f'inference time: {((end-start)/10000.0)}\n')
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

        return top1_correct, top5_correct, total_num, mean_loss
