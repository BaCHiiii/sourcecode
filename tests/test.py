#check pytorch resnet18 architecture
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
pattern = torch.zeros((224,224), dtype=torch.uint8)
pattern[:5,:5] = 255
pattern[:5,-5:] = 255
pattern[-5:,:5] = 255
pattern[-5:,-5:] = 255
print(pattern[:5,:5])
print(pattern[:5,-5:])
print(pattern[-5:,:5])
print(pattern[-5:,-5:])
print(pattern[10:20,10:20])
weight = torch.zeros((224,224), dtype=torch.float32)
weight[:5,:5] = 1.0
weight[:5,-5:] = 1.0
weight[-5:,:5] = 1.0
weight[-5:,-5:] = 1.0
print(weight[:5,:5])
print(weight[:5,-5:])
print(weight[-5:,:5])       
print(weight[-5:,-5:])
print(weight[10:20,10:20])