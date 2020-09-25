import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim                                                      

from tqdm import tqdm
import numpy as np
import pandas as pd

from model import FaceMobileNet
from model.resnet import ResIRSE
from model.metric import ArcFace, CosFace
from model.loss import FocalLoss
from dataset import load_data
from config import config as conf


def extract_embeddings(dataloader, net):
    cuda = torch.cuda.is_available()
    with torch.no_grad():
        net = net.module.to(device)
        #net.eval()
        embeddings = np.zeros((len(dataloader.dataset), 512))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = net.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels



# Data Setup
dataloader, class_num = load_data(conf, training=True)
embedding_size = conf.embedding_size
device = conf.device

# Network Setup
if conf.backbone == 'resnet':
    net = ResIRSE(embedding_size, conf.drop_ratio).to(device)
else:
    net = FaceMobileNet(embedding_size).to(device)

if conf.metric == 'arcface':
    metric = ArcFace(embedding_size, class_num).to(device)
else:
    metric = CosFace(embedding_size, class_num).to(device)

net = nn.DataParallel(net)
metric = nn.DataParallel(metric)

# Training Setup
if conf.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
else:
    criterion = nn.CrossEntropyLoss()

if conf.optimizer == 'sgd':
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)
else:
    optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],
                            lr=conf.lr, weight_decay=conf.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

# Checkpoints Setup
checkpoints = conf.checkpoints
os.makedirs(checkpoints, exist_ok=True)

if conf.restore:
    weights_path = osp.join(checkpoints, conf.restore_model)
    net.load_state_dict(torch.load(weights_path, map_location=device))

# Start training
net.train()
min_loss = 10000
for e in range(conf.epoch):
    for data, labels in tqdm(dataloader, desc=f"Epoch {e}/{conf.epoch}",
                             ascii=True, total=len(dataloader)):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        embeddings = net(data)
        thetas = metric(embeddings, labels)
        loss = criterion(thetas, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {e}/{conf.epoch}, Loss: {loss}")

    backbone_path = osp.join(checkpoints, f"arcfacemodel.pth")
    if loss < min_loss:
	       min_loss = loss
	       torch.save(net.state_dict(),backbone_path)
    scheduler.step()
    

train_embeddings_cl, train_labels_cl = extract_embeddings(dataloader, net)
train_embeddings_cl_df = pd.DataFrame(train_embeddings_cl)
train_embeddings_cl_df.to_csv("embedding_space.csv",index=False)
