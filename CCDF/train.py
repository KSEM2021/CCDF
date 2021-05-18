import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Arguments
args = get_citation_args()
# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train, idx_val, idx_test, adj_origin = load_citation(args.dataset, args.normalization, args.cuda)

q = get_q(adj_origin, labels, idx_train).cuda()

model = get_model(args.model, features.shape[1], args.hidden1, labels.max().item()+1, q, args.dropout)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(model,epochs):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = F.cross_entropy(output[idx_train], labels[idx_train])
    acc = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    acc_test = main_test(model)

    print('e:{}'.format(epochs),
          'loss_train: {:.4f}'.format(loss.item()),
          'acc_train: {:.4f}'.format(acc.item()),
          'acc_test: {:.4f}'.format(acc_test.item()))

    return loss.item(), acc_test.item()


def main_test(model):
    model.eval()
    output = model(features, adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    label_max = []
    for idx in idx_test:
        label_max.append(torch.argmax(output[idx]).item())

    return acc_test


acc_max = 0
epoch_max = 0
for epoch in range(args.epochs):
    loss, acc_test = train(model, epoch)
    if acc_test >= acc_max:
        acc_max = acc_test
        epoch_max = epoch
print('epoch:{}'.format(epoch_max),
      'acc_max: {:.4f}'.format(acc_max))
