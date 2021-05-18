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
k = similarity_k(adj_origin, labels, idx_train).cuda()

model = get_model(args.model, features.shape[1], labels.max().item()+1, args.hidden,k, args.dropout)

def train_regression(model, features,
                     labels,
                     val_labels,
                     idx_train, idx_val,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr,dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_item = loss_train.item()
        loss_train.backward()
        optimizer.step()
        print("loss: {:.3f}  train_acc: {:.3f}".format(loss_item, acc_train))
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        acc_val = accuracy(output[idx_val], val_labels)

    return model,  train_time, acc_val

def test_regression(model, features, test_labels, idx_test, adj):
    model.eval()
    test_acc_history = []
    output = model(features, adj)
    acc_test=accuracy(output[idx_test], test_labels)
    test_acc_history.append(acc_test.item())
    return acc_test,test_acc_history

if args.model == "CCDF":
    model, train_time, acc_val = train_regression(model, features, labels, labels[idx_val],
                     idx_train, idx_val, args.epochs, args.weight_decay, args.lr, args.dropout)

    acc_test, test_acc_history= test_regression(model, features, labels[idx_test], idx_test, adj)

print(" Test Accuracy: {:.4f}".format(max(test_acc_history)))
#print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
#print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
