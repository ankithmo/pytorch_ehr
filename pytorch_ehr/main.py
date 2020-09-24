
# -*- coding: utf-8 -*-

"""
ZhiGroup/pytorch_ehr:
This Class is mainly for the creation of the EHR patients' visits embedding
which is the key input for all the deep learning models in this Repo
@authors: Lrasmy , Jzhu  @ DeguiZhi Lab - UTHealth SBMI
Last revised Feb 20 2020

ankithmo/pytorch_ehr:
@authors: ankithmo @ Sze-chuan Suen Lab - USC ISE
Last revised Sep 23 2020
"""

from __future__ import print_function, division
from io import open
import string
import re
import random

import os
import os.path as osp
import sys
import argparse
import time
import math

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

try:
    import cPickle as pickle
except:
    import pickle

import models
from EHRDataloader import EHRdataFromPickles, EHRdataloader  
import utils as ut 
from EHREmb import EHREmbeddings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(root_dir='data', files=['toy.train'], test_ratio=0.2, valid_ratio=0.1,
        batch_size=128, which_model='DRNN', cell_type='GRU', input_size=[15817],
        embed_dim=128, hidden_size=128, dropout_r=0.1, n_layers=1, bii=False,
        time=False, preTrainEmb='', output_dir='models', model_prefix='toy.train',
        model_customed='', lr=10**-2, L2=10**-4, eps=10**-8, num_epochs=100,
        patience=5, optimizer='adam', seed=0, use_cuda=False):

    """
        Predictive Analytics on EHR
    
        Args:
            - root_dir: Path to the folders with pickled file(s)
            - files: Name(s) of pickled file(s), separated by space. so the argument will be saved as a list 
                    If list of 1: data will be first split into train, validation and test, then 3 dataloaders will be created.
                    If list of 3: 3 dataloaders will be created from 3 files directly. 
                    Files must be in the following order: training, validation and test.
            - test_ratio: Test data size 
                Default: 0.2
            - valid_ratio: Validation data size 
                Default: 0.1
            - batch_size: Batch size for training, validation or test 
                Default: 128
            - which_model: Choose from {"RNN", "DRNN", "QRNN", "TLSTM", "LR", "RETAIN"}
            - cell_type: For RNN based models, choose from {"RNN", "GRU", "LSTM"}
            - input_size: Input dimension(s) separated in space the output will be a list, decide which embedding types to use. 
                        If len of 1, then  1 embedding; len of 3, embedding medical, diagnosis and others separately (3 embeddings)
                Default: [15817]
            - embed_dim: Number of embedding dimension
                Default: 128
            - hidden_size: Size of hidden layers 
                Default: 128
            - dropout_r: Probability for dropout
                Default: 0.1
            - n_layers: Number of Layers, for Dilated RNNs, dilations will increase exponentialy with mumber of layers 
                Default: 1
            - bii: Indicator of whether Bi-directin is activated. 
                Default: False
            - time: Indicator of whether time is incorporated into embedding. 
                Default: False
            - preTrainEmb: Path to pretrained embeddings file. 
                Default:''
            - output_dir: Output directory where the best model will be saved and logs written 
                Default: we will create'../models/'
            - model_prefix: Prefix name for the saved model e.g: toy.train 
                Default: [(training)file name]
            - model_customed: Second customed specs of name for the saved model e.g: _RNN_GRU. 
                Default: none
            - lr: Learning rate 
                Default: 0.01
            - L2: L2 regularization 
                Default: 0.0001
            - eps: Term to improve numerical stability 
                Default: 0.00000001
            - num_epochs: Number of epochs for training 
                Default: 100
            - patience: Number of stagnant epochs to wait before terminating training 
                Default: 5
            - optimizer: Select which optimizer to train. Upper/lower case does not matter
                Default: adam
            - seed: Seed for reproducibility 
                Default:0
            - use_cuda: Use GPU 
                Default:False
    """
    ###########################################################################
    # 1. Data preparation
    ########################################################################### 
    print("\nLoading and preparing data...")
    if len(files) == 1:
        print('1 file found. Data will be split into train, validation and test.')
        data = EHRdataFromPickles(root_dir = root_dir, 
                                  file_name = files[0], 
                                  sort = False,
                                  test_ratio = test_ratio, 
                                  valid_ratio = valid_ratio,
                                  model = which_model,
                                  seed = seed) #No sort before splitting
        # Dataloader splits
        train, test, valid = data.__splitdata__() #this time, sort is true
        # can comment out this part if you dont want to know what's going on here
        print("\nSee an example data structure from training data:")
        print(data.__getitem__(35, seeDescription = True))
    elif len(files) == 2:
        print('2 files found. 2 dataloaders will be created for train and validation')
        train = EHRdataFromPickles(root_dir = root_dir, 
                                    file_name = files[0], 
                                    sort = True,
                                    model = which_model,
                                    seed = seed)
        valid = EHRdataFromPickles(root_dir = root_dir, 
                                    file_name = files[1], 
                                    sort = True,
                                    model = which_model,
                                    seed = seed)
        test = None
    else:
        print('3 files found. 3 dataloaders will be created for each')
        train = EHRdataFromPickles(root_dir = root_dir, 
                                    file_name = files[0], 
                                    sort = True,
                                    model = which_model,
                                    seed = seed)
        valid = EHRdataFromPickles(root_dir = root_dir, 
                                    file_name = files[1], 
                                    sort = True,
                                    model = which_model,
                                    seed = seed)
        test = EHRdataFromPickles(root_dir = root_dir, 
                                    file_name = files[2], 
                                    sort = True,
                                    model = which_model,
                                    seed = seed)
        print("\nSee an example data structure from training data:")
        print(train.__getitem__(40, seeDescription = True))
    
    print(f"\nTraining data contains {len(train)} patients")
    print(f"Validation data contains {len(valid)} patients")
    print(f"Test data contains {len(test)} patients" if test else "No test file provided")
    ###########################################################################
    # 2. Model loading
    ###########################################################################
    print(f"\n{args.which_model} model initialization...", end="")
    pack_pad = True if which_model == "RNN" else False
    if which_model == 'RNN': 
        ehr_model = models.EHR_RNN(input_size = input_size, 
                                  embed_dim = embed_dim, 
                                  hidden_size = hidden_size,
                                  use_cuda = use_cuda,
                                  n_layers = n_layers,
                                  dropout_r = dropout_r,
                                  cell_type = cell_type,
                                  bii= bii,
                                  time= time,
                                  preTrainEmb = preTrainEmb)
    elif which_model == 'DRNN': 
        ehr_model = models.EHR_DRNN(input_size = input_size, 
                                  embed_dim = embed_dim, 
                                  hidden_size = hidden_size,
                                  use_cuda = use_cuda,
                                  n_layers = n_layers,
                                  dropout_r = dropout_r, #default =0 
                                  cell_type = cell_type, #default ='DRNN'
                                  bii = False,
                                  time = time, 
                                  preTrainEmb = preTrainEmb)
    elif which_model == 'QRNN': 
        ehr_model = models.EHR_QRNN(input_size = input_size, 
                                  embed_dim = embed_dim, 
                                  hidden_size = hidden_size,
                                  use_cuda = use_cuda,
                                  n_layers = n_layers,
                                  dropout_r = dropout_r, #default =0.1
                                  cell_type = 'QRNN', #doesn't support normal cell types
                                  bii = False, #QRNN doesn't support bi
                                  time = time,
                                  preTrainEmb = preTrainEmb)
    elif which_model == 'TLSTM': 
        ehr_model = models.EHR_TLSTM(input_size = input_size, 
                                  embed_dim = embed_dim, 
                                  hidden_size = hidden_size,
                                  use_cuda = use_cuda,
                                  n_layers = n_layers,
                                  dropout_r = dropout_r, #default =0.1
                                  cell_type = 'TLSTM', #doesn't support normal cell types
                                  bii = False, 
                                  time = time, 
                                  preTrainEmb = preTrainEmb)
    elif which_model == 'RETAIN': 
        ehr_model = models.RETAIN(input_size = input_size, 
                                  embed_dim = embed_dim, 
                                  hidden_size = hidden_size,
                                  use_cuda = use_cuda,
                                  n_layers = n_layers)
    else: 
        ehr_model = models.EHR_LR_emb(input_size = input_size,
                                     embed_dim = embed_dim,
                                     use_cuda = use_cuda,
                                     preTrainEmb = preTrainEmb)
    print("Done")
    ###########################################################################
    # 3. call dataloader and create a list of minibatches
    ###########################################################################
    # separate loader and minibatches for train, test, validation 
    # Note: mbs stands for minibatches
    print('\nCreating the list of training minibatches')
    train_mbs = list(tqdm(EHRdataloader(train, use_cuda = use_cuda, batch_size = batch_size, 
                                        packPadMode = pack_pad)))
    print('\nCreating the list of valid minibatches')
    valid_mbs = list(tqdm(EHRdataloader(valid, use_cuda = use_cuda, batch_size = batch_size, 
                                        packPadMode = pack_pad)))
    print ('\nCreating the list of test minibatches')
    test_mbs = list(tqdm(EHRdataloader(test, use_cuda = use_cuda, batch_size = batch_size, 
                                        packPadMode = pack_pad))) if test else None
    
    # make sure cuda is working
    if use_cuda:
        ehr_model = ehr_model.cuda() 
        
    print(f"\n{args.optimizer.title()} optimizer initialization...", end="")
    #model optimizers to choose from. Upper/lower case dont matter
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(ehr_model.parameters(), 
                               lr = lr, 
                               weight_decay = L2,
                               eps = eps)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(ehr_model.parameters(), 
                                   lr = lr, 
                                   weight_decay = L2,
                                   eps = eps)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(ehr_model.parameters(), 
                                  lr = lr, 
                                  weight_decay = L2) 
    elif args.optimizer.lower() == 'adamax':
        optimizer = optim.Adamax(ehr_model.parameters(), 
                                 lr = lr, 
                                 weight_decay = L2,
                                 eps = eps)
    elif args.optimizer.lower() == 'asgd':
        optimizer = optim.ASGD(ehr_model.parameters(), 
                               lr = lr, 
                               weight_decay = L2)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(ehr_model.parameters(), 
                                  lr = lr, 
                                  weight_decay = L2,
                                  eps = eps)
    elif args.optimizer.lower() == 'rprop':
        optimizer = optim.Rprop(ehr_model.parameters(), 
                                lr = lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(ehr_model.parameters(), 
                              lr = lr, 
                              weight_decay = L2)
    else:
        raise NotImplementedError
    print("Done")
    ###########################################################################
    # 4. Train, validation and test. default: batch shuffle = true 
    ###########################################################################
    try:
        ut.epochs_run(num_epochs, 
                      train = train_mbs, 
                      valid = valid_mbs, 
                      test = test_mbs, 
                      model = ehr_model, 
                      optimizer = optimizer,
                      shuffle = True, 
                      #batch_size = batch_size, 
                      which_model = which_model, 
                      patience = patience,
                      output_dir = output_dir,
                      model_prefix = model_prefix,
                      model_customed = model_customed)

    #we can keyboard interupt now 
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    ###########################################################################
    
#do the main file functions and runs 
if __name__ == "__main__":
    # Keyword arguments
    parser = argparse.ArgumentParser(description='Predictive Analytics on EHR with Pytorch')

    #EHRdataloader 
    parser.add_argument('--root_dir', type = str, default = 'data',
                        help='the path to the folders with pickled file(s)')
    
    ### Kept original -files variable not forcing original unique naming for files
    parser.add_argument('--files', nargs='+', default = ['toy.train'],
                        help='''the name(s) of pickled file(s), separated by space. so the argument will be saved as a list 
                        If list of 1: data will be first split into train, validation and test, then 3 dataloaders will be created.
                        If list of 3: 3 dataloaders will be created from 3 files directly. 
                        Files must be in the following order: training, validation and test.''')

    parser.add_argument('--test_ratio', type = float, default = 0.2,
                        help='test data size [default: 0.2]')
    parser.add_argument('--valid_ratio', type = float, default = 0.1,
                        help='validation data size [default: 0.1]')
    parser.add_argument('--batch_size', type=int, default = 128,
                        help='batch size for training, validation or test [default: 128]')
    #EHRmodel
    parser.add_argument('--which_model', type = str, default = 'DRNN',
                        choices= ['RNN','DRNN','QRNN','TLSTM','LR','RETAIN'], 
                        help='choose from {"RNN","DRNN","QRNN","TLSTM","LR","RETAIN"}') 
    parser.add_argument('--cell_type', type = str, default = 'GRU',
                        choices=['RNN', 'GRU', 'LSTM'], 
                        help='For RNN based models, choose from {"RNN", "GRU", "LSTM"}') ## LR removed QRNN and TLSTM 11/6/19
    parser.add_argument('--input_size', nargs='+', type=int, default = [15817],
                        help='''input dimension(s) separated in space the output will be a list, decide which embedding types to use. 
                        If len of 1, then  1 embedding; len of 3, embedding medical, diagnosis and others separately (3 embeddings) [default:[15817]]''')
    parser.add_argument('--embed_dim', type=int, default = 128,
                        help='number of embedding dimension [default: 128]')
    parser.add_argument('--hidden_size', type=int, default = 128,
                        help='size of hidden layers [default: 128]')
    parser.add_argument('--dropout_r', type=float, default = 0.1,
                        help='the probability for dropout[default: 0.1]')
    parser.add_argument('--n_layers', type=int, default = 1,
                        help='number of Layers, for Dilated RNNs, dilations will increase exponentialy with mumber of layers [default: 1]')
    parser.add_argument('--bii', type=bool, default = False,
                        help='indicator of whether Bi-directin is activated. [default: False]')
    parser.add_argument('--time', type=bool, default = False,
                        help='indicator of whether time is incorporated into embedding. [default: False]')
    parser.add_argument('--preTrainEmb', type= str, default = '',
                        help='path to pretrained embeddings file. [default:'']')
    parser.add_argument("--output_dir",type=str, default = 'models',
                        help="The output directory where the best model will be saved and logs written [default: we will create'../models/'] ")
    parser.add_argument('--model_prefix', type = str, default = 'toy.train',
                        help='the prefix name for the saved model e.g: hf.train [default: [(training)file name]')
    parser.add_argument('--model_customed', type = str, default = '',
                        help='the 2nd customed specs of name for the saved model e.g: _RNN_GRU. [default: none]')
    # training 
    parser.add_argument('--lr', type=float, default = 10**-2,
                        help='learning rate [default: 0.01]')
    parser.add_argument('--L2', type=float, default = 10**-4,
                        help='L2 regularization [default: 0.0001]')
    parser.add_argument('--eps', type=float, default = 10**-8,
                        help='term to improve numerical stability [default: 0.00000001]')
    parser.add_argument('--num_epochs', type=int, default = 100,
                        help='number of epochs for training [default: 100]')
    parser.add_argument('--patience', type=int, default = 5,
                        help='number of stagnant epochs to wait before terminating training [default: 5]')
    parser.add_argument('--optimizer', type=str, default = "adam",
                        choices=  ['adam','adadelta','adagrad', 'adamax', 'asgd','rmsprop', 'rprop', 'sgd'], 
                        help='Select which optimizer to train [default: adam]. Upper/lower case does not matter') 
    parser.add_argument('--seed', type= int, default = 0,
                        help='seed for reproducibility [default:0]')
    parser.add_argument('--use_cuda', type=bool, default = False,
                        help='Use GPU [default:False]')
    args = parser.parse_args()

    main(root_dir=args.root_dir, files=args.files, 
                            test_ratio=args.test_ratio, 
                            valid_ratio=args.valid_ratio,
                            batch_size=args.batch_size, 
                            which_model=args.which_model, 
                            cell_type=args.cell_type, 
                            input_size=args.input_size,
                            embed_dim=args.embed_dim, 
                            hidden_size=args.hidden_size, 
                            dropout_r=args.dropout_r, n_layers=args.n_layers, 
                            bii=args.bii, time=args.time, 
                            preTrainEmb=args.preTrainEmb, 
                            output_dir=args.output_dir, 
                            model_prefix=args.model_prefix,
                            model_customed=args.model_customed, lr=args.lr, 
                            L2=args.L2, eps=args.eps, num_epochs=args.num_epochs,
                            patience=args.patience, optimizer=args.optimizer, 
                            seed=args.seed, use_cuda=args.use_cuda)
