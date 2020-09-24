
import sys
import random
import math
try:
    import cPickle as pickle
except:
    import pickle

import torch
from torch import optim
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
#from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

from torch.autograd import Variable

import numpy as np


class TPLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda, bias=True):
        super(TPLSTM, self).__init__()
        self.input_size = input_size -1 ### as the last element is time and we split it our from oun input and assign to gate
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * self.hidden_size, self.input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * self.hidden_size, self.hidden_size))
        self.W_decomp = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * self.hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * self.hidden_size))
            self.b_decomp = Parameter(torch.Tensor(self.hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('b_decomp', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, ip, hx):
        outputh = []
        outputc = []
        #h = hx[0][0]
        for i in range(ip.size()[0]):
            h,c = self.TPLSTMCell(ip[i],hx, self.weight_ih, self.weight_hh, 
                                    self.W_decomp, self.bias_ih, self.bias_hh,
                                    self.b_decomp)
            hx = (h,c)
            outputh.append(h)
            outputc.append(c)
        return outputh, hx, outputc
    
    def TPLSTMCell(self, ip, hidden, w_ih, w_hh, w_decomp, b_ih=None, b_hh=None,
                    b_decomp=None):
        t = torch.transpose(ip, 0, 1)[-1]
        ip = (torch.transpose(ip, 0, 1)[:-1]).transpose(0,1)    
        hx, cx = hidden
        T = self.map_elapse_time(t)
        C_ST = F.tanh(F.linear(cx, w_decomp, b_decomp))
        C_ST_dis = (T * C_ST.squeeze(0)).unsqueeze(0) ###starting time discount
        cpt = cx - C_ST + C_ST_dis
        gates = F.linear(ip, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        ct = (forgetgate * cpt) + (ingate * cellgate) ## Current Memory cell with time
        ht = outgate * F.tanh(ct)

        return ht, ct
    
    def map_elapse_time(self, t):
        c1 = torch.Tensor([1.0])
        c2 = torch.Tensor([2.7183])
        #print('t',t)       #print ('t abs',torch.abs(t*100))
        Ones = torch.ones([1,self.hidden_size])   
        if self.use_cuda:
            c1=c1.cuda()
            c2=c2.cuda()
            Ones=Ones.cuda()
        T = torch.div(c1, torch.log(t + c2))#, name='Log_elapse_time')
        T = torch.matmul(T.view(-1,1), Ones)
        #T[T.ne(T)] = 0.0000001 ##remove nans

        return T
