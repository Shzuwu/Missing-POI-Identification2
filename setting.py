# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:27:37 2021

@author: Administrator
"""

import torch
import argparse
import sys
from network import RnnFactory


class Setting:
    def parse(self):
        self.guess_TKY = any(['TKY' in argv for argv in sys.argv])
        parser = argparse.ArgumentParser()
        if self.guess_TKY:
            self.parse_TKY(parser)
        else:
            self.parse_NYC(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()
        
        ## setting
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s
        
        # data management
        self.dataset_file = r'data/original data/{}'.format(args.dataset)
        self.max_users = 0
        self.sequence_length = 10
        self.batch_size = args.batch_size
        self.min_checkins = self.sequence_length * 2 *3 + 5
        
        # evaluation
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda', args.gpu)
            # self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
  
    def parse_arguments(self, parser):
        parser.add_argument('--gpu', default = 0, type = int)
        parser.add_argument('--hidden_dim', default = 16, type = int)
        parser.add_argument('--weight_decay', default = 0.0, type = float)
        parser.add_argument('--lr', default = 0.01, type = float)
        parser.add_argument('--epochs', default = 100, type = int)
        parser.add_argument('--rnn', default = 'gru', type = str, help = 'the GPU iplementation on use: [rnn|gru|lstm]')

        parser.add_argument('--dataset', default = 'NYC_new.txt', type = str)
        parser.add_argument('--validate_epoch', default=5, type=int, help='run each validation after this amount of epochs')
        parser.add_argument('--report_user', default=-1, type=int, help='report every x user on evaluation (-1: ignore)') 

    def parse_NYC(self, parser):
        parser.add_argument('--batch-size', default = 256, type = int)
        parser.add_argument('--lambda_t', default = 0.2, type = float)
        parser.add_argument('--lambda_s', default = 100, type = float)

    def parse_TKY(self, parser):
        parser.add_argument('--batch-size', default = 256, type=int)
        parser.add_argument('--lambda_t', default = 0.2, type = float)
        parser.add_argument('--lambda_s', default = 100, type = float)
        
    def __str__(self):        
        return ('parse with TKY default settings' if self.guess_TKY else 'parse with NYC default settings') + '\n'\
            + 'use device: {}'.format(self.device)   
