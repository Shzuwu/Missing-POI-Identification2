# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum

class Rnn(Enum):
    ''' The available RNN units '''
    
    RNN = 0
    GRU = 1    
    LSTM = 2    
    
    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM        
        raise ValueError('{} not supported in --rnn'.format(name))  
        

class RnnFactory():
    ''' Creates the desired RNN unit. '''
    
    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)
                
    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'        
    
    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]
        
    def create(self, hidden_size):  # input_size == hidden_size
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)
        # input_size:(catg + poi + timeslot + user)
        # output_size:(hidden_size)
        
        
def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:        
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))        
    else:        
        return FixNoiseStrategy(hidden_size)

class H0Strategy():
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    
    def on_init(self, user_len, device):
        pass
    
    def on_reset(self, user):
        pass
    
    def on_reset_test(self, user, device):
        return self.on_reset(user)
    
class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''
    
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
    
    def on_init(self, batch_size, device):
        hs = []
        for i in range(batch_size):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, batch_size, self.hidden_size).to(device)
#       h0——>>(number_layer:1, batch_size:256, hidden_size10)
    
    def on_reset(self, user):
        return self.h0

class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''
    
    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy
    
    def on_init(self, batch_size, device):
        h = self.h_strategy.on_init(batch_size, device)
        c = self.c_strategy.on_init(batch_size, device)
        return (h,c)
    
    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h,c)
    
class Flashback(nn.Module):
    def __init__(self, poi_count, user_count, catg_count, catgLayer_count, timeslot_count, hidden_size, f_t, f_t1, f_s, rnn_factory):
        super().__init__()
        self.poi_count = poi_count
        self.user_count = user_count
        self.catg_count = catg_count
        self.catgLayer_count = catgLayer_count
        self.timeslot_count = timeslot_count
        self.f_t = f_t
        self.f_t1 = f_t1
        self.f_s = f_s
        self.hidden_size = hidden_size
        
        '''Embedding'''
        self.poi_encoder = nn.Embedding(poi_count, hidden_size)
        self.user_encoder = nn.Embedding(user_count, hidden_size)
        self.catg_encoder = nn.Embedding(catg_count, hidden_size)
        self.catgLayer_encoder = nn.Embedding(catgLayer_count, hidden_size)
        self.timeslot_encoder = nn.Embedding(timeslot_count, hidden_size)
       
        '''Create RNN'''
        self.rnn_poi_forward = rnn_factory.create(hidden_size)     
        self.rnn_poi_backward = rnn_factory.create(hidden_size)
        self.rnn_catg_forward = rnn_factory.create(hidden_size)
        self.rnn_catg_backward = rnn_factory.create(hidden_size)
        self.rnn_catgLayer_forward = rnn_factory.create(hidden_size)
        self.rnn_catgLayer_backward= rnn_factory.create(hidden_size)

        input_len = 4
        input_len1 = 6
        self.input_len = input_len
        self.input_len1 = input_len1
        '''线性'''
        self.fc_poi = nn.Linear(input_len*hidden_size, poi_count)     # user + timeslot + poi_forward + poi_backward
        self.fc_catg = nn.Linear(input_len*hidden_size, catg_count)   # user + timeslot + catg_forward + catg_backward
        self.fc_catgLayer = nn.Linear(input_len*hidden_size, catg_count)  # user + timeslot + catgLayer_forward + catgLayer_backward
        
        self.fc1_poi = nn.Linear(input_len1*hidden_size, poi_count)    # user + timeslot + poi_forward + poi_backward + catg_forward + catg_backward
        self.fc1_catg = nn.Linear(input_len1*hidden_size, catg_count)  # user + timeslot + poi_forward + poi_backward + catg_forward + catg_backward
        self.fc1_catgLayer = nn.Linear(input_len1*hidden_size, poi_count)    # user + timeslot + poi_forward + poi_backward + catgLayer_forward + catgLayer_backward

        '''激活'''
        self.activate = nn.Sigmoid()
        
    def forward(self, h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, \
                y_tsecond, y_tslot, y_coord, y_poi, y_catg):
        
        # (seq_len, batch_size)
        seq_len, batch_size = x_poi_f.size()
        p_u = self.user_encoder(x_user)      #(256, 16)
        p_tsf = self.timeslot_encoder(x_tsf) #(10, 256, 16)  (seq_len, batch_size, embeding_size)
        p_tsb = self.timeslot_encoder(x_tsb) #(10, 256, 16)
        p_poi_f = self.poi_encoder(x_poi_f)  #(10, 256, 16)
        p_poi_b = self.poi_encoder(x_poi_b)  #(10, 256, 16)
        p_catg_f = self.catg_encoder(x_catg_f)   #(10, 256, 16)
        p_catg_b = self.catg_encoder(x_catg_b)   #(10, 256, 16)
        p_catgLayer_f = self.catgLayer_encoder(x_catgLayer_f)   #(10, 256, 16)
        p_catgLayer_b = self.catgLayer_encoder(x_catgLayer_b)   #(10, 256, 16)
        
        p_tslot = self.timeslot_encoder(y_tslot)
        
        '''输入RNN网络'''
        out_poi_forward, h_pf = self.rnn_poi_forward(p_poi_f, h)
        out_poi_backward, h_pb = self.rnn_poi_backward(p_poi_b, h)
        out_catg_forward, h_cf = self.rnn_catg_forward(p_catg_f,h)
        out_catg_backward, h_cb = self.rnn_catg_backward(p_catg_b, h)
        out_catgLayer_forward, h_cf1 = self.rnn_catgLayer_forward(p_catgLayer_f, h)
        out_catgLayer_backward, h_cb1 = self.rnn_catgLayer_backward(p_catgLayer_b, h)
        
        
        ''' x_tf, x_tb, x_cof, x_cob, y_tsecond, y_tslot, y_coord, y_poi, y_catg'''
        out_w_poi_forward = torch.zeros(batch_size, self.hidden_size, device=x_user.device)
        out_w_poi_backward = torch.zeros(batch_size, self.hidden_size, device=x_user.device)
        out_w_catg_forward = torch.zeros(batch_size, self.hidden_size, device=x_user.device)
        out_w_catg_backward = torch.zeros(batch_size, self.hidden_size, device=x_user.device)
        out_w_catgLayer_forward = torch.zeros(batch_size, self.hidden_size, device=x_user.device)
        out_w_catgLayer_backward = torch.zeros(batch_size, self.hidden_size, device=x_user.device)

        sum_w_space_forward = torch.zeros(batch_size, 1, device=x_user.device)
        sum_w_space_backward = torch.zeros(batch_size, 1, device=x_user.device)
        sum_w_time_forward = torch.zeros(batch_size, 1, device=x_user.device)
        sum_w_time_backward = torch.zeros(batch_size, 1, device=x_user.device)
        sum_w_time1_forward = torch.zeros(batch_size, 1, device=x_user.device)
        sum_w_time1_backward = torch.zeros(batch_size, 1, device=x_user.device)
        sum_w_forward = torch.zeros(batch_size, 1, device=x_user.device)
        sum_w_backward = torch.zeros(batch_size, 1, device=x_user.device)
        
        for i in range(seq_len):

            dist_t_forward = y_tsecond - x_tf[i]
            dist_t_backward = x_tb[i] - y_tsecond
            dist_s_forward = torch.norm(x_cof[-1]-x_cof[i], dim=-1)
            dist_s_backward = torch.norm(x_cob[-1]-x_cob[i], dim=-1)

            t_forward = self.f_t(dist_t_forward)
            t_backward = self.f_t(dist_t_backward)
            t1_forward = self.f_t1(dist_t_forward)
            t1_backward = self.f_t1(dist_t_backward)
            s_forward = self.f_s(dist_s_forward)
            s_backward = self.f_s(dist_s_backward)
            
            t_forward = t_forward.unsqueeze(1)
            t_backward = t_backward.unsqueeze(1)
            t1_forward = t1_forward.unsqueeze(1)
            t1_backward = t1_backward.unsqueeze(1)
            s_forward = s_forward.unsqueeze(1)
            s_backward = s_backward.unsqueeze(1)

            '''考虑时间周期的影响'''
            w_forward = t_forward * s_forward + 1e-10
            w_backward = t_backward * s_backward + 1e-10
            t_forward += (1e-10)
            t_backward += (1e-10)
            t1_forward += (1e-10)
            t1_backward += (1e-10)
            s_forward += (1e-10)
            s_backward += (1e-10)

            '''求和'''
            sum_w_forward += w_forward
            sum_w_backward += w_backward
            sum_w_time_forward += t_forward
            sum_w_time_backward += t_backward
            sum_w_time1_forward += t1_forward
            sum_w_time1_backward += t1_backward
            sum_w_space_forward += s_forward
            sum_w_space_backward += s_backward

            '''
            可以微调'''
            out_w_poi_forward += s_forward * out_poi_forward[i]
            out_w_poi_backward += s_backward * out_poi_backward[i]
            out_w_catgLayer_forward += t_forward * out_catgLayer_forward[i]
            out_w_catgLayer_backward += t_backward * out_catgLayer_backward[i]
        
        '''连带微调'''
        out_w_poi_forward /= sum_w_space_forward
        out_w_poi_backward /= sum_w_space_backward
        out_w_catgLayer_forward /= sum_w_time_forward
        out_w_catgLayer_backward/= sum_w_time_backward
        
        '''初始化 input_len = 4, input_len1 = 6'''
        out_poi = torch.zeros(batch_size, self.input_len*self.hidden_size, device=x_user.device)
        out_catgLayer = torch.zeros(batch_size, self.input_len*self.hidden_size, device=x_user.device)
        out_allLayer = torch.zeros(batch_size, self.input_len1*self.hidden_size,device=x_user.device)

        '''赋值'''
        out_poi = torch.cat([p_u, p_tslot, out_w_poi_forward, out_w_poi_backward], dim=1)
        out_catgLayer = torch.cat([p_u, p_tslot, out_w_catgLayer_forward, out_w_catgLayer_backward], dim=1)
        out_allLayer = torch.cat([p_u, p_tslot, out_w_poi_forward, out_w_poi_backward, out_w_catgLayer_forward, out_w_catgLayer_backward], dim=1)     

        '''可以调整'''
        y_pred_poi = self.fc_poi(out_poi)
        y_pred_catgLayer = self.fc_catgLayer(out_catgLayer)

        y_pred_poi1 = self.fc1_poi(out_allLayer)
        y_pred_catgLayer1 = self.fc1_catgLayer(out_allLayer)

        return y_pred_poi1, y_pred_catgLayer1