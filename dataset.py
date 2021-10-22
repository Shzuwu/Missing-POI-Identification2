# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:32:48 2021

@author: Administrator
"""

import random
import torch
from enum import Enum
from torch.utils.data import Dataset
import numpy as np
import time

class Split(Enum):
    
    TRAIN = 0
    VALIDATE = 1
    TEST = 2
    
class PoiDataset(Dataset):
    
    def __init__(self, users, timeSeconds, timeSlots, coords, locs, catgs, catgLayers, sequence_length, split, loc_count):
        self.users = users
        self.timeSeconds = timeSeconds
        self.timeSlots = timeSlots
        
        self.coords = coords
        self.locs = locs
        self.catgs = catgs
        self.catgsLayers = catgLayers
        self.sequence_length = sequence_length
        self.split = split
        self.loc_count = loc_count
        
        train_split = '2012-12-14 12:00:00'  # '2012-08-14 12:00:00'  # followd by Bi-STDDP
        test_split = '2013-01-15 12:00:00'  # '2012-11-15 12:00:00'  
        
        train_split_timetamp = time.mktime(time.strptime(train_split, "%Y-%m-%d %H:%M:%S"))
        test_split_timetamp = time.mktime(time.strptime(test_split, "%Y-%m-%d %H:%M:%S"))
        
        self.new_users = []
        self.new_timeSeconds = []
        self.new_timeSlots = []
        self.new_coords = []
        self.new_locs = []
        self.new_catgs = []
        self.new_catgLayers = []
        
        for i, (user, timeSecond, timeSlot, coord, loc, catg, catgLayer) in enumerate(zip(self.users, self.timeSeconds, self.timeSlots, self.coords, self.locs, self.catgs, self.catgsLayers)):
            item_len = len(timeSecond)
            train_threshold = np.sum(np.array(timeSecond) < train_split_timetamp)   # 计算训练样本数量
            test_threshold = np.sum(np.array(timeSecond) < test_split_timetamp)     # 计算测试样本数量 
            train_rate = train_threshold / item_len
            
            # delete users who has less ten samples 
            '''（1）如果训练样本数量小于10；
            （2）或者训练样本占比小于50%；
            （3）或者测试样本小于1，则pass'''
            if train_threshold <= (sequence_length*2+10) or train_rate < 0.7 or (item_len - test_threshold) <= (sequence_length*2+1):
#                print(user+1)
#                print(train_rate)
                continue
            
            if (split == Split.TRAIN):
                self.new_users.append(user)
                self.new_timeSeconds.append(timeSecond[:train_threshold])
                self.new_timeSlots.append(timeSlot[:train_threshold])
                self.new_coords.append(coord[:train_threshold])
                self.new_locs.append(loc[:train_threshold])
                self.new_catgs.append(catg[:train_threshold])
                self.new_catgLayers.append(catgLayer[:train_threshold])
                
            if (split == Split.VALIDATE):
                if (test_threshold - train_threshold) <= (sequence_length*2+1):
                    continue
                else:
                    self.new_users.append(user)
                    self.new_timeSeconds.append(timeSecond[train_threshold:test_threshold])
                    self.new_timeSlots.append(timeSlot[train_threshold:test_threshold])
                    self.new_coords.append(coord[train_threshold:test_threshold])
                    self.new_locs.append(loc[train_threshold:test_threshold])
                    self.new_catgs.append(catg[train_threshold:test_threshold])
                    self.new_catgLayers.append(catgLayer[train_threshold:test_threshold])
                 
            if (split == Split.TEST): 
                if (item_len - test_threshold) <= (sequence_length*2+1):
                    continue
                else:
                    self.new_users.append(user)
                    self.new_timeSeconds.append(timeSecond[test_threshold:])
                    self.new_timeSlots.append(timeSlot[test_threshold:])
                    self.new_coords.append(coord[test_threshold:])
                    self.new_locs.append(loc[test_threshold:])
                    self.new_catgs.append(catg[test_threshold:])
                    self.new_catgLayers.append(catgLayer[test_threshold:])
            
        self.sequence_users = []
        self.sequence_pois_forward = []
        self.sequence_pois_backward = []
        self.sequence_catgs_forward = []
        self.sequence_catgs_backward = []
        self.sequence_catgLayers_forward = []
        self.sequence_catgLayers_backward = []
        self.sequence_timeSeconds_forward = []
        self.sequence_timeSeconds_backward = []
        self.sequence_timeSlots_forward = []
        self.sequence_timeSlots_backward = []
        self.sequence_coords_forward = []
        self.sequence_coords_backward = []
        self.sequence_query_timeSeconds = []
        self.sequence_query_timeSlots = []
        self.sequence_quary_coords = []
        self.sequence_result_poi = []
        self.sequence_result_catg = []
        self.sequence_result_catgLayer = []
        
        for i , (user, timeSecond, timeSlot, coord, loc, catg, catgLayer) in enumerate(zip(self.new_users, self.new_timeSeconds, self.new_timeSlots, self.new_coords, self.new_locs, self.new_catgs, self.new_catgLayers)):
            # 样本数量
            seq_count = len(loc) - sequence_length * 2
            
            # for every user
            for i in range(seq_count):
                start_forward = i
                end_forward = start_forward + sequence_length
                
                start_backward = end_forward + 1
                end_backward = start_backward + sequence_length
                
                # for user
                self.sequence_users.append(user)
                # for timeSecond
                self.sequence_timeSeconds_forward.append(timeSecond[start_forward : end_forward])
                self.sequence_timeSeconds_backward.append(timeSecond[start_backward : end_backward][::-1])
                # for timeSlot
                self.sequence_timeSlots_forward.append(timeSlot[start_forward : end_forward])
                self.sequence_timeSlots_backward.append(timeSlot[start_backward : end_backward][::-1])
                # for timeSlot
                # for coord
                self.sequence_coords_forward.append(coord[start_forward : end_forward])
                self.sequence_coords_backward.append(coord[start_backward : end_backward][::-1])
                # for loc
                self.sequence_pois_forward.append(loc[start_forward : end_forward])
                self.sequence_pois_backward.append(loc[start_backward : end_backward][::-1])
                # for catg
                self.sequence_catgs_forward.append(catg[start_forward : end_forward])
                self.sequence_catgs_backward.append(catg[start_backward : end_backward][::-1])
                # for catgLayer
                self.sequence_catgLayers_forward.append(catgLayer[start_forward : end_forward])
                self.sequence_catgLayers_backward.append(catgLayer[start_backward : end_backward][::-1])
                # for query item
                self.sequence_query_timeSeconds.append(timeSecond[end_forward])
                self.sequence_query_timeSlots.append(timeSlot[end_forward])
                self.sequence_quary_coords.append(coord[end_forward])
                # for result
                self.sequence_result_poi.append(loc[end_forward])
                self.sequence_result_catg.append(catg[end_forward])
                self.sequence_result_catgLayer.append(catgLayer[end_forward])
        
        
        print('用户：',len(self.new_users))
        print('长度：',len(self.sequence_users))
        
        self.sample_len = len(self.sequence_result_poi)
        
    def __len__(self):
        return self.sample_len
            
    def __getitem__(self, index):
        x_user = torch.tensor(self.sequence_users[index])
        
        x_tf = torch.tensor(self.sequence_timeSeconds_forward[index])
        x_tb= torch.tensor(self.sequence_timeSeconds_backward[index])      
        x_tsf = torch.tensor(self.sequence_timeSlots_forward[index])   
        x_tsb= torch.tensor(self.sequence_timeSlots_backward[index])  
        x_cof = torch.tensor(self.sequence_coords_forward[index])  
        x_cob= torch.tensor(self.sequence_coords_backward[index])
        x_poi_f = torch.tensor(self.sequence_pois_forward[index])
        x_poi_b= torch.tensor(self.sequence_pois_backward[index])
        x_catg_f= torch.tensor(self.sequence_catgs_forward[index])
        x_catg_b = torch.tensor(self.sequence_catgs_backward[index])
        x_catgLayer_f = torch.tensor(self.sequence_catgLayers_forward[index])
        x_catgLayer_b = torch.tensor(self.sequence_catgLayers_backward[index])
        
        y_tsecond = torch.tensor(self.sequence_query_timeSeconds[index])
        y_tslot = torch.tensor(self.sequence_query_timeSlots[index])
        y_coord = torch.tensor(self.sequence_quary_coords[index])
        
        y_poi = torch.tensor(self.sequence_result_poi[index])
        y_catg = torch.tensor(self.sequence_result_catg[index])
        y_catgLayer = torch.tensor(self.sequence_result_catgLayer[index])
        
        return x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg, y_catgLayer
    




































        