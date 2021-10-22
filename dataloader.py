# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:47:17 2021

@author: Administrator
"""

import os.path
import sys

from datetime import datetime
import time
from dataset import PoiDataset

def timeValue(day_hourminute):
    day = eval(day_hourminute.split(':')[0])
    hourminute = day_hourminute.split(':')[1]
    date_divide = 8
    if '0200' <= hourminute < '0500':
        index = 0
    elif '0500' <= hourminute < '0800':
        index = 1
    elif '0800' <= hourminute < '1100':
        index = 2
    elif '1100' <= hourminute < '1400':
        index = 3
    elif '1400' <= hourminute < '1700':
        index = 4
    elif '1700' <= hourminute < '2000':
        index = 5
    elif '2000' <= hourminute < '2300':
        index = 6    
    elif hourminute >= '2300' or hourminute < '0200':
        index = 7
    else:
        print('time error:', hourminute)
    return day*date_divide+index

class PoiDataLoader():
    def __init__(self, min_checkins=0):
        self.min_checkins = min_checkins
        
        self.user2id = {}
        self.poi2id = {}
        self.catg2id = {}
        self.catg2id_layer = {}
        
        self.users = []
        self.timeSeconds = []
        self.timeSlots = []
#        self.timeCovert = []
        self.coords = []
        self.locs = []
        self.catgs = []
        self.catgLayers = []
    
    def user_count(self):
        return len(self.users)
    
    def poi_count(self):
        return len(self.poi2id)
    
    def catg_count(self):
        return len(self.catg2id)
    
    def catgLyaer_count(self):
        return len(self.catg2id_layer)
    
    def timeslot_count(self):
        return self.date_divide * self.week_divide
    
    def timeValue(self, day_hourminute):
        self.date_divide = 8
        self.week_divide = 7
        day = eval(day_hourminute.split(':')[0])
        hourminute = day_hourminute.split(':')[1]
        if '0200' <= hourminute < '0500':
            index = 0
        elif '0500' <= hourminute < '0800':
            index = 1
        elif '0800' <= hourminute < '1100':
            index = 2
        elif '1100' <= hourminute < '1400':
            index = 3
        elif '1400' <= hourminute < '1700':
            index = 4
        elif '1700' <= hourminute < '2000':
            index = 5
        elif '2000' <= hourminute < '2300':
            index = 6    
        elif hourminute >= '2300' or hourminute < '0200':
            index = 7
        else:
            print('time error:', hourminute)
        return day * self.date_divide + index
    
    def read(self, file):
        self.read_users(file)
        self.read_pois(file)
    
    def read_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        
        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                    
                prev_user = user
                visit_cnt = 1
        
        # 给最后一个人的确认
        if visit_cnt >= self.min_checkins:
            self.user2id[prev_user] = len(self.user2id)
    
    def create_dataset(self, sequence_length, split):
        print('hello world!')
        return PoiDataset(self.users.copy(),\
                          self.timeSeconds.copy(),\
                          self.timeSlots.copy(),\
                          self.coords.copy(),\
                          self.locs.copy(),\
                          self.catgs.copy(),\
                          self.catgLayers.copy(),\
                          sequence_length,\
                          split,\
                          len(self.poi2id))
    
    
    def read_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        
        user_timeSecond = []
        user_timeSlot = []
#        user_timeCovert = []
        user_coord = []
        user_loc = []
        user_catg = []
        user_catgLayer = []
        self.poi2coord = []
        
        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue
            user = self.user2id.get(user)
            
#            time_second = (datetime.strptime(tokens[1], '%a %b %d %X +0000 %Y') - datetime(2010, 1, 1)).total_seconds()
            time_second = float(tokens[7])
            time_slot = self.timeValue(time.strftime("%w:%H%M", time.strptime(tokens[1],'%a %b %d %X +0000 %Y')))


            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)
            
            location = int(tokens[4])
            if self.poi2id.get(location) is None:
                self.poi2id[location] = len(self.poi2id)
                self.poi2coord.append(coord)
            location = self.poi2id.get(location)
            
            category = int(tokens[5])
            if self.catg2id.get(category) is None:
                self.catg2id[category] = len(self.catg2id)
            category = self.catg2id.get(category)
            
            category_layer = int(tokens[6])
            if self.catg2id_layer.get(category_layer) is None:
                self.catg2id_layer[category_layer] = len(self.catg2id_layer)
            category_layer = self.catg2id_layer.get(category_layer)     
            
            
            if user == prev_user:
                user_timeSecond.append(time_second)
                user_timeSlot.append(time_slot)
#                user_timeCovert.append(time_covert)
                user_coord.append(coord)
                user_loc.append(location)
                user_catg.append(category)
                user_catgLayer.append(category_layer)
            else:
                self.users.append(prev_user)
                self.timeSeconds.append(user_timeSecond)
                self.timeSlots.append(user_timeSlot)
#                self.timeCovert.append(user_timeCovert)
                self.coords.append(user_coord)
                self.locs.append(user_loc)
                self.catgs.append(user_catg)
                self.catgLayers.append(user_catgLayer)
                
                prev_user = user
                user_timeSecond = [time_second]
                user_timeSlot = [time_slot]
#                user_timeCovert = [time_covert]
                user_coord = [coord]
                user_loc = [location]
                user_catg = [category]
                user_catgLayer = [category_layer]
        
        self.users.append(prev_user)
        self.timeSeconds.append(user_timeSecond)
        self.timeSlots.append(user_timeSlot)
#        self.timeCovert.append(user_timeCovert)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
        self.catgs.append(user_catg)
        self.catgLayers.append(user_catgLayer)
                
            
            
                
        
        
        
        
        
        
        
    