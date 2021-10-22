# -*- coding: utf-8 -*-

split = Split.TEST
self.new_users = []
self.new_timeSeconds = []
self.new_timeSlots = []
self.new_coords = []
self.new_locs = []
self.new_catgs = []

train_split = '2012-12-14 12:00:00'  # '2012-08-14 12:00:00'  # followd by Bi-STDDP
test_split = '2012-01-15 12:00:00'  # '2012-11-15 12:00:00'  

train_split_timetamp = time.mktime(time.strptime(train_split, "%Y-%m-%d %H:%M:%S"))
test_split_timetamp = time.mktime(time.strptime(test_split, "%Y-%m-%d %H:%M:%S"))


for i, (user, timeSecond, timeSlot, coord, loc, catg) in enumerate(zip(self.users, self.timeSeconds, self.timeSlots, self.coords, self.locs, self.catgs)):
    item_len = len(timeSecond)
    train_threshold = np.sum(np.array(timeSecond) < train_split_timetamp)
    test_threshold = np.sum(np.array(timeSecond) < test_split_timetamp)
    train_rate = train_threshold / item_len
    
    # delete users who has less ten samples
    if train_threshold <= (sequence_length*2+10) or train_rate < 0.7:
        print(user+1)
        print(train_rate)
        continue    
    
    if (split == Split.TRAIN):
        self.new_users.append(user)
        self.new_timeSeconds.append(timeSecond[:train_threshold])
        self.new_timeSlots.append(timeSlot[:train_threshold])
        self.new_coords.append(coord[:train_threshold])
        self.new_locs.append(loc[:train_threshold])
        self.new_catgs.append(catg[:train_threshold])
        
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
    

self.sequence_users = []
self.sequence_pois_forward = []
self.sequence_pois_backward = []
self.sequence_catgs_forward = []
self.sequence_catgs_backward = []
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


for i , (user, timeSecond, timeSlot, coord, loc, catg) in enumerate(zip(self.new_users, self.new_timeSeconds, self.new_timeSlots, self.new_coords, self.new_locs, self.new_catgs)):
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
        # for query item
        self.sequence_query_timeSeconds.append(timeSecond[end_forward])
        self.sequence_query_timeSlots.append(timeSlot[end_forward])
        self.sequence_quary_coords.append(coord[end_forward])
        # for result
        self.sequence_result_poi.append(loc[end_forward])
        self.sequence_result_catg.append(catg[end_forward])


print('用户：',len(self.new_users))
print('长度：',len(self.sequence_users))




















































import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader





class diabetesDataset(Dataset):
    def __init__(self,path):
        super(diabetesDataset,self).__init__()
        xy = np.loadtxt(path,delimiter = ',',dtype = np.float32)
        self.x_data = torch.Tensor(xy[:,:-1])
        self.y_data = torch.Tensor(xy[:,[-1]])
        self.len = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

dataset_1 = diabetesDataset('diabetes.csv')
datatrain = DataLoader(dataset_1, batch_size = 32, shuffle = True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self ).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1) 
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoie(self.linear3(x))
        return x
model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for i, data in enumerate(datatrain, 0):
        inputs, labels = data
        
        
        
#        print(inputs)
#        print(labels)
#        print('*********')
