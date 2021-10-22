# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:47:55 2021

@author: Administrator
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import Split

from setting import Setting
from dataloader import PoiDataLoader
from trainer import FlashbackTrainer
from network import create_h0_strategy
from evaluation import Evaluation

setting = Setting()
setting.parse()
print(setting)

# for training set
poi_loader = PoiDataLoader(setting.min_checkins)
poi_loader.read(setting.dataset_file)
dataset = poi_loader.create_dataset(setting.sequence_length, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=setting.batch_size, shuffle=True)

# for validate set
dataset_validate = poi_loader.create_dataset(setting.sequence_length, Split.VALIDATE)
dataloader_validata = DataLoader(dataset_validate, batch_size=setting.batch_size, shuffle=False)

# for test set
dataset_test = poi_loader.create_dataset(setting.sequence_length, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=setting.batch_size, shuffle=False)

# for training 
trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
trainer.prepare(poi_loader.poi_count(), poi_loader.user_count(), poi_loader.catg_count(), poi_loader.catgLyaer_count(), poi_loader.timeslot_count(), poi_loader.poi2coord, setting.hidden_dim, setting.rnn_factory, setting.device)

# for Evaluation
evaluation_vaid = Evaluation(dataset_validate, dataloader_validata, poi_loader.user_count(), h0_strategy, trainer, setting)
evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)

# for Optimization
optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120,160], gamma=0.6)

poi2coord = torch.tensor(poi_loader.poi2coord, device=setting.device)
print('Seqence_len:',setting.sequence_length)
print('lambda_s',setting.lambda_s)
print('lambda_t',setting.lambda_t)
print('hidden', setting.hidden_dim)

for epoch in range(6):
    h = h0_strategy.on_init(setting.batch_size, device = setting.device)
    loss_value = []
    print('epoch:',epoch+1)
    for i,(x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg, y_catgLayer) in enumerate(dataloader):
        
        length = len(x_user)
        h = h[:,:length,:]

        x_user = x_user.squeeze().to(setting.device)

        x_tf = torch.transpose(x_tf.squeeze(),0,1).to(setting.device)
        x_tb = torch.transpose(x_tb.squeeze(),0,1).to(setting.device)
        x_tsf = torch.transpose(x_tsf.squeeze(),0,1).to(setting.device)
        x_tsb = torch.transpose(x_tsb.squeeze(),0,1).to(setting.device)
        x_cof = torch.transpose(x_cof.squeeze(),0,1).to(setting.device)
        x_cob = torch.transpose(x_cob.squeeze(),0,1).to(setting.device)

        x_poi_f = torch.transpose(x_poi_f.squeeze(),0,1).to(setting.device)
        x_poi_b = torch.transpose(x_poi_b.squeeze(),0,1).to(setting.device)
        x_catg_f = torch.transpose(x_catg_f.squeeze(),0,1).to(setting.device)
        x_catg_b = torch.transpose(x_catg_b.squeeze(),0,1).to(setting.device)
        x_catgLayer_f = torch.transpose(x_catgLayer_f.squeeze(),0,1).to(setting.device)
        x_catgLayer_b = torch.transpose(x_catgLayer_b.squeeze(),0,1).to(setting.device)

        y_tsecond = y_tsecond.squeeze().to(setting.device)
        y_tslot = y_tslot.squeeze().to(setting.device)
        y_coord = y_coord.squeeze().to(setting.device)
        y_poi = y_poi.squeeze().to(setting.device)
        y_catg = y_catg.squeeze().to(setting.device)
        y_catgLayer = y_catgLayer.squeeze().to(setting.device)

        optimizer.zero_grad()
        loss = trainer.loss(h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg, y_catgLayer)
        loss.backward()
        loss_value.append(loss.item())
        optimizer.step()

    scheduler.step()
#    print(f'Used learning rate: {scheduler.get_last_lr()[0]}')
    print(np.mean(loss_value))
    if (epoch + 1) % 2 == 0:
        # evaluation_vaid.evaluate()
        # print('*****************')
        evaluation_test.evaluate()
       
       
       

       
       
       
       
       
       
       
       
       
       