# -*- coding: utf-8 -*-
import torch
import numpy as np

class Evaluation:
    
    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
        
    
    def evaluate(self):
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)
        # (number_of_layer:1, batch_size:256, hidden_size:16)
        
        with torch.no_grad():
            
            iter_cnt = 0
            recall1_poi = 0
            recall5_poi = 0
            recall10_poi = 0
            average_precision_poi = 0
            
            recall1_catg = 0
            recall5_catg = 0
            recall10_catg = 0
            average_precision_catg = 0

            recall1_catgLayer = 0
            recall5_catgLayer = 0
            recall10_catgLayer = 0
            average_precision_catgLayer = 0
            
            for i,(x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg, y_catgLayer) in enumerate(self.dataloader):
                length = len(x_user)
                h = h[:,:length,:]
                iter_cnt = iter_cnt + length
#
                
                x_user = x_user.squeeze().to(self.setting.device)
        
                x_tf = torch.transpose(x_tf.squeeze(),0,1).to(self.setting.device)
                x_tb = torch.transpose(x_tb.squeeze(),0,1).to(self.setting.device)
                x_tsf = torch.transpose(x_tsf.squeeze(),0,1).to(self.setting.device)
                x_tsb = torch.transpose(x_tsb.squeeze(),0,1).to(self.setting.device)
                x_cof = torch.transpose(x_cof.squeeze(),0,1).to(self.setting.device)
                x_cob = torch.transpose(x_cob.squeeze(),0,1).to(self.setting.device)
                
                x_poi_f = torch.transpose(x_poi_f.squeeze(),0,1).to(self.setting.device)
                x_poi_b = torch.transpose(x_poi_b.squeeze(),0,1).to(self.setting.device)
                x_catg_f = torch.transpose(x_catg_f.squeeze(),0,1).to(self.setting.device)
                x_catg_b = torch.transpose(x_catg_b.squeeze(),0,1).to(self.setting.device)
                x_catgLayer_f = torch.transpose(x_catgLayer_f.squeeze(),0,1).to(self.setting.device)
                x_catgLayer_b = torch.transpose(x_catgLayer_b.squeeze(),0,1).to(self.setting.device)
                
                y_tsecond = y_tsecond.squeeze().to(self.setting.device)
                y_tslot = y_tslot.squeeze().to(self.setting.device)
                y_coord = y_coord.squeeze().to(self.setting.device)
                y_poi = y_poi.squeeze()   
                y_catg = y_catg.squeeze()
                y_catgLayer = y_catgLayer.squeeze()
                
                y_pred_poi, y_pred_catgLayer = self.trainer.evaluate(h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg, y_catgLayer)
                # (batchsize, loc_count)和（batchsize，catg_loc）
                
                for j in range(length):
                    
                    '''for POI'''
                    o_poi = y_pred_poi[j]
#                    o_poi = o_poi.cpu().detach().numpy()
                    o_poi = o_poi.cpu().numpy()
                    ind_poi = np.argpartition(o_poi, -10)[-10:]
                    
                    r_poi = ind_poi[np.argsort(-o_poi[ind_poi], axis=0)]
                    r_poi = torch.tensor(r_poi)
                    
                    # compute for MAP
                    y_poi_index = y_poi[j]      # ground truth_index
                    y_poi_value = o_poi[y_poi_index]
                    upper_poi = np.where(o_poi > y_poi_value)[0]
                    map_poi = 1. / (1+len(upper_poi))
                    
                    recall1_poi += y_poi_index in r_poi[:1]
                    recall5_poi += y_poi_index in r_poi[:5]
                    recall10_poi+= y_poi_index in r_poi[:10]
                    average_precision_poi += map_poi

                    # '''For category'''
                    # o_catg = y_pred_catg[j]
                    # #                    o_catg = o_catg.cpu().detach().numpy()
                    # o_catg = o_catg.cpu().numpy()
                    # ind_catg = np.argpartition(o_catg, -10)[-10:]
                    #
                    # r_catg = ind_catg[np.argsort(-o_catg[ind_catg], axis=0)]
                    # r_catg = torch.tensor(r_catg)
                    #
                    # y_catg_index = y_catg[j]
                    # y_catg_value = o_catg[y_catg_index]
                    # upper_catg = np.where(o_catg > y_catg_value)[0]
                    # map_catg = 1. / (1 + len(upper_catg))
                    #
                    # recall1_catg += y_catg_index in r_catg[:1]
                    # recall5_catg += y_catg_index in r_catg[:5]
                    # recall10_catg += y_catg_index in r_catg[:10]
                    # average_precision_catg += map_catg
                    
                    '''For categoryLayer'''
                    o_catgLayer = y_pred_catgLayer[j]
#                    o_catg = o_catg.cpu().detach().numpy()
                    o_catgLayer = o_catgLayer.cpu().numpy()
                    ind_catgLayer = np.argpartition(o_catgLayer, -10)[-10:]

                    r_catgLayer = ind_catgLayer[np.argsort(-o_catgLayer[ind_catgLayer], axis=0)]
                    r_catgLayer = torch.tensor(r_catgLayer)

                    y_catgLayer_index = y_catgLayer[j]
                    y_catgLayer_value = o_catgLayer[y_catgLayer_index]
                    upper_catgLayer = np.where(o_catgLayer > y_catgLayer_value)[0]
                    map_catgLayer = 1./(1+len(upper_catgLayer))

                    recall1_catgLayer += y_catgLayer_index in r_catgLayer[:1]
                    recall5_catgLayer += y_catgLayer_index in r_catgLayer[:5]
                    recall10_catgLayer += y_catgLayer_index in r_catgLayer[:10]
                    average_precision_catgLayer += map_catgLayer

            print('POI:Recall1:', recall1_poi/iter_cnt)
            print('POI:Recall5:', recall5_poi/iter_cnt)
            print('POI:Recall10:', recall10_poi/iter_cnt)
            print('POI:MAP:', average_precision_poi/iter_cnt)
            
            # print('Catg:Recall1:', recall1_catg/iter_cnt)
            # print('Catg:Recall5:', recall5_catg/iter_cnt)
            # print('Catg:Recall10:', recall10_catg/iter_cnt)
            # print('Catg:MAP:', average_precision_catg/iter_cnt)

            print('CatgLayer:Recall1:', recall1_catgLayer / iter_cnt)
            print('CatgLayer:Recall5:', recall5_catgLayer / iter_cnt)
            print('CatgLayer:Recall10:', recall10_catgLayer / iter_cnt)
            print('CatgLayer:MAP:', average_precision_catgLayer / iter_cnt)

                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
