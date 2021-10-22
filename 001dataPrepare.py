# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:17:42 2021

@author: Administrator
1. 将user，POI和category进行编码
2. 将原来的一个POI对应多个坐标的情况，通过均值的方式，实现统一化
"""

import numpy as np
import pandas as pd
import time

# read categories
path0 = r'Foursquare_categories.csv'
df0 = pd.read_csv(path0, sep='\t',encoding='unicode_escape')
df0.columns = ['category_tree']
category_tree = list(df0['category_tree'])
category_dict = dict()
for i in range(len(category_tree)):
    temp = category_tree[i]
    temp = temp.split(',')
    temp = temp[::2]
    temp = [i for i in temp if i != '']
    category_tree[i] = temp
    
    while True:
        key = temp.pop()
        if key not in category_dict:
            category_dict[key] = temp[::-1]
        if len(temp) == 1:
            break

print('hello world')
data = 'TKY'
path = r'data/original data/dataset_TSMC2014_{}.txt'.format(data)
df = pd.read_csv(path, header=None, sep='\t',encoding='unicode_escape')
df_columns = ['user', 'poi', 'categoryID', 'category', 'lat', 'lon', 'off_set', 'timestamp']
df.columns = df_columns

# 统一category,（lat，lon）
df_data = pd.DataFrame()
df_data['user'] = df['user']
df_data['timestamp'] = df['timestamp']
df_data['lat'] = df['lat']
df_data['lon'] = df['lon']
df_data['poi'] = df['poi']
df_data['category'] = df['category']

#
category_small = list(set(df['category']))
category_small = ['Cafe' if i=='Café' else i for i in category_small]
category_big = [category_dict[i][-1] for i in category_small]
Name_category = pd.DataFrame()
Name_category['category_small'] = category_small
Name_category['category_big'] = category_big
Name_category.to_csv(r'{}_category.csv'.format(data), sep = '\t', index = False, header = False)

# 取L1 category或者是L2 category
list_catg = list(df_data['category'])
list_catg1 = []
for i in range(len(list_catg)):
    catg_temp = list_catg[i]

    if len(category_dict.get(catg_temp, ["Food"])) == 1:
        list_catg1.append(catg_temp)
    else:
        list_catg1.append(category_dict[catg_temp][-2])
df_data['category1'] = list_catg1

df_data['timeConvert'] = [time.mktime(time.strptime(
            df_data['timestamp'][index], '%a %b %d %X +0000 %Y')) for index in df_data['timestamp'].index]


df_data = df_data.sort_values(by=['user', 'timeConvert'])
#df_data = df_data.drop(['timeConvert'], axis=1)

# str2index
list_user = sorted(list(set(df_data['user'])))
list_poi = sorted(list(set(df_data['poi'])))
list_category = sorted(list(set(df_data['category'])))
list_category1 = sorted(list(set(df_data['category1'])))

list_items = list_user + list_poi + list_category + list_category1
dict_items = dict()
for index, value in enumerate(list_items):
    dict_items[value] = index

user = [dict_items[i] for i in df_data['user']]
poi = [dict_items[i] for i in df_data['poi']]
category = [dict_items[i] for i in df_data['category']]
category1 = [dict_items[i] for i in df_data['category1']]
df_data['user'] = user
df_data['poi'] = poi
df_data['category'] = category
df_data['category1'] = category1
df_data = df_data.set_index('user')


## average loc for mulit same POI
# built a dict——>key:poi, value: multi coords
poi2coord = dict()
list_poi = list(df_data['poi'])
list_coord = list(zip(df_data['lat'], df_data['lon']))
for i in range(len(list_poi)):
    if list_poi[i] not in poi2coord:
        poi2coord[list_poi[i]] = [list_coord[i]]
    else:
        poi2coord[list_poi[i]].append(list_coord[i])
# for averange        
poi2coord1 = poi2coord.copy()
for i in list(poi2coord1.keys()):
    poi2coord1[i] = list(np.mean(poi2coord1[i], axis=0))

list_lat = list(df_data['lat'])
list_lon = list(df_data['lon'])
for i in range(len(list_poi)):
    coord_temp = poi2coord1[list_poi[i]]
    list_lat[i] = coord_temp[0]
    list_lon[i] = coord_temp[1]
df_data['lat'] = list_lat
df_data['lon'] = list_lon
df_data.to_csv(r'data/original data/{}_new.txt'.format(data), sep = '\t', index = True, header = False)

# for Flashback dataset generate
temp = df_data.reset_index()
temp = temp.sort_values(by=['user', 'timeConvert'],ascending=False)

df_dataFlash = pd.DataFrame()
temp_user = list(temp['user'])
temp_time = list(temp['timestamp'])
temp_lat = list(temp['lat'])
temp_lon = list(temp['lon'])
temp_poi = list(temp['poi'])
temp_time1 = []
for i in range(len(temp_time)):
    temp_time1.append(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.strptime(temp_time[i], '%a %b %d %X +0000 %Y')))
    
df_dataFlash['user'] = temp_user
df_dataFlash['timestamp'] = temp_time1
df_dataFlash['lat'] = temp_lat
df_dataFlash['lon'] = temp_lon
df_dataFlash['poi'] = temp_poi
df_dataFlash = df_dataFlash.set_index('user')
df_dataFlash.to_csv(r'data/original data/{}_newFlash.txt'.format(data), sep = '\t', index = True, header = False)
