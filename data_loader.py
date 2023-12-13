import os
import random
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class Sets(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        users = torch.tensor(self.data.iloc[index][0], dtype=torch.long)
        items = torch.tensor(self.data.iloc[index][1], dtype=torch.long)
        label = torch.tensor(self.data.iloc[index][2], dtype=torch.long)
        return users, items, label
    
    
    def __len__(self):
        return len(self.data)



class MyDataset:
    def __init__(self, data):
        self.data_name = data
        # ['entity_emb.pt', 'entity_list.txt', 'item_list.txt', 'kg_final.txt', 'relation_emb.pt', 'relation_list.txt', 'test.txt', 'train.txt', 'train1.txt', 'user_list.txt', 'valid1.txt']
        self.data_path = {
            "entity_list": os.path.join(r"mine\data", data, 'entity_list.txt'),
            "relation_list": os.path.join(r"mine\data", data, 'relation_list.txt'),
            "item_list": os.path.join(r"mine\data", data, 'item_list.txt'),
            "user_list": os.path.join(r"mine\data", data, 'user_list.txt'),
            "train": os.path.join(r"mine\data", data, 'train.txt'),
            "test": os.path.join(r"mine\data", data, 'test.txt'),
            "valid": os.path.join(r"mine\data", data, 'valid1.txt'),
            "kg": os.path.join(r"mine\data", data, 'kg_final.txt')
        }
        
        self.user_list = pd.read_csv(self.data_path["user_list"], sep=' ')
        self.item_list = pd.read_csv(self.data_path["item_list"], sep=' ')
        
        self.num_user = len(self.user_list)
        self.num_item = len(self.item_list)
        
        
        
        self.user_items_dict = self._get_user_item_dict()
        self.uikg = self._to_uikg()
        self.neigbhors = self._get_neigbhors()
        
        
        self.train_set = self._to_label_with_negative_sampels()
    
    
    def _get_num(self):
        '''
        return: num_user, num_item, num_relation, num_entity
        '''
        return self.num_user, self.num_item, self.num_relation, self.num_entity
        
    def _get_user_item_dict(self):
        '''
        return: a dcit of user-item interface-> user:[item1, item2, ...]
        '''
        user_items = dict()
        with open(self.data_path["train"], 'r') as f:
            for line in f:
                l = line.strip().split()
                l = [int(item) for item in l]
                user = l[0]
                items = l[1:]
                user_items[user] = items
        return user_items
        
    def _to_uikg(self):
        if os.path.exists(os.path.join(r"mine\data", self.data_name, 'uikg.csv')):
            
            print("Loading uikg file!")
            
            uikg = pd.read_csv(os.path.join(r"mine\data", self.data_name, 'uikg.csv'), sep=' ')
            m = uikg.max(axis=0)
            self.num_entity = max(m["h"], m["t"]) + 1
            self.num_relation = m["r"] + 1
            
            print("entity num: {}, relation num: {}".format(self.num_entity, self.num_relation))
            
            return uikg
        
        print("Constructing uikg file!")
        
        kg = pd.read_csv(self.data_path["kg"], sep=' ', names=['h', 'r', 't'])
        
        m = kg.max(axis=0)
        max_en = max(m["h"], m["t"]) + 1
        max_rel = m["r"] + 1
        
        self.num_entity = max_en + self.num_user
        self.num_relation = max_rel + 1
        print("entity num: {}, relation num: {}".format(self.num_entity, self.num_relation))
        self.user_items = dict()
        
        new_h, new_r, new_t = [], [], []
        
        for user in tqdm(self.user_items_dict.keys()):
            for item in self.user_items_dict[user]:
                new_h.append(user + max_en)
                new_r.append(max_rel)
                new_t.append(item)
                
        new_data = {
            'h':new_h,
            'r':new_r,
            't':new_t
        }
        uikg = pd.concat([kg, pd.DataFrame(new_data)], axis=0)
        uikg.to_csv(os.path.join(r"mine\data", self.data_name, 'uikg.csv'), sep=' ', index=False)
        return uikg
    
    def _get_neigbhors(self):
        '''
        return: a dict of neighbors-> h:[(r,t), ...]
        '''
        print("Constructing neighbors!")
        if os.path.exists(os.path.join(r"mine\data", self.data_name, 'neighbors.json')):
            neighbors = json.load(open(os.path.join(r"mine\data", self.data_name, 'neighbors.json'), 'r'))
            return neighbors
        
        neighbors = {key:[] for key in range(self.num_entity)}
        for i in tqdm(range(len(self.uikg))):
            h, r, t = self.uikg.iloc[i][0], self.uikg.iloc[i][1], self.uikg.iloc[i][2]
            h, r, t = int(h), int(r), int(t)
            neighbors[h].append((r,t))
        try:
            json.dump(neighbors, open(os.path.join(r"mine\data", self.data_name, 'neighbors.json'), 'w'), indent=4)
        except:
            print("dict failed to save into json file!")
        finally:
            print("dict save into json file!")
        
        return neighbors
            
    def _to_label_with_negative_sampels(self, negative_samples=1):
        '''
        params: negative_samples -> int of number negative samples
        return: datasets -> class<Sets: torch.utils.data.Dataset> (users, items, labels)
        '''
        print("Constructing labels with negative sampling!")
        

        users, items, lables = [], [], []
        for user in self.user_items_dict.keys():
            users += [user] * len(self.user_items_dict[user] * (1+negative_samples))
            lables += [1] * len(self.user_items_dict[user])
            lables += [0] * len(self.user_items_dict[user] * negative_samples)
            
            items += self.user_items_dict[user]
            
            negative_set = []
            for _ in range(negative_samples * len(self.user_items_dict[user])):
                negative_item = random.randint(0, self.num_item - 1)
                while negative_item in self.user_items_dict[user] or negative_item in negative_set:
                    negative_item = random.randint(0, self.num_item - 1)
                negative_set.append(negative_item)
            
            items += negative_set

                
        # print(len(users), len(items), len(lables))
        data = pd.DataFrame({'user':users, 'item':items, 'label':lables})

        datasets = Sets(data)

        return datasets
    
    def get_set_with_label(self, types="test"):
        '''
        params: types -> str ("test", "valid")
        return: datasets -> class<Sets: torch.utils.data.Dataset> (users, items, labels)
        '''
        
        user_id = []
        item_id = []
        label = []
        
        with open(self.data_path[types], 'r') as f:
            for line in f:
                l = line.strip().split()
                l = [int(item) for item in l]
                user = l[0] + self.num_entity
                items = l[1:]
                user_id += [user] * len(items)
                item_id += items
                label += [1] * len(items)
        
        data = pd.DataFrame({'user':user_id, 'item':item_id, 'label':label})
        datasets = Sets(data)

        return datasets
