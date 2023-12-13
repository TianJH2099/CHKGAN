import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, ndcg_score


class BertSelfAttention(nn.Module):
    def __init__(self, dim, head_nums=2):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = head_nums
        self.attention_head_size = int(dim / head_nums)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(dim, self.all_head_size)
        self.key = nn.Linear(dim, self.all_head_size)
        self.value = nn.Linear(dim, self.all_head_size)

        self.dense = nn.Linear(dim, dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)                             # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)                                 # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)                             # [Batch_size x Seq_length x Hidden_size]
        
        query_layer = self.transpose_for_scores(mixed_query_layer)                # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)                    # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)                # [Batch_size x Num_of_heads x Seq_length x Head_size]

        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)                    # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs, value_layer)                # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()            # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)              # [Batch_size x Seq_length x Hidden_size]
        
        output =  self.dense(context_layer)
        
        return output

class CHKGAT(nn.Module):
    def __init__(self, args, num_entity, num_relation, num_item):
        super(CHKGAT, self).__init__()
        self.threshold = args.threshold
        self.device = args.device
        self.label_smoothing = args.label_smoothing
        self.agg = args.aggregator
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.num_item = num_item
        self.dim = args.dim
        self.threshold = args.threshold
        
        self.entity_embed = nn.Embedding(num_entity, args.dim)
        self.relation_embed = nn.Embedding(num_relation, args.dim)
        # self.W_r = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (num_relation, dim, dim, dim)), dtype=torch.float, device="cpu", requires_grad=True))
        self.l = nn.Linear(args.dim, 2*args.dim, bias=True)
        
        self.self_attention = BertSelfAttention(args.dim, args.head_nums)
        self.dropout = nn.Dropout(args.dropout)
        self.sigmoid = nn.Sigmoid()
        
        self.loss = nn.BCELoss()
        
        if args.pretrain_embed:
            self.load_pretrained(args)
        
    def load_pretrained(self, args):
        if os.path.exists(os.path.join(r"mine\data", args.data_name, "pretrain\entity_emb.pt")) and os.path.exists(os.path.join(r"mine\data", args.data_name,"pretrain\relation_emb.pt")):
            
            pretrained_en = torch.load(os.path.join(r"mine\data", args.data_name, "pretrain\entity_emb.pt"))
            pretrained_rel = torch.load(os.path.join(r"mine\data", args.data_name,"pretrain\relation_emb.pt"))
            print("loading pretrained embedding of entities and relations!")
            pre_num_en = pretrained_en.shape[0]
            pre_num_rel = pretrained_rel.shape[0]
            
            if pre_num_en >= self.entity_embed.weight.data.shape[0]:
                self.entity_embed.weight.data = pretrained_en[:self.num_entity]
            else:
                self.entity_embed.weight.data[:pre_num_en] = pretrained_en
            
            if pre_num_rel >= self.relation_embed.weight.data.shape[0]:
                self.relation_embed.weight.data = pretrained_rel[:self.num_relation]
            else:
                self.relation_embed.weight.data[:pre_num_rel] = pretrained_rel
            
            print("loading pretrained embedding successful!")
        
        else:
            print("no pretrained embedding!")
    
    def aggregator(self, neighbors):
        '''
        param: neighbors -> dict{entity: [neighbors]}, [neighbors]-> [(r, t), ...]
        '''
        temp = torch.zeros(self.num_entity, self.dim).to(self.device)
        for entity in tqdm(neighbors.keys(), desc="aggreation entity information!"):
            hete_r = []
            entity = int(entity)
            en_embed = self.entity_embed(torch.tensor(entity).to(self.device))    # (dim)
            tail_embed = [[] for _ in range(self.num_relation)]
            for (r, t) in neighbors[str(entity)]:
                tail_embed[r].append(self.entity_embed(torch.tensor(t).to(self.device)))
            
            for i in range(self.num_relation):
                if len(tail_embed[i]) == 0:
                    hete_r.append(torch.zeros(self.dim).to(self.device))
                else:
                    rel = self.relation_embed(torch.tensor(i).to(self.device))                    # (dim)
                    tail = torch.stack(tail_embed[i])               # (n, dim)
                    
                    # relation aggreative calculating
                    prefer = torch.concat((en_embed, rel))            # (2*dim)
                    t_l = self.l(tail)                              # (n, 2*dim)
                    t_l = self.dropout(t_l)                         # (n, 2*dim)
                    
                    w = torch.matmul(prefer, t_l.transpose(1, 0))   # (n)
                    w = F.softmax(w, dim=0)
                    agg = torch.matmul(w, tail)                     # (dim)
                    hete_r.append(agg)
            hete_r = torch.stack(hete_r).to(self.device)            # (num_relation, dim)
            hete_r = hete_r.unsqueeze(0)                            # (1, num_relation, dim)
            out = self.self_attention(hete_r)                       # (1, num_relation, dim)
            out = out.squeeze(0)                                    # (num_relation, dim)
            out = out.mean(dim=0) # (dim)
            
            if self.agg == 'sum':
                temp[entity] += out
            elif self.agg == 'replace':
                stemp[entity] = out
            elif self.agg == 'mean':
                out = (out + en_embed) / 2
                temp[entity] = out
        
        self.entity_embed.weight.data = temp

    def forward(self, users, items):
        '''
        params: users, items -> torch<tensor> of shape (batch_size)
        return: scores -> torch<tensor> of shape (batch_size)
        return: ranking_socres -> torch<tensor> of shape (batch_size, num_item)
        '''
        # Embedding lookup for user and item
        batch_size = users.shape[0]

        user_embed = self.entity_embed(users) 
        item_embed = self.entity_embed(items)

        
        buy = self.relation_embed(torch.LongTensor([self.num_relation-1] * batch_size).to(self.device))
        
        all_items = self.entity_embed(torch.arange(self.num_item).to(self.device)) # (num_item, dim)
        
        # Compute scores
        scores = torch.matmul(user_embed, item_embed.T) # (batch_size, batch_size)
        scores = torch.diagonal(scores) # (batch_size)
        
        distance = torch.norm(user_embed + buy - item_embed, p=1, dim=1) # (batch_size)
        
        total_scores = distance + scores # (batch_size)
        predict = self.sigmoid(total_scores)
        
        # Cumpute ranking scores
        ranking_scores = torch.matmul(user_embed, all_items.T) # (batch_size, num_item)
        
        user_embed = user_embed.unsqueeze(1).repeat(1, self.num_item, 1) # (batch_size, num_item, dim)
        buy = buy.unsqueeze(1).repeat(1, self.num_item, 1) # (batch_size, num_item, dim)
        all_items = all_items.unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, num_item, dim)
        
        ranking_distance = torch.norm(user_embed + buy - all_items, p=1, dim=-1) # (batch_size, num_item)

        total_ranking_scores = ranking_distance + ranking_scores # (batch_size, num_item)
        ranking_predict = self.sigmoid(total_ranking_scores)
        
        # predict = (predict > self.threshold).float()
        # ranking_predict = (ranking_predict > self.threshold).float()

        return predict, ranking_predict
    
    def evaluate(self, users, items, labels):
        '''
        params: users, items, labels -> torch<tensor> of shape (batch_size)
        return: percision, recall, f1 -> float
        return: ndcg -> dict{NDCG5, NDCG10, NDCG20, NDCG50, NDCG100} -> float
        '''
        batch_size = labels.shape[0]
        
        predict, ranking_predict = self.forward(users, items)
        
        ranking_labels = F.one_hot(items, self.num_item)
        
        ndcg = dict()
        
        predict = (predict > self.threshold).float()
        predict = predict.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        ranking_predict = ranking_predict.detach().cpu().numpy()
        ranking_labels = ranking_labels.detach().cpu().numpy()

        precision = precision_score(labels, predict, average='binary')
        recall = recall_score(labels, predict, average='binary')
        f1 = f1_score(labels, predict, average='binary')
        
        ndcg['NDCG5'] = ndcg_score(ranking_labels, ranking_predict, k=5)
        ndcg['NDCG10'] = ndcg_score(ranking_labels, ranking_predict, k=10)
        ndcg['NDCG20'] = ndcg_score(ranking_labels, ranking_predict, k=20)
        ndcg['NDCG50'] = ndcg_score(ranking_labels, ranking_predict, k=50)
        ndcg['NDCG50'] = ndcg_score(ranking_labels, ranking_predict, k=100)
        
        return dict({'precision':precision, 'recall':recall, 'f1':f1, 'ndcg':ndcg})