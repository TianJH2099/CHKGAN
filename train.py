import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import logging


def set_log(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # build file handler and set level
    file_handler = logging.FileHandler(args.log_path + '\\' + "model_on_{}.txt".format(args.data_name))
    file_handler.setLevel(logging.DEBUG)
    
    # set log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 将处理程序添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def train(args, model, my_dataset):

    logger = set_log(args)
    
    if args.optim ==  'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    
    # ---------------valid and test datasets
    # valid_set = my_dataset.get_set_with_label('valid')
    # test_set = my_dataset.get_set_with_label('test')
    
    # valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    
    neighbors = my_dataset._get_neigbhors()
    
    
    for i in range(args.epoch):
        if i == 0:
            dataset = my_dataset.train_set
            
        else:
            dataset = my_dataset._to_label_with_negative_sampels()
        
        train_size = int(args.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        
        
        max_eval = 0;
        train_loss = 0;
        
        model.train()
        if (i-1) % args.agg_step == 0:
            for _ in range(args.num_agg):
                    model.aggregator(neighbors)
        for users, items, labels in tqdm(train_loader, desc='Training CF!'):
            optimizer.zero_grad() 
            
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            predict, ranking_predict = model.forward(users, items)
            ranking_labels = F.one_hot(items, my_dataset.num_item)
            # labels smoothing
            ranking_labels = ranking_labels * (1-args.label_smoothing) + args.label_smoothing * torch.ones_like(ranking_labels)
            labels = labels * (1-args.label_smoothing) + args.label_smoothing * torch.ones_like(labels)

            loss = model.loss(predict, labels) + model.loss(ranking_predict, ranking_labels)
            
            train_loss += loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            
        logger.info("epoch: {}, loss: {}".format(i+1, train_loss))
        
        if (i+1) % args.log_step == 0:
            model.eval()
            with torch.no_grad():
                flag = False
                precision, recall, f1, ndcg = 0, 0, 0, 0
                total = 1;
                for users, items, labels in tqdm(valid_loader):
                    
                    users, items, labels = users.to(device), items.to(device), labels.to(device)

                    eval = model.evaluate(users, items, labels)
                    
                    temp_ndcg = eval['ndcg']
                    max_key = max(temp_ndcg, key=temp_ndcg.get)
                    ndcg += temp_ndcg[max_key]
                    
                    precision += eval['precision']
                    recall += eval['recall']
                    f1 += eval['f1']

                    total += 1
                
                precision /= total
                recall /= total
                f1 /= total
                ndcg /= total
                
                if ndcg > max_eval:
                    max_eval = ndcg
                    flag = True
                
                logger.info("evaluate: percision -> {:.4f}, recall -> {:.4f}, f1 -> {:.4f} , best NDCG@N -> {:4f}".format(precision, recall, f1, ndcg))
                    
                if (not os.path.exists(args.model_path + '\\' + "{}_checkpoint.pth".format(args.data_name))) or flag:
                    checkpoint = model.state_dict()
                    torch.save(checkpoint, os.path.join(r"mine\checkpoints", "{}_checkpoint.pth".format(args.data_name)))
        
    checkpoint = model.state_dict()
    torch.save(checkpoint, os.path.join(r"mine\checkpoints", "{}_{}epoch.pth".format(args.data_name, args.epoch)))