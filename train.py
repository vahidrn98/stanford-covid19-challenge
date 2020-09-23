
import math, json, gc, random, os, sys, time
import numpy as np, pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tqdm import tqdm

import models
import losses
from dataset import RNADataset
from utils import AverageMeter
from config import args

## Model
#get comp data
train = pd.read_json('../train.json', lines=True)
test = pd.read_json('../test.json', lines=True)
sample_sub = pd.read_csv("../sample_submission.csv")

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

train = train[train.signal_to_noise >= 1]


def train_epcoh(model, loader, optimizer, criterion, device, epoch):
    losses = AverageMeter()

    model.train()
    t = tqdm(loader)
    for i, d in enumerate(t):

        #print(d)

        X = d['seq'].to(device)
        y = d['target'].to(device)

        pred_y = model(X)

        loss = criterion(y, pred_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        losses.update(loss.item(), bs)

        t.set_description(f"Train E:{epoch} - Loss:{losses.avg:0.5f}")
    
    t.close()
    return losses.avg

def valid_epoch(model, loader, criterion, device, epoch):
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        t = tqdm(loader)
        for i, d in enumerate(t):

            X = d['seq'].to(device)
            y = d['target'].to(device)

            pred_y = model(X)
            
            #print(y.shape, pred_y.shape)
            
            loss = criterion(y, pred_y)

            bs = X.size(0)
            losses.update(loss.item(), bs)

            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.5f}")
        
    t.close()
    return losses.avg

def test_predic(model, loader, device):
    
    outputs_dict = {
        "ids" : [],
        "predicts" : []
    }
    
    model.eval()
    
    with torch.no_grad():
        t = tqdm(loader)
        for i, d in enumerate(t):
            X = d['seq'].float().to(device)
            ids = d['ids']
            
            outs = model(X).cpu().detach().numpy().tolist()
            
            outputs_dict['predicts'].extend(outs)
            outputs_dict['ids'].extend(ids)
            
    return outputs_dict

def main():

    # Setting seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    args.save_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    public_predictions = [] 
    public_ids         = []

    private_predictions = []
    private_ids         = []

    public_df = test.query("seq_length == 107").reset_index(drop=True)
    private_df = test.query("seq_length == 130").reset_index(drop=True)

    public_dataset = RNADataset(public_df)
    private_dataset = RNADataset(private_df)

    public_loader = DataLoader(public_dataset, batch_size=args.batch_size, shuffle=False)
    private_loader = DataLoader(private_dataset, batch_size=args.batch_size, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skf = KFold(args.n_folds, shuffle=True, random_state=seed)

    for i, (train_index, valid_index) in enumerate(skf.split(train, train['SN_filter'])):
        print("#"*20)
        print(f"##### Fold : {i}")

        args.fold = i

        train_df = train.iloc[train_index].reset_index(drop=True)
        valid_df = train.iloc[valid_index].reset_index(drop=True)
        
        #valid_df = valid_df[valid_df.SN_filter == 1].reset_index(drop=True)

        train_dataset = RNADataset(train_df)
        valid_dataset = RNADataset(valid_df)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        model = models.__dict__[args.network](args, pred_len=68)
        model = model.to(device)

        criterion = losses.__dict__[args.losses]()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_loss = 99999

        for epoch in range(args.epochs):

            train_loss = train_epcoh(model, train_loader, optimizer, criterion, device, epoch)
            valid_loss = valid_epoch(model, valid_loader, criterion, device, epoch)

            content = f"""
                {time.ctime()} \n
                Fold:{args.fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}, \n
                Train Loss:{train_loss:0.4f} - Valid Loss:{valid_loss:0.4f} \n
            """
            print(content)

            with open(f'{args.save_path}/log_{args.exp_name}.txt', 'a') as appender:
                appender.write(content + '\n')
            
            if valid_loss < best_loss:
                print(f"######### >>>>>>> Model Improved from {best_loss} -----> {valid_loss}")
                torch.save(model.state_dict(), os.path.join(args.save_path, f"fold-{args.fold}.bin"))
                best_loss = valid_loss
            
            torch.save(model.state_dict(), os.path.join(args.save_path, f"last-fold-{args.fold}.bin"))
            
        public_model = models.__dict__[args.network](args, pred_len=107).to(device)
        public_model.load_state_dict(torch.load(os.path.join(args.save_path, f"fold-{args.fold}.bin")))
        
        private_model = models.__dict__[args.network](args, pred_len=130).to(device)
        private_model.load_state_dict(torch.load(os.path.join(args.save_path, f"fold-{args.fold}.bin")))
        
        public_pred_dict = test_predic(public_model, public_loader, device)
        private_pred_dict = test_predic(private_model, private_loader, device)
        
        public_predictions.append(np.array(public_pred_dict["predicts"]).reshape(629 * 107 , 5))
        private_predictions.append(np.array(private_pred_dict["predicts"]).reshape(3005 * 130, 5))
        
        public_ids.append(public_pred_dict["ids"])
        private_ids.append(private_pred_dict["ids"])

    public_ids1 = [f"{id}_{i}" for id in public_ids[0] for i in range(107)]
    private_ids1 = [f"{id}_{i}" for id in private_ids[0] for i in range(130)]

    public_preds = np.mean(public_predictions, axis=0)
    private_preds = np.mean(private_predictions, axis=0)

    public_pred_df = pd.DataFrame(public_preds, columns=target_cols)
    public_pred_df["id_seqpos"] = public_ids1

    private_pred_df = pd.DataFrame(private_preds, columns=target_cols)
    private_pred_df["id_seqpos"] = private_ids1

    pred_sub_df = public_pred_df.append(private_pred_df)

    pred_sub_df.to_csv(os.path.join(args.save_path, f"{args.sub_name}_submission.csv"), index=False)

if __name__ == "__main__":
    main()
