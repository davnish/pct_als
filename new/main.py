import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import pct
from data import Dales
import time
from sklearn.metrics import accuracy_score
import numpy as np
torch.manual_seed(42)

# Hyperparameter----

grid_size = 20 #The size of the grid from 500mx500m
points_taken = 2048 #points taken per each grid
batch_size = 1
lr = 1e-2
epoch = 30
batch_eval_inter = 100
eval_train_test = 10
n_embd = 256
n_heads = 4
n_layers = 1
step_size = 20 # Reduction of Learning at how many epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_test = 10

# ------------------

# Splitting the data
_dales = Dales(grid_size, points_taken)
train_dataset, test_dataset = random_split(_dales, [0.9, 0.1])

# Loading the data
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# Initialize the model
model = pct(n_embd, n_heads, n_layers)
model = model.to(device)

# loss ,Optimizer, Scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = 0.1)


#Training the model
def train_loop(i, see_batch_loss = False):
    model.train()
    total_loss = 0
    y_true = []
    y_preds = []
    for batch, (data, label) in enumerate(train_loader):
        data , label = data.to(device), label.to(device)


        logits = model(data)


        optimizer.zero_grad()


        loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.view(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        preds = logits.max(dim = -1)[1].view(-1)

        y_true.extend(label.view(-1).cpu().tolist())
        y_preds.extend(preds.detach().cpu().tolist())
        
        if see_batch_loss:
            if batch%batch_eval_inter == 0:
                print(f'Batch_Loss_{batch} : {loss.item()}')

    if i%eval_train_test==0:
        val_loss, val_acc = test_loop(test_loader)
        print(f'Epoch {i+1}: train_loss: {(total_loss/len(train_loader)):.4f} | train_acc: {(accuracy_score(y_true, y_preds)):.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | lr: {scheduler.get_last_lr()}')
        


def test_loop(loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        y_true = []
        y_preds = []
        for data, label in loader:
            data , label = data.to(device), label.to(device)
            logits = model(data)


            loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.view(-1))
            
            total_loss+=loss.item()
            preds = logits.max(dim = -1)[1].view(-1)
            
            y_true.extend(label.view(-1).cpu().tolist())
            y_preds.extend(preds.detach().cpu().tolist())

    return total_loss/len(loader), accuracy_score(y_true, y_preds)
    # print(f'val_loss: {total_loss/len(test_loader)}, val_acc: {accuracy_score(y_true, y_preds)}')  

if __name__ == '__main__':
    print(f'{n_embd = }, {n_layers = }, {n_heads = }, {batch_size = }, {lr = }')
    start = time.time()
    for epoch in range(epoch): 
        train_loop(epoch)
        scheduler.step()

        # break
        
    end = time.time()

    print(f'Total_time: {end-start}')

