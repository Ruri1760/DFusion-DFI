import torch
import numpy as np
from dataloader_part import*
from tqdm import tqdm
from sklearn.metrics import accuracy_score,roc_auc_score, average_precision_score,f1_score,recall_score,precision_score


def train_part(train_loader,optimizer,model,loss_function):
    loss_record = []
    true_train_num,total = 0,0
    model.train()
    train_pbar = tqdm(train_loader, position=0, leave=True)
    for data in train_pbar :
        labels_input = data["label"].to(torch.long).to(device) 
        train_output = model(data)
        #---------------------------------------------------------#
        loss =loss_function(train_output, labels_input).to(device)
        #---------------------------------------------------------#
    
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

        loss_record.append(loss.detach().item()) 
        with torch.no_grad():
            _, predicted = torch.max(train_output, 1)
            correct = (predicted == labels_input).sum().item()
            true_train_num += correct
            total +=len(predicted)
            train_pbar.set_postfix({'loss':loss.detach().item()}) 
    mean_train_loss = sum(loss_record) / len(loss_record)
    print("mean_loss:",mean_train_loss,"mean_acc:",true_train_num/total,"true_num:",true_train_num)
    torch.cuda.empty_cache()
def valid_part(test_loader,model):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    true_test_num,total = 0,0
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        
        for batch in test_loader:
            labels = batch['label'].to(device)
            test_outputs= model(batch)
            probs = softmax(test_outputs)
            _, preds = torch.max(test_outputs, 1)  
            correct = (preds == labels).sum().item()
            true_test_num += correct
            total +=len(preds)

            all_labels.append(labels.cpu())
            all_probs.append(probs[:, 1].cpu())  
            all_preds.append(preds.cpu())
        
        #print("mean_acc:",true_test_num/total,"true_num:",true_test_num)
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()  
    y_pred = torch.cat(all_preds).numpy()  
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob) 
    torch.cuda.empty_cache()
    return {
        'acc': acc,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr, 
        'f1': f1,
        'pre': precision,
        'recall': recall,
    }
