from train_process import*
from DFusion_DFI import DFusion_DFI_model
#from model_txt import fusion_model
#from atom_singal_block import DFIFusionModel as fusion_model
import torch.nn as nn
from dataloader_part import create_data_loaders2
import random
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')  
device = "cuda" if torch.cuda.is_available() else'cpu'
def set_randam(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def main(file_name = './datasets/dfi_final.csv', flag = 1, save_name="best_DFI.ckpt"):

    early_step,final_score = 0,0
    train_loader, val_loader, test_loader = create_data_loaders2(
        file_name,  batch_size=108*4, flag=flag)
    model = DFusion_DFI_model().to(device)
    I_rate = 6e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=I_rate , weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
    n_epochs = 50
    for epoch in range(n_epochs):
        print("epoch:",epoch+1," / ", n_epochs)
        loss_function=torch.nn.CrossEntropyLoss()
        train_part(train_loader,optimizer,model,loss_function)
        metric = valid_part(val_loader,model)
        for key, value in metric.items():
            print(key, value)
        scheduler.step()
        early_step +=1
        if final_score < (metric['auc_roc']+metric["auc_pr"]+metric["f1"]):
            early_step = 0
            final_score = metric['auc_roc']+metric["auc_pr"]+metric["f1"]
            torch.save(model.state_dict(),"./save_model/"+save_name) #保存训练模型
            print('saving model with high auc plus aupr {:.5f}...'.format(final_score))
            # temp_metric = valid_part(test_loader,model)
            # for key, value in temp_metric.items():
            #    print(key, value)
            test_model = DFusion_DFI_model().to(device)
            test_model.load_state_dict(torch.load("./save_model/"+save_name))
            print("The validation set reached the maximum value. \n test:")
            test_metric = valid_part(test_loader, test_model)
            for key, value in test_metric.items():
                print(key, value)
        else:
            print("The validation set did not reach the maximum value.:")
            # temp_metric = valid_part(test_loader,model)
            # for key, value in temp_metric.items():
            #    print(key, value)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        print("\n")

set_randam(2025)
k = main(flag=1, save_name="DFsuion-DFI.ckpt")