from train_process import*
from DFusion_DFI import DFusion_DFI_model
import random
def set_randam(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(file_name):
    df = pd.read_excel(file_name)
    pos_dataset,neg_dataset = deal_with_data(df)
    train_pos_X,train_pos_Y,test_pos_X,test_pos_Y = train_test_data1(pos_dataset)
    train_neg_X,train_neg_Y,test_neg_X,test_neg_Y = train_test_data1(neg_dataset)
    n_epochs = 15
    other_property=[]
    for i in range(5):
        print("5fold:",i+1)
        loader_train,loader_test = get_dataloader(np.concatenate((train_pos_X[i], train_neg_X[i]), axis=0),
                                                  np.concatenate((train_pos_Y[i], train_neg_Y[i]), axis=0),
                                                  np.concatenate((test_pos_X[i], test_neg_X[i]), axis=0),
                                                  np.concatenate((test_pos_Y[i], test_neg_Y[i]), axis=0),
                                                  batch_size=64)
        model = DFusion_DFI_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5 , weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[13], gamma=0.1)
        for epoch in range(n_epochs):
            print("epoch:",epoch+1)
            loss_function=torch.nn.CrossEntropyLoss()
            train_part(loader_train,optimizer,model,loss_function)
            scheduler.step()
            model.eval()
            temp_metric = valid_part(loader_test,model)
            for key, value in temp_metric.items():
                print(key, value)
            print("\n")
        other_property.append(list(temp_metric.values()))
        torch.save(model.state_dict(),"./save_model/"+str(i)+"Dfusion_DFI_cross_change.ckpt") #保存训练模型
    return np.mean(np.array(other_property), axis=0),other_property

#DFIS_drugs_contain_descr_structure.xlsx
#DFIS_small_molecule_drugs.xlsx
set_randam(2025)
k, other_property= main('./datasets/DFIS_small_molecule_drugs.xlsx')
keys = ["acc","test_auc","test_aupr","test_f1","test_pre","test_recall"]
for i in range(6):
    print(keys[i],k[i])
print("var:",np.mean(np.var(np.array(other_property), axis=0))*100*100)