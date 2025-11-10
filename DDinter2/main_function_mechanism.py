from train_process_mechanism import*
from DFusion_DFI import DFusion_DFI_model
#from txt_singal_block import fusion_model
#from model_atom_01 import DFIFusionModel as fusion_model
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
    datasets= deal_with_data_mechanism(df)
    train_X,train_Y,test_X,test_Y = train_test_data1(datasets)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    n_epochs = 15
    other_property=[]
    for i in range(5):
        print("fold:", i+1)
        loader_train,loader_test = get_dataloader(train_X[i],train_Y[i],test_X[i],test_Y[i],batch_size=64)
        model = DFusion_DFI_model(out_dim=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5 , weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16], gamma=0.1)
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
        #torch.save(model.state_dict(),"./save_model/"+str(i)+"Dfusion_DFI.ckpt") #保存训练模型
    return np.mean(np.array(other_property), axis=0),other_property

#DFIS_drugs_contain_descr_structure.xlsx
#DFIS_small_molecule_drugs.xlsx
set_randam(2025)
k, other_property= main('./datasets/DFIS_small_molecule_drugs.xlsx')
keys = ["acc","test_auc","test_aupr","test_f1","test_pre","test_recall"]
for i in range(6):
    print(keys[i],k[i])
print("var:",np.mean(np.var(np.array(other_property), axis=0))*100*100)