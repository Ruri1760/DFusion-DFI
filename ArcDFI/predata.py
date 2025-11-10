import pandas as pd
import pickle
import torch
from rdkit import Chem
import numpy as np
from transformers import AutoModel, AutoTokenizer
device = "cuda:2" if torch.cuda.is_available() else'cpu'
chem_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", do_lower_case=False)
chem_model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").to(device)
def switch_dict(temp_dict, save_name):
    drug_LLM = {}
    over_size = []
    for i,(key,smile) in enumerate(temp_dict.items()):
        chem_intput = chem_tokenizer.batch_encode_plus([smile], add_special_tokens=True, padding=True)
        chem_ids = torch.tensor(chem_intput['input_ids']).to(device)
        tokens = chem_tokenizer.convert_ids_to_tokens(chem_ids[0].cpu().numpy())
        chem_mask = torch.tensor(chem_intput["attention_mask"]).to(device)
        if chem_ids.shape[1]>512:
            over_size.append(i)
            continue
        with torch.no_grad():
            chem_outputs = chem_model(input_ids = chem_ids, attention_mask = chem_mask)
        chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
        if smile not in drug_LLM:
            drug_LLM[key] = {"smiles_text_fea":chem_feature,
                               "smiles_token":tokens}
    print(over_size)

    with open("./pre_params/" + save_name, "wb") as p: 
        pickle.dump(drug_LLM, p)

def get_dict(file_name,column_name1,column_name2):
    df = pd.read_csv(file_name)
    temp_dict = {}
    for i in range(len(df)):
        id = df[column_name1][i]
        smiles = df[column_name2][i]
        if id not in temp_dict.keys() and not pd.isna(smiles):
            temp_dict[id]=smiles
    return temp_dict

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False,device=device):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2 
                ]) + [atom.GetIsAromatic()]

    if explicit_H:
        results = results + [atom.GetTotalNumHs()]
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results).to(device)

def get_atom_fea(smiles_dict, save_name):
    smiles_atom_fea = {}
    for key,smiles in smiles_dict.items():
        try:
            mol = Chem.MolFromSmiles(smiles) 
            n_features = [(atom_features(atom)) for atom in mol.GetAtoms()]
            n_features = torch.stack(n_features).to(device)
        
            n_sequece  = [ atom.GetSymbol() for atom in mol.GetAtoms()]
            smiles_atom_fea[key] = {"atom_seq": n_sequece,
                                "atom_fea": n_features}
        except:
            print("error:",key, smiles)
            smiles_atom_fea[key] = {"atom_seq": None,
                                "atom_fea": None}
        
    with open("./pre_params/"+save_name, "wb") as p: 
            pickle.dump(smiles_atom_fea, p)

drug_dict = get_dict("./datasets/dfi_final.csv","drugcompound_id","drugcompound_smiles") 
food_dict = get_dict("./datasets/dfi_final.csv","foodcompound_id","foodcompound_smiles")
get_atom_fea(drug_dict, "drug_all_atom_fea.pkl") 
get_atom_fea(drug_dict, "food_all_atom_fea.pkl") 
switch_dict(food_dict, "food_fea.pkl")
switch_dict(drug_dict, "drug_fea.pkl")
with open('./pre_params/drug_id2smiles.pkl', 'wb') as f:
    pickle.dump(drug_dict,f)
with open('./pre_params/food_id2smiles.pkl', 'wb') as f:
    pickle.dump(food_dict,f)
