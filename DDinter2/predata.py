import pandas as pd
import pickle
import torch
from rdkit import Chem
import numpy as np
from transformers import AutoModel, AutoTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_text_fea(input_text):
    model_dir = "NovaSearch/stella_en_1.5B_v5"

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    with torch.no_grad():
        input_data = tokenizer(input_text, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = model(**input_data)[0]

    last_hidden_np = last_hidden_state[0].cpu().numpy()  # (seq_len, hidden_size)

    tokens = tokenizer.convert_ids_to_tokens(input_data["input_ids"][0].cpu().numpy())
    if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    return {"text_fea": last_hidden_np,
            "text_token": tokens}

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

df  = pd.read_excel('./datasets/DFIS_small_molecule_drugs.xlsx')
df2 = pd.read_csv("./datasets/drug_data.csv")
unique_drug_ids = sorted(df['Drug ID'].unique())
unique_food_ids = sorted(df['Food name'].unique())

drug_id2idx = {drug_id: idx for idx, drug_id in enumerate(unique_drug_ids)}
max_drug_idx = len(unique_drug_ids) - 1 if unique_drug_ids else -1  # 药物最大编号

food_id2idx = {
    food_id: (max_drug_idx + 1 + idx) 
    for idx, food_id in enumerate(unique_food_ids)
}

id2idx = {** drug_id2idx, **food_id2idx}
idx2id = {v: k for k, v in id2idx.items()}

with open('./pre_params/drug_food_ID.pkl', 'wb') as f:
    pickle.dump(id2idx,f)

drug_smiles = {}
for i in range(len(df)):
    name = df2['ID'][i]
    smiles = df2['Canonical SMILES'][i]
    sequences = df2["Sequences"][i]
    if not pd.isna(smiles) and smiles != 'None':
        drug_smiles[name] = smiles

with open('./pre_params/drug_id2smiles.pkl', 'wb') as f:
    pickle.dump(drug_smiles,f)

size = [0, 0, 0, 0]
smiles_atom_fea = {}
for key,smiles in drug_smiles.items():
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
        
with open("./pre_params/drug_all_atom_fea.pkl", "wb") as p: 
        pickle.dump(smiles_atom_fea, p)

drug_df = pd.read_csv("./datasets/drug_data.csv")
drug_descr = {}
for i in range(len(drug_df)):
    name = drug_df["ID"][i]
    description = drug_df["Description"][i]
    if not pd.isnull(description):
        drug_descr[name] = get_text_fea(description)

with open("./pre_params/drug_txt_fea.pkl", "wb") as p: 
        pickle.dump(drug_descr, p) 

food_df = pd.read_excel("./datasets/food_descr_manage.xlsx")
food_dict = {}
for i in range(len(food_df)):
    name = food_df["food_name"][i]
    descr = food_df["descr and manage"][i]
    if not pd.isnull(descr):
        food_dict[name] = get_text_fea(descr)

with open("./pre_params/food_txt_fea.pkl", "wb") as p: 
        pickle.dump(food_dict, p) 