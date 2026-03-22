import os
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
import numpy as np
import torch_geometric
from Bio import SeqIO  # For parsing FASTA files
import torch_geometric.utils as pyg_utils
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Helper function to load the ESM data for a protein
def load_esm(esm_dir, protein_id):
    esm_path = os.path.join(esm_dir, f"{protein_id}_embedding.pt")
    esm_data = torch.load(esm_path)  # Assuming ESM data is stored in .pt files
    return esm_data


def load_lihua(lihua_dir, protein_id):
    # 生成边特征文件的路径
    lihua_path = os.path.join(lihua_dir, f"{protein_id}.npy")
    
    # 加载边特征
    lihua_features = np.load(lihua_path)  # 假设边特征以.npy格式存储
    
    # 将边特征转换为PyTorch张量
    lihua_features_tensor = torch.tensor(lihua_features, dtype=torch.float32)
    
    return lihua_features_tensor
    
def load_node(node_dir, protein_id):
    node_path = os.path.join(node_dir, f"{protein_id}_node_attr.pt")
    node_data = torch.load(node_path)  # Assuming ESM data is stored in .pt files
    return node_data
    
def load_edge_features(edge_features_dir, protein_id):
    # 生成边特征文件的路径
    edge_features_path = os.path.join(edge_features_dir, f"{protein_id}_attr.npy")
    
    # 加载边特征
    edge_features = np.load(edge_features_path)  # 假设边特征以.npy格式存储
    
    # 将边特征转换为PyTorch张量
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)
    
    return edge_features_tensor

def load_blosum():
    with open(Dataset_Path + 'BLOSUM62_dim23.txt', 'r') as f:
        result = {}
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result
    
    
def load_features(sequence_name, sequence, mean, std, blosum):
    # len(sequence) * 23
    blosum_matrix = np.array([blosum[i] for i in sequence])
    # len(sequence) * 71
    oneD_matrix = np.load(Dataset_Path + 'node_features/' + sequence_name + '.npy')
    # len(sequence) * 94
    feature_matrix = np.concatenate([blosum_matrix, oneD_matrix], axis=1)
    feature_matrix = (feature_matrix - mean) / std
    part1 = feature_matrix[:,0:20]
    part2 = feature_matrix[:,23:]
    # len(sequence) * 91
    feature_matrix = np.concatenate([part1,part2],axis=1)
    return feature_matrix
    

# Helper function to load the adjacency matrix for a protein
def load_adj_matrix(adj_dir, protein_id):

    adj_path = os.path.join(adj_dir, f"{protein_id}_adjacency.npy")
    
    adj_matrix = np.load(adj_path)  
    
    edge_index = pyg_utils.dense_to_sparse(torch.tensor(adj_matrix))[0]  
    
    edge_index = torch.unique(edge_index, dim=1)
    edge_index = edge_index[:, :edge_index.shape[1] // 2] 

    num_nodes = adj_matrix.shape[0] 
    deg = pyg_utils.degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    
    return edge_index, deg


def load_values():
    # (23,)
    blosum_mean = np.load(Dataset_Path + 'eSol_blosum_mean.npy')
    blosum_std = np.load(Dataset_Path + 'eSol_blosum_std.npy')

    # (71,)
    oneD_mean = np.load(Dataset_Path + 'eSol_oneD_mean.npy')
    oneD_std = np.load(Dataset_Path + 'eSol_oneD_std.npy')

    mean = np.concatenate([blosum_mean, oneD_mean])
    std = np.concatenate([blosum_std, oneD_std])

    return mean, std
    



def load_fasta(fasta_dir, protein_id):
    fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
    with open(fasta_path, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence = str(record.seq) 
            length = len(sequence) 
            return sequence, length 


def load_solubility_data(csv_file):
    df = pd.read_csv(csv_file)
    solubility_dict = dict(zip(df['gene'], df['solubility']))  
    return solubility_dict




class ProteinDataset(Dataset):
    def __init__(self, protein_id_file, node_dir,esm_dir, adj_dir, fasta_dir, solubility_csv,lihua_dir):
        super(ProteinDataset, self).__init__()
        self.protein_ids = self.load_protein_ids(protein_id_file)
        self.esm_dir = esm_dir
        self.node_dir = node_dir
        self.adj_dir = adj_dir
        self.fasta_dir = fasta_dir
        self.solubility_dict = load_solubility_data(solubility_csv)  
        self.blosum = load_blosum()
        self.edge_dir = adj_dir
        self.lihua_dir = lihua_dir
        self.mean, self.std = load_values()
        

    def load_protein_ids(self, protein_id_file):
        with open(protein_id_file, 'r') as f:
            return [line.strip() for line in f]



    def len(self):
        return len(self.protein_ids)



    def get(self, idx):
        protein_id = self.protein_ids[idx]

        esm_features = load_esm(self.esm_dir, protein_id).squeeze(0)
        esm_features = esm_features[:-2]
        
        
        node_attr_features = load_node(self.node_dir, protein_id).squeeze(0)
        
        lihua_features = load_lihua(self.lihua_dir,protein_id).squeeze(0)
        lihua_features = np.squeeze(lihua_features)
        sequence,length = load_fasta(self.fasta_dir, protein_id)
        
        sequence_feature = load_features(protein_id, sequence, self.mean, self.std, self.blosum)
        sequence_feature = torch.tensor(sequence_feature, dtype=torch.float)
        node_attr_features = node_attr_features.to(device)
        sequence_feature = sequence_feature.to(device)
        node_features = torch.cat([node_attr_features, sequence_feature], dim=-1)

        edge_index, deg = load_adj_matrix(self.adj_dir, protein_id)
        
        
        
        
        edge_features =load_edge_features(self.edge_dir,protein_id)

        solubility = torch.tensor([self.solubility_dict.get(protein_id, 0.0)], dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index, y=solubility,deg=deg,esm_features = esm_features,length = length,sequence =sequence,attr =edge_features,lihua = lihua_features)

        return data
        
        

from torch_geometric.data import DataLoader


'''

solubility_csv = "/home/inspur/marshall/autodl-tmp/gat/eSol_train.csv"
protein_id_file = "/home/inspur/marshall/autodl-tmp/gat/protein_id.txt"
esm_dir ="/home/inspur/marshall/autodl-tmp/results/"
adj_dir = "/home/inspur/marshall/autodl-tmp/at94/"
fasta_dir = "/home/inspur/marshall/autodl-tmp/gat/fasta/"
additional_features_dir = "/home/inspur/marshall/autodl-tmp/gat/node_features/"
node_dir = "/home/inspur/marshall/autodl-tmp/gat/attr/"
lihua_dir="/home/inspur/marshall/autodl-tmp/gat/total/overmaxshort_min/"

Dataset_Path ="/home/inspur/marshall/autodl-tmp/gat/"

def print_graph_dimensions(data_loader, num_graphs=100):
    for i, data in enumerate(data_loader):
        if i >= num_graphs:
            break
        
        print(f"Graph {i+1}:")
        print(f"  Node features (x): {data.x.shape}")
        print(f"  esm features (esm): {data.esm_features.shape}")
        print(f"  Edge index (edge_index): {data.edge_index.shape}")
        print(f"  Solubility label (y): {data.y.shape}")
        print(f"  lenth (lenth): {data.length.shape}")
        print(f"  lenth (lenth): {len(data.sequence)}")
        print(f"  Edge index (edge_index): {data.attr.shape}")
        print(f"  lihua: {data.lihua.shape}")
        print(f"  edge: {data.attr.shape}")
        print("-" * 30)

dataset = ProteinDataset(protein_id_file, node_dir, esm_dir, adj_dir, fasta_dir, solubility_csv,lihua_dir)

test_loader = DataLoader(dataset, batch_size=1, shuffle=True)  

print_graph_dimensions(test_loader)'''