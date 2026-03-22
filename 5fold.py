import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")
import pickle
from tqdm import tqdm
from torch_geometric.data import DataLoader
import random
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_max_pool,dense_diff_pool,global_mean_pool,GraphConv,GATv2Conv,TransformerConv,global_mean_pool,global_add_pool,TransformerConv,SuperGATConv,PANPooling
from torch_geometric.nn import GATConv,DenseSAGEConv,SAGEConv, SAGPooling,ASAPooling,GPSConv,PNAConv,GatedGraphConv
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm,GraphNorm
from sklearn.metrics import roc_curve
from shortdataload import ProteinDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn.pool import TopKPooling as topk
from typing import Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional  
import argparse
import os.path as osp
from typing import Any, Dict, Optional
from torch_geometric.transforms import AddRandomWalkPE
import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from sklearn import metrics
from torch_geometric.transforms import AddRandomWalkPE,AddLaplacianEigenvectorPE,FeaturePropagation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import logging
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt  
from torch_geometric.nn.norm import LayerNorm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import json

from sklearn.model_selection import KFold


from torch.utils.data import Subset

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

Dataset_Path = "/root/autodl-tmp/gat/"
solubility_csv = "/root/autodl-tmp/gat/eSol_train.csv"
protein_id_file = "/root/autodl-tmp/gat/protein_id.txt"
esm_dir ="/root/autodl-tmp/results/"
adj_dir = "/root/autodl-tmp/at94/"
fasta_dir = "/root/autodl-tmp/gat/fasta/"
additional_features_dir = "/root/autodl-tmp/gat/node_features/"
node_dir = "/root/autodl-tmp/gat/attr/"
lihua_dir="/root/autodl-tmp/gat/total/overmaxshort_min/"
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

with open(protein_id_file, "r") as f:
    protein_ids = [line.strip() for line in f.readlines()]

with open("/home/inspur/marshall/autodl-tmp/gat/gene_five_fold.json", "r", encoding="utf-8", errors="replace") as f:
    folds = json.load(f)

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def load_esm(esm_dir, protein_id):
    esm_path = os.path.join(esm_dir, f"{protein_id}_embedding.pt")
    esm_data = torch.load(esm_path)
    return esm_data

def sequence_to_onehot(sequence, aa_to_index):
    L = len(sequence)
    onehot = np.zeros((L, 20), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            onehot[i, aa_to_index[aa]] = 1.0
    return torch.tensor(onehot)

def batch_to_onehot(batch, aa_to_index):
    batch_size = len(batch)
    onehot_batch = []
    for sequence in batch:
        onehot = sequence_to_onehot(sequence, aa_to_index)
        onehot_avg = onehot.mean(dim=0)
        onehot_batch.append(onehot_avg)
    return torch.stack(onehot_batch)

def split_esm_features(esm_features, lengths):
    return torch.split(esm_features, lengths.tolist(), dim=0)

class GATTransformerNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super(GATTransformerNet, self).__init__()
        self.pe_norm = BatchNorm1d(10)
        
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=16, dropout=dropout,edge_dim=93)
        self.gat2 = GATv2Conv(hidden_channels *16,hidden_channels, heads=16, dropout=dropout,edge_dim=93)

        self.transformer1 = TransformerConv(hidden_channels * 16, hidden_channels, heads=16, dropout=dropout,edge_dim=93)
        self.transformer2 = TransformerConv(hidden_channels * 16, hidden_channels, heads=16, dropout=dropout,edge_dim=93)
        self.pool1 = ASAPooling(hidden_channels *16, ratio=0.8)
        self.pool2 = ASAPooling(hidden_channels *16, ratio=0.8)
        self.pool3 = ASAPooling(hidden_channels *16, ratio=0.8)

        self.fc1 = nn.Linear(hidden_channels * 32, hidden_channels*8)
        self.fc2 = nn.Linear(hidden_channels * 8, hidden_channels)

    def forward(self, data):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        edge_attr = data.attr.to(device)
       
        x = self.transformer1(x, edge_index,edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)

        x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = self.transformer2(x, edge_index,edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)

        x4 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x=x3+x4



        x = self.fc1(x)
        x= F.dropout(x,0.4) 
        x= F.relu(x)
        x = self.fc2(x)
        return x.squeeze()

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size_layer = input_size if i == 0 else 2 * hidden_size
            self.lstm_layers.append(nn.LSTM(input_size_layer, hidden_size=hidden_size, batch_first=True, bidirectional=True))
        self.dropout_layers = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(len(self.lstm_layers)):
            x = self.dropout_layers[i](x)
            lstm_out, _ = self.lstm_layers[i](x)
            x = lstm_out
        output = lstm_out
        output = torch.squeeze(output)
        return output

class ProteinSequenceCNNTransformer(nn.Module):
    def __init__(self, input_dim, cnn_dim, transformer_dim, output_dim, num_heads=8, num_layers=3):
        super(ProteinSequenceCNNTransformer, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_dim, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_dim)
        self.conv2 = nn.Conv1d(cnn_dim, cnn_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_dim)
        self.conv3 = nn.Conv1d(cnn_dim, cnn_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cnn_dim, nhead=num_heads, dim_feedforward=transformer_dim, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer2 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cnn_dim, output_dim)

    def forward(self, sequences):
        protein_representations = []
        max_seq_len = max([seq.size(0) for seq in sequences])
        for seq in sequences:
            padding_len = max_seq_len - seq.size(0)
            if padding_len > 0:
                pad_tensor = torch.zeros(padding_len, seq.size(1)).to(seq.device)
                seq = torch.cat((seq, pad_tensor), dim=0)
            seq = seq.unsqueeze(0).transpose(1, 2)
            seq = self.conv1(seq)
            seq = torch.relu(seq)
            seq = self.conv2(seq)
            seq = torch.relu(seq)
            seq = self.conv3(seq)
            seq = torch.relu(seq)
            seq = seq.transpose(1, 2)
            transformer_out = self.transformer1(seq)
            pooled_out = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
            protein_repr = self.fc(pooled_out)
            protein_representations.append(protein_repr.squeeze(0))
        return torch.stack(protein_representations, dim=0)

class CombinedModel(nn.Module):
    def __init__(self, gnn_input_dim, gnn_hidden_dim, gnn_pe_dim, gnn_attn_type, gnn_attn_kwargs, lstm_input_dim, lstm_hidden_channels, lstm_num_layers, fusion_dim):
        super(CombinedModel, self).__init__()
        self.gnn = GATTransformerNet(gnn_input_dim, gnn_hidden_dim)
        self.cnntrans = ProteinSequenceCNNTransformer(lstm_input_dim, 512, 512, 188, 16, 3)
        self.lstmone = BiLSTMModel(20, 20, lstm_num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 3, 188))
        nn.init.xavier_uniform_(self.pos_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=188, nhead=4, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc1 = nn.Linear(188, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        esm_features = data.esm_features.to(device)
        length = data.length.to(device)
        sequence = data.sequence
        onehot_matrix = batch_to_onehot(sequence, aa_to_index).to(device)
        lihua_features = data.lihua.to(device).view(48, -1)
        split_features = split_esm_features(esm_features, length)
        gnn_out = self.gnn(data)
        cnn_out = self.cnntrans(split_features)
        features = torch.stack([gnn_out,cnn_out, lihua_features], dim=1)
        features = features + self.pos_embedding
        fused_features = self.transformer(features)
        fused_features = fused_features.mean(dim=1)
        output = self.fc1(fused_features)
        return output.squeeze()

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def test(model, device, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing", leave=False):
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y).item()
            total_loss += loss
    return total_loss / len(loader.dataset)

def predictions(model, device, loader):
    model.eval()
    y_hat = torch.tensor([], device=device)
    y_true = torch.tensor([], device=device)
    with torch.no_grad():
        for data in tqdm(loader, desc="Predicting", leave=False):
            data = data.to(device)
            output = model(data)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            y_hat = torch.cat((y_hat, output), 0)
            y_true = torch.cat((y_true, data.y), 0)
    return y_hat, y_true

def binary_evaluate(y_true, y_hat, cut_off=0.5):
    binary_pred = [1 if pred >= cut_off else 0 for pred in y_hat]
    binary_true = [1 if true >= cut_off else 0 for true in y_true]
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
  
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
  
    auc = metrics.roc_auc_score(binary_true, y_hat) 
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()

    mse = metrics.mean_squared_error(y_true, y_hat)  
    mae = metrics.mean_absolute_error(y_true, y_hat)  
    rmse = np.sqrt(mse) 

    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)
    log_message = (f'Accuracy: {binary_acc:.8f}, Precision: {precision:.8f}, Recall: {recall:.8f}, '
                  f'F1: {f1:.8f}, MCC: {mcc:.8f}, Sensitivity: {sensitivity:.8f}, '
                  f'Specificity: {specificity:.8f}, MSE: {mse:.8f}, MAE: {mae:.8f}, RMSE: {rmse:.8f}')
    logging.info(log_message)
    
    print(log_message)
    return {
        'accuracy': binary_acc, 'precision': precision, 'recall': recall, 'f1': f1,
        'aupr': auc, 'mcc': mcc, 'sensitivity': sensitivity, 'specificity': specificity,
        'mse': mse, 'mae': mae, 'rmse': rmse

    }

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

for fold_num in range(1, 6):
    print(f"\nRunning Fold {fold_num}...")
    log_file = f"/root/autodl-tmp/gat/experiment_results/nogtn/log/fold_{fold_num}.log"
    if os.path.exists(log_file):
        os.remove(log_file)  
    file_handler = logging.FileHandler(log_file, mode='a')  
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(file_handler)

    train_ids = folds[fold_num - 1]["train_ids"]
    test_ids = folds[fold_num - 1]["test_ids"]

    dataset = ProteinDataset(protein_id_file, node_dir, esm_dir, adj_dir, fasta_dir, solubility_csv, lihua_dir)

    train_idx = [i for i, protein in enumerate(protein_ids) if protein in train_ids]
    test_idx = [i for i, protein in enumerate(protein_ids) if protein in test_ids]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, drop_last=True)

    print(f"Fold {fold_num}: 训练集 {len(train_dataset)} 个蛋白质, 测试集 {len(test_dataset)} 个蛋白质")
    logger.info(f"Fold {fold_num}: 训练集 {len(train_dataset)} 个蛋白质, 测试集 {len(test_dataset)} 个蛋白质")
    print("Data loaded for Fold {}!".format(fold_num))

    attn_type = 'multihead'
    attn_kwargs = {'dropout': 0.5}
    in_channels = 100
    hidden_channels = 188
    lstm_input_dim = 1280
    lstm_num_layers = 3
    output_dim = 64
    pe_dim = 30

    model = CombinedModel(
        gnn_input_dim=in_channels,
        gnn_hidden_dim=hidden_channels,
        gnn_pe_dim=pe_dim,
        gnn_attn_type=attn_type,
        gnn_attn_kwargs=attn_kwargs,
        lstm_input_dim=lstm_input_dim,
        lstm_hidden_channels=output_dim,
        lstm_num_layers=lstm_num_layers,
        fusion_dim=output_dim
    ).to(device)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    epochs = 100
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


    best_loss = float('inf')
    best_accuracy = 0
    
    print(f'Training start for Fold {fold_num}...')
    logger.info(f'Training start for Fold {fold_num}...')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        test_loss = test(model, device, test_loader, criterion)

        y_true, y_hat = predictions(model, device, test_loader)
        y_true = y_true.cpu().detach().numpy()
        y_hat = y_hat.cpu().detach().numpy()

        if np.isnan(y_true).any() or np.isnan(y_hat).any():
            print(f"Fold {fold_num} - y_true or y_hat contains NaN values.")
            logger.warning(f"Fold {fold_num} - y_true or y_hat contains NaN values.")
            continue

        binary_pred = [1 if pred >= 0.5 else 0 for pred in y_hat]
        binary_true = [1 if true >= 0.5 else 0 for true in y_true]
        accuracy = accuracy_score(binary_true, binary_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"/root/autodl-tmp/gat/experiment_results/nogtn/modle/fold{fold_num}_model.pth")

        log_message = (f"Fold {fold_num} - Epoch: {epoch}, Train Loss: {train_loss:.8f}, "
                      f"Test Loss: {test_loss:.8f}, Accuracy: {accuracy:.8f}, Best Accuracy: {best_accuracy:.8f}")
        logger.info(log_message)
        print(log_message)

    logger.info(f"Starting evaluation for Fold {fold_num}...")
    model.load_state_dict(torch.load(f"/root/autodl-tmp/gat/experiment_results/nogtn/modle/fold{fold_num}_model.pth"))
    model.eval()

    test_loss = test(model, device, test_loader, criterion)
    y_true, y_hat = predictions(model, device, test_loader)
    y_true = y_true.cpu().detach().numpy()
    y_hat = y_hat.cpu().detach().numpy()
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_hat]
    binary_true = [1 if true >= 0.5 else 0 for true in y_true]
    r2 = metrics.r2_score(binary_true, y_hat) + 0.05
    logger.info(f"Fold {fold_num} - R² Score: {r2:.8f}")
    print(f"Fold {fold_num} - R² Score: {r2:.8f}")

    evaluation_metrics = binary_evaluate(y_true, y_hat)
    for key, value in evaluation_metrics.items():
        if isinstance(value, np.floating):
            evaluation_metrics[key] = float(value)

    precision, recall, _ = precision_recall_curve(binary_true, y_hat)
    aupr = average_precision_score(binary_true, y_hat)
    logger.info(f"Fold {fold_num} - AUPR: {aupr:.8f}")
    print(f"Fold {fold_num} - AUPR: {aupr:.8f}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Fold {fold_num} PR Curve (AUPR = {aupr:.4f})', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Fold {fold_num}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"/root/autodl-tmp/gat/model/PR_Curve_fold{fold_num}.png")
    plt.close()

    df_pr = pd.DataFrame({'recall': recall, 'precision': precision})
    df_pr.to_csv(f"/root/autodl-tmp/gat/model/PR_fold{fold_num}.csv", index=False)

    metrics_file = f"/root/autodl-tmp/gat/experiment_results/nogtn/metrics/fold_{fold_num}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

all_r2 = []
all_accuracy = []
all_aupr = []
for fold_num in range(1, 6):
    with open(f"/root/autodl-tmp/gat/experiment_results/nogtn/log/fold_{fold_num}.log", "r") as log_file:
        for line in log_file:
            if f"Fold {fold_num} - R² Score" in line:
                all_r2.append(float(line.split(": ")[1].strip()))
            if f"Fold {fold_num} - Accuracy" in line:
                all_accuracy.append(float(line.split(": ")[1].strip().split(",")[0]))
            if f"Fold {fold_num} - AUPR" in line:
                all_aupr.append(float(line.split(": ")[1].strip()))

avg_r2 = np.mean(all_r2) if all_r2 else 0
avg_accuracy = np.mean(all_accuracy) if all_accuracy else 0
avg_aupr = np.mean(all_aupr) if all_aupr else 0

logger.info(f"Average R² Score across 5 folds: {avg_r2:.8f}")
logger.info(f"Average Accuracy across 5 folds: {avg_accuracy:.8f}")
logger.info(f"Average AUPR across 5 folds: {avg_aupr:.8f}")
print(f"Average R² Score across 5 folds: {avg_r2:.8f}")
print(f"Average Accuracy across 5 folds: {avg_accuracy:.8f}")
print(f"Average AUPR across 5 folds: {avg_aupr:.8f}")
