import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from Bio import PDB

def parse_cif(cif_path):

    parser = PDB.MMCIFParser()
    structure = parser.get_structure("protein", cif_path)
    aa_coordinates = []
    atom_coords = {'N': [], 'CA': [], 'C': [], 'O': []}  

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  

                    for atom in ['N', 'CA', 'C', 'O']:
                        try:
                            atom_coords[atom].append(residue[atom].get_coord())
                        except KeyError:
                            atom_coords[atom].append(None)  

                    try:
                        ca_atom = residue['CA']
                        aa_coordinates.append(ca_atom.get_coord()) 
                    except KeyError:
                        continue  

    return np.array(aa_coordinates), atom_coords

def calculate_edge_features(coords, atom_coords, threshold=8.0):

    num_residues = len(coords)
    edges = []
    adjacency_matrix = np.zeros((num_residues, num_residues), dtype=np.uint8)

    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                edges.append((i, j, dist))
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1 

    return edges, adjacency_matrix

def generate_edge_vectors(edges, coords, atom_coords, max_sequence_distance=64):

    edge_features = []
    for i, j, dist in edges:

        rbf_features = []
        for r in range(1, 16):
            sigma_r = 1.5 ** (r - 1)
            rbf = np.exp(-dist**2 / (2 * sigma_r**2))  
            rbf_features.append(rbf)


        N_i, CA_i, C_i, O_i = atom_coords['N'][i], atom_coords['CA'][i], atom_coords['C'][i], atom_coords['O'][i]
        N_j, CA_j, C_j, O_j = atom_coords['N'][j], atom_coords['CA'][j], atom_coords['C'][j], atom_coords['O'][j]
        

        delta_N = N_j - N_i if N_i is not None and N_j is not None else np.zeros(3)
        delta_CA = CA_j - CA_i if CA_i is not None and CA_j is not None else np.zeros(3)
        delta_C = C_j - C_i if C_i is not None and C_j is not None else np.zeros(3)
        delta_O = O_j - O_i if O_i is not None and O_j is not None else np.zeros(3)
        

        relative_position_features = np.concatenate([delta_N, delta_CA, delta_C, delta_O])


        sequence_distance = np.abs(i - j)
        sequence_features = [0] * 66
        if sequence_distance <= max_sequence_distance:
            sequence_features[sequence_distance] = 1
        

        edge_vector = np.concatenate([rbf_features, relative_position_features, sequence_features])
        edge_features.append(edge_vector)

    return np.array(edge_features)

def standardize_features(edge_features):

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(edge_features)
    return standardized_features

def save_edge_features(protein_name, edge_features, save_dir):

    save_path = os.path.join(save_dir, f"{protein_name}_attr.npy")
    np.save(save_path, edge_features)
    print(f"Saved edge features for {protein_name} to {save_path}")

def save_adjacency_matrix(protein_name, adjacency_matrix, save_dir):

    save_path = os.path.join(save_dir, f"{protein_name}_adjacency.npy")
    np.save(save_path, adjacency_matrix)
    print(f"Saved adjacency matrix for {protein_name} to {save_path}")
    
def process_proteins(cif_dir, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for protein_name in os.listdir(cif_dir):
        protein_path = os.path.join(cif_dir, protein_name, "pred.model_idx_0.cif")
        if os.path.isfile(protein_path):
            print(f"Processing {protein_name}...")
            

            coords, atom_coords = parse_cif(protein_path)
            

            edges, adjacency_matrix = calculate_edge_features(coords, atom_coords, threshold=8.0)


            save_adjacency_matrix(protein_name, adjacency_matrix, save_dir)
            

            edge_features = generate_edge_vectors(edges, coords, atom_coords)
            

            print(edge_features.shape)

            save_edge_features(protein_name, edge_features, save_dir)


# 设定CIF文件夹和保存文件夹路径
cif_dir = ""# CIF文件夹路径
save_dir = "" # 生成的边向量保存路径


process_proteins(cif_dir, save_dir)
