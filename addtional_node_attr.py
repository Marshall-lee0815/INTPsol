# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from Bio import PDB


def calculate_distance(atom1, atom2):
    return torch.norm(atom1 - atom2)


def calculate_angle(A, B, C):
    AB = A - B
    BC = C - B
    cosine_angle = torch.dot(AB, BC) / (torch.norm(AB) * torch.norm(BC))
    angle = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0)) 
    return torch.mul(angle, 180.0 / torch.pi) 


def calculate_dihedral_angle(A, B, C, D):
    AB = A - B
    BC = B - C
    CD = C - D
    
    n1 = torch.cross(AB, BC)
    n2 = torch.cross(BC, CD)
    
    n1 = n1 / torch.norm(n1)
    n2 = n2 / torch.norm(n2)
    
    m1 = torch.cross(n1, BC)
    m2 = torch.cross(n2, BC)
    
    x = torch.dot(n1, n2)
    y = torch.dot(m1, m2)
    dihedral_angle = torch.atan2(y, x)
    
    return torch.mul(dihedral_angle, 180.0 / torch.pi)


def process_protein_structure(base_dir, save_dir, device, distance_threshold=8.0):
    all_node_features = []

    for protein_dir in os.listdir(base_dir):
        protein_path = os.path.join(base_dir, protein_dir)
        if os.path.isdir(protein_path):
            cif_path = os.path.join(protein_path, "pred.model_idx_1.cif")
            
            if os.path.exists(cif_path):

                parser = PDB.MMCIFParser()
                structure = parser.get_structure(protein_dir, cif_path)
                

                atom_coordinates = []
                atom_ids = []
                
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                atom_coordinates.append(atom.get_coord())
                                atom_ids.append(atom.get_name())  
                
                atom_coordinates = np.array(atom_coordinates)
                atom_coordinates = torch.tensor(atom_coordinates, dtype=torch.float32, device=device)
                

                c_alpha_coords = []
                c_coords = []
                n_coords = []
                
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.has_id("CA"):  # Cα原子
                                c_alpha_coords.append(residue["CA"].get_coord())
                            if residue.has_id("C"):  # C原子
                                c_coords.append(residue["C"].get_coord())
                            if residue.has_id("N"):  # N原子
                                n_coords.append(residue["N"].get_coord())
                
                c_alpha_coords = torch.tensor(c_alpha_coords, dtype=torch.float32, device=device)
                c_coords = torch.tensor(c_coords, dtype=torch.float32, device=device)
                n_coords = torch.tensor(n_coords, dtype=torch.float32, device=device)


                num_atoms = len(c_alpha_coords)
                node_features = []

                for i in range(num_atoms):

                    dist_Ca_C = calculate_distance(c_alpha_coords[i], c_coords[i])
                    dist_Ca_N = calculate_distance(c_alpha_coords[i], n_coords[i])


                    angle_a =calculate_angle(n_coords[i], c_alpha_coords[i], c_coords[i])
                    angle_b = calculate_angle(c_coords[i - 1], n_coords[i], c_alpha_coords[i]) if i > 0 else 0
                    angle_y = calculate_angle(c_alpha_coords[i], c_coords[i], n_coords[i + 1]) if i < num_atoms - 1 else 0

                    angle_q = calculate_angle(n_coords[i], c_alpha_coords[i], c_coords[i])
                    angle_w = calculate_angle(c_coords[i], c_alpha_coords[i], n_coords[i])
                    if i < num_atoms - 1:
                        angle_e = calculate_angle(c_coords[i], n_coords[i + 1], c_alpha_coords[i])
                    else:
                        angle_e = 0  


                    dihedral_angle = calculate_dihedral_angle(c_alpha_coords[i - 1], c_coords[i], n_coords[i], c_alpha_coords[i + 1]) if i + 1 < num_atoms - 1 else 0
                    

                    node_features.append([dist_Ca_C.item() if isinstance(dist_Ca_C, torch.Tensor) else dist_Ca_C,dist_Ca_N.item() if isinstance(dist_Ca_N, torch.Tensor) else dist_Ca_N,angle_a.item() if isinstance(angle_a, torch.Tensor) else angle_a,angle_b.item() if isinstance(angle_b, torch.Tensor) else angle_b,angle_y.item() if isinstance(angle_y, torch.Tensor) else angle_y,angle_q.item() if isinstance(angle_q, torch.Tensor) else angle_q,angle_w.item() if isinstance(angle_w, torch.Tensor) else angle_w,angle_e.item() if isinstance(angle_e, torch.Tensor) else angle_e,dihedral_angle.item() if isinstance(dihedral_angle, torch.Tensor) else dihedral_angle])
                
                node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
                

                protein_save_dir = os.path.join(save_dir, protein_dir)
                if not os.path.exists(protein_save_dir):
                    os.makedirs(protein_save_dir)


                protein_name = protein_dir  
                save_path = os.path.join(protein_save_dir, f"{protein_name}_node_attr.pt")  
                
                torch.save(node_features, save_path)  

                all_node_features.append(node_features)
            else:
                print(f"{cif_path}")
    return all_node_features


base_dir = "/output_dir"# 
save_dir = "/prediction/attr/" # 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


all_node_features = process_protein_structure(base_dir, save_dir, device)
if all_node_features:

    print(all_node_features[0])  