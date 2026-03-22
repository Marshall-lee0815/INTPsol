# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:23:09 2024

@author: marshall
"""
import os
import torch
import esm
from Bio import SeqIO


model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


output_folder = ""
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def read_protein_sequences(txt_file):
    sequences = []
    with open(txt_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequences.append((record.id, str(record.seq)))
    return sequences


sequences = read_protein_sequences("")


for protein_id, sequence in sequences:

    data = [(protein_id, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)


    batch_tokens = batch_tokens.to(device)


    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33].cpu()


    output_path = os.path.join(output_folder, f"{protein_id}_embedding.pt")
    torch.save(token_representations, output_path)
    print(f"Saved embedding for {protein_id} to {output_path}")
