import numpy as np
from Bio import PDB
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def get_ca_coords(cif_file):
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein', cif_file)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    return np.array(ca_coords)

def compute_contact_map(coords, threshold=8.0):
    dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2, axis=-1))
    contact_map = (dist_matrix < threshold).astype(int)
    return contact_map

def plot_contact_map(contact_map):
    plt.imshow(contact_map, cmap='Greys', interpolation='none')
    plt.title('Protein Contact Map')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.show()


cif_file = ""
ca_coords = get_ca_coords(cif_file)
contact_map = compute_contact_map(ca_coords)


G = nx.Graph()

for i in range(len(contact_map)):
    G.add_node(i)

for i in range(len(contact_map)):
    for j in range(len(contact_map[i])):
        if contact_map[i][j] == 1:
            G.add_edge(i, j)

nx.draw(G, with_labels=True, node_color='skyblue', node_size=70, edge_color='gray')
plt.show()


print(contact_map.shape)
plot_contact_map(contact_map)
