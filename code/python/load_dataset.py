import csv
import numpy as np
import pandas as pd
import os
from scvi.dataset import CortexDataset, PbmcDataset, CbmcDataset, BrainSmallDataset, BrainLargeDataset, FrontalCortexDropseqDataset


def load_dataset(dataset_name):
    save_path = os.path.join('data/', dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if dataset_name == 'cortex':
        loaded_data = CortexDataset(save_path=save_path, total_genes=19972)
    elif dataset_name == 'pbmc':
        loaded_data = PbmcDataset(save_path=save_path, save_path_10X=os.path.join(save_path, '10x'))
    elif dataset_name == 'cbmc':
        loaded_data = CbmcDataset(save_path=os.path.join(save_path, 'citeSeq/'))
    elif dataset_name == 'brain_small':
        loaded_data = BrainSmallDataset(save_path=save_path, save_path_10X=os.path.join(save_path, '10x'))
    elif dataset_name == 'brain_large':
        loaded_data = BrainLargeDataset(save_path=save_path, sample_size_gene_var=10000, max_cells_to_keep=10000,
                                        nb_genes_to_keep=2000, loading_batch_size=1000)
    elif dataset_name == 'drop_seq':
        loaded_data = FrontalCortexDropseqDataset(save_path=save_path)
    else:
        raise AttributeError('dataset not available')

    return loaded_data


def load_brain_large_small():
    brain_small = load_dataset('brain_small')
    brain_large = load_dataset('brain_large')
    print(brain_large.X.shape)
    print(brain_large.labels)
    print(brain_large.cell_types)
    print(brain_large.gene_names)


def load_pbmc_cbmc():

    # load datasets
    pbmc = load_dataset('pbmc')
    cbmc = load_dataset('cbmc')

    # align based on genes
    eqv_names = []
    with open('data/pbmc/pbmcs.txt', 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            eqv_names.append(line[:-1])

    pbmc_selected_genes = [orig_name for orig_name, eqv_name in zip(pbmc.gene_names, eqv_names) if
                           eqv_name in cbmc.gene_names]
    cbmc_selected_genes = [eqv_names[pbmc.gene_names.tolist().index(pbmc_gene)] for pbmc_gene in pbmc_selected_genes]
    pbmc_idx = pbmc.genes_to_index(pbmc_selected_genes, on="gene_names")
    cbmc_idx = cbmc.genes_to_index(cbmc_selected_genes, on="gene_names")

    cbmc_features = cbmc.X.astype(dtype=np.float32)
    pbmc_features = pbmc.X.toarray().astype(dtype=np.float32)

    cbmc_features = cbmc_features[:, cbmc_idx]
    pbmc_features = pbmc_features[:, pbmc_idx]

    # organize cbmc based on selected classes
    cbmc_selected_cells = ['B', 'CD4 T', 'CD8 T', 'DC', 'NK', 'Mk', 'CD14+ Mono']
    df_labels = pd.read_csv('data/cbmc/labels.csv')
    cbmc_cell_types = df_labels.values[:, 2]
    cbmc_mask = [cell_type in cbmc_selected_cells for cell_type in cbmc_cell_types]
    cbmc_features = cbmc_features[cbmc_mask, :]
    cbmc_cell_types = cbmc_cell_types[cbmc_mask]
    cbmc_labels = np.array([cbmc_selected_cells.index(cell) for cell in cbmc_cell_types], dtype=int)

    # organize pbmc based on selected classes
    pbmc_selected_cells = ['B cells', 'CD4 T cells', 'CD8 T cells', 'Dendritic Cells', 'NK cells',
                           'Megakaryocytes', 'CD14+ Monocytes']
    pbmc_all_labels = pbmc.cell_types_to_labels(pbmc.cell_types).tolist()
    pbmc_cell_types = np.array([pbmc.cell_types[pbmc_all_labels.index(label)] for label in pbmc.labels])
    pbmc_mask = [cell_type in pbmc_selected_cells for cell_type in pbmc_cell_types]
    pbmc_features = pbmc_features[pbmc_mask, :]
    pbmc_cell_types = pbmc_cell_types[pbmc_mask]
    pbmc_labels = np.array([pbmc_selected_cells.index(cell) for cell in pbmc_cell_types], dtype=int)
    pbmc_cell_types = np.array([cbmc_selected_cells[pbmc_selected_cells.index(cell_type)] for cell_type in pbmc_cell_types])

    return pbmc_features, pbmc_labels, pbmc_cell_types, \
           cbmc_features, cbmc_labels, cbmc_cell_types, \
           cbmc_selected_genes, len(cbmc_selected_cells)


def load_adj(path):
    full_path = os.path.join(path, 'adj.csv')
    num_nodes = -1
    adj = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            adj.append([float(row[i]) for i in range(1, len(row))])

    adj = np.asarray(adj)
    return adj, num_nodes


# def load_classes(path):
#     full_path = os.path.join(path, 'classes.csv')
#     labels = []
#     class_names = []
#     num_graphs = -1
#     num_classes = 0
#     with open(full_path, mode='r') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         for row in csv_reader:
#             num_graphs += 1
#             if num_graphs == 0:
#                 continue
#             if row[2] in class_names:
#                 labels.append(class_names.index(row[2]))
#             else:
#                 class_names.append(row[2])
#                 labels.append(num_classes)
#                 num_classes += 1
#
#     labels = np.asarray(labels)
#     one_hot_labels = np.zeros((num_graphs, num_classes))
#     one_hot_labels[np.arange(num_graphs), labels] = 1
#
#     return labels, one_hot_labels, num_graphs, num_classes

def load_classes(path):
    full_path = os.path.join(path, 'classes.csv')
    classes = pd.read_csv(full_path)
    # print(classes)
    nans = pd.isna(classes['class']).values
    classes.dropna(axis=0, inplace=True)
    labels = classes['id'].values.astype(int)
    # print(labels)
    num_classes = np.max(labels)
    num_graphs = labels.shape[0]
    labels -= np.ones(shape=(num_graphs,), dtype=int)
    one_hot_labels = np.zeros((num_graphs, num_classes))
    one_hot_labels[np.arange(num_graphs), labels] = 1
    return labels, one_hot_labels, num_graphs, num_classes, nans


def load_train_classes(path):
    full_path = os.path.join(path, 'classes.csv')
    classes = pd.read_csv(full_path)
    nans = pd.isna(classes['class']).values
    classes.dropna(axis=0, inplace=True)
    labels = classes['id'].values.astype(int)
    print(labels)
    num_classes = np.max(labels)
    num_graphs = labels.shape[0]
    labels -= np.ones(shape=(num_graphs,), dtype=int)
    one_hot_labels = np.zeros((num_graphs, num_classes))
    one_hot_labels[np.arange(num_graphs), labels] = 1
    return labels, one_hot_labels, num_graphs, num_classes, nans

def load_test_classes(path):
    full_path = os.path.join(path, 'classes.csv')
    classes = pd.read_csv(full_path)
    nans = pd.isna(classes['class']).values
    classes.dropna(axis=0, inplace=True)
    labels = classes['id'].values.astype(int)
    print(labels)
    num_classes = np.max(labels)
    num_graphs = labels.shape[0]
    labels -= np.ones(shape=(num_graphs,), dtype=int)
    one_hot_labels = np.zeros((num_graphs, num_classes))
    one_hot_labels[np.arange(num_graphs), labels] = 1
    return labels, one_hot_labels, num_graphs, num_classes, nans

def load_features(path, is_binary=False):
    full_path = os.path.join(path, 'features.csv')
    num_nodes = -1
    features = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            if is_binary:
                features.append([1 if float(row[i]) > 0 else 0 for i in range(1, len(row))])
            else:
                features.append([float(row[i]) for i in range(1, len(row))])
    features = np.asarray(features)
    features = features.T
    return features


def load_train_features(path, is_binary=False):
    full_path = os.path.join(path, 'train_features.csv')
    num_nodes = -1
    features = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            if is_binary:
                features.append([1 if float(row[i]) > 0 else 0 for i in range(1, len(row))])
            else:
                features.append([float(row[i]) for i in range(1, len(row))])
    features = np.asarray(features)
    features = features.T
    return features

def load_test_features(path, is_binary=False):
    full_path = os.path.join(path, 'test_features.csv')
    num_nodes = -1
    features = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            if is_binary:
                features.append([1 if float(row[i]) > 0 else 0 for i in range(1, len(row))])
            else:
                features.append([float(row[i]) for i in range(1, len(row))])
    features = np.asarray(features)
    features = features.T
    return features