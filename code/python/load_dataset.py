import csv
import numpy as np
import pandas as pd
import os


def load_adj(path):
    full_path = os.path.join(path, 'adj.txt')
    num_nodes = -1
    adj = []
    with open(full_path, mode='r') as txt_file:
        for row in txt_file:
            row = row.split(",")
            num_nodes += 1
            if num_nodes == 0:
                continue
            adj.append([float(row[i]) for i in range(0, len(row))])

    adj = np.asarray(adj)
    return adj, num_nodes


def load_classes(path, type, max_labels=None):
    full_path = os.path.join(path, 'classes_{type}.txt'.format(type=type))
    classes = pd.read_csv(full_path)
    nans = pd.isna(classes['cell_type']).values
    classes.dropna(axis=0, inplace=True)
    classes['id'] = pd.factorize(classes.cell_type)[0]
    labels = classes['id'].values
    labels -= (np.min(labels) - 1)
    # labels = classes['id'].values.astype(int)
    print(labels)
    if (max_labels is None) or max_labels >= np.max(labels):
        num_classes = np.max(labels)
        num_graphs = labels.shape[0]
        labels -= np.ones(shape=(num_graphs,), dtype=int)
        one_hot_labels = np.zeros((num_graphs, num_classes))
        one_hot_labels[np.arange(num_graphs), labels] = 1
        return labels, one_hot_labels, num_graphs, num_classes, nans
    else:
        num_classes = max_labels
        num_graphs = labels.shape[0]
        for_one_hot = np.where(labels <= max_labels, labels, 0)
        labels = np.where(labels <= max_labels, labels, max_labels + 1)
        labels -= np.ones(shape=(num_graphs,), dtype=int)
        one_hot_labels = np.zeros((num_graphs, num_classes))
        one_hot_labels[np.arange(num_graphs), for_one_hot] = 1
        return labels, one_hot_labels, num_graphs, max_labels + 1, nans


def load_features(path, type, is_binary=False):
    full_path = os.path.join(path, 'data_{type}.txt'.format(type=type))
    num_nodes = -1
    features = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            if is_binary:
                features.append([1 if float(row[i]) > 0 else 0 for i in range(0, len(row))])
            else:
                features.append([float(row[i]) for i in range(0, len(row))])
    features = np.asarray(features)
    features = features.T
    return features
