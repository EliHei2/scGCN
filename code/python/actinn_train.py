from load_dataset import load_pbmc_cbmc
import pandas as pd
import os
import numpy as np
import subprocess


def prepare_data(path):
    pbmc_features, pbmc_labels, pbmc_cell_types, \
    cbmc_features, cbmc_labels, cbmc_cell_types, gene_names, _ = load_pbmc_cbmc()

    if not os.path.exists(path):
        os.makedirs(path)

    train_df = pd.DataFrame(data=pbmc_features.T, index=gene_names)
    train_df.to_csv(os.path.join(path, 'actinn_pbmc_features.csv'))

    test_df = pd.DataFrame(data=cbmc_features.T, index=gene_names)
    test_df.to_csv(os.path.join(path, 'actinn_cbmc_features.csv'))

    with open(os.path.join(path, 'actinn_pbmc_labels.txt'), mode='w') as f:
        for i, label in enumerate(pbmc_cell_types):
            f.write('{}\t'.format(i) + label + '\n')

    with open(os.path.join(path, 'actinn_cbmc_labels.txt'), mode='w') as f:
        for i, label in enumerate(cbmc_cell_types):
            f.write('{}\t'.format(i) + label + '\n')


def accuracy(path):
    ground_truth = []
    with open(os.path.join(path, 'actinn_cbmc_labels.txt'), mode='r') as f:
        for line in f:
            f = line.split()
            ground_truth.append(f[1])

    predicted = []
    with open(os.path.join(path, 'predicted_label.txt'), mode='r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            f = line.split()
            predicted.append(f[1])

    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)

    acc = np.mean(ground_truth == predicted)
    return acc

if __name__ == '__main__':
    path = './actinn_files/'
    prepare_data(path)
    learning_rate = 0.0001
    n_epochs = 50
    batch_size = 128
    subprocess.call(['./format_data_run.sh', str(learning_rate), str(n_epochs), str(batch_size)], cwd=path)
    acc = accuracy(path)
    print('test acc={:.3f}'.format(acc))
