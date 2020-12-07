import tensorflow as tf
from load_dataset import *
from utils import *
from one_run import one_run
import os
import time

timestr = time.strftime("%Y%m%d-%H%M%S")

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'inception', 'Model string.')  # gcn, gcn_cheby, inception
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 36, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 18, 'Numbe:qr of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 9, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.05, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_bool('featureless', False, 'featureless')

n_layers_hyperopt_dict = [2, 3]
locality_hyperopt_dict = [{"num_locality": 2, "locality_sizes": [[5, 1], [6, 2], [7, 3], [8, 4]]},
                  {"num_locality": 3, "locality_sizes": [[6, 3, 1], [8, 3, 1], [6, 4, 2], [8, 4, 2]]}]

# data directory
base_path = 'mm_2020_12_04'
# base_path = 'data_train/pbmc'

# read adj matrix
adj, num_nodes = load_adj(base_path)

# read in train data
# read labels
train_labels, train_one_hot_labels, train_num_graphs, train_num_classes, train_nan_idx = load_classes(base_path,
                                                                                                      type='train')
train_class_dist = [train_labels.tolist().count(i) / train_num_graphs for i in range(train_num_classes)]
# read features
train_features = load_features(base_path, type='train', is_binary=False)
train_features = train_features[~train_nan_idx]
# shuffle data
idx = np.arange(train_num_graphs)
np.random.shuffle(idx)
train_labels = train_labels[idx]
train_one_hot_labels = train_one_hot_labels[idx, :]
train_features = train_features[idx, :]

# for later analyses
with open(base_path + "/train_idx.txt", "w") as f:
    for s in idx:
        f.write(str(s) + "\n")
# sparse features
train_sparse_features = []
for i in range(train_num_graphs):
    train_sparse_features.append(sparse_to_tuple(sp.coo_matrix(np.expand_dims(np.transpose(train_features[i, :]), 1))))

# class weights
train_graph_weights = [1 / train_class_dist[train_labels[i]] for i in range(train_num_graphs)]

# read in test data

# read labels
test_labels, test_one_hot_labels, test_num_graphs, test_num_classes, test_nan_idx = load_classes(base_path,
                                                                                                 type='test',
                                                                                                 max_labels=train_num_classes)
test_class_dist = [test_labels.tolist().count(i) / test_num_graphs for i in range(test_num_classes)]

# read features
test_features = load_features(base_path, type='test', is_binary=False)
test_features = test_features[~test_nan_idx]

# shuffle data
idx = np.arange(test_num_graphs)
np.random.shuffle(idx)
test_labels = test_labels[idx]
test_one_hot_labels = test_one_hot_labels[idx, :]
test_features = test_features[idx, :]

# for later analyses
with open(base_path + "/test_idx.txt", "w") as f:
    for s in idx:
        f.write(str(s) + "\n")

# sparse features
test_sparse_features = []
for i in range(test_num_graphs):
    test_sparse_features.append(sparse_to_tuple(sp.coo_matrix(np.expand_dims(np.transpose(test_features[i, :]), 1))))

# class weights
test_graph_weights = [1 / test_class_dist[test_labels[i]] for i in range(test_num_graphs)]

for n_layers in n_layers_hyperopt_dict:
    for setting in locality_hyperopt_dict:
        for locality_sizes in setting['locality_sizes']:
            tf.reset_default_graph()
            # define placeholders
            num_supports = np.max(locality_sizes) + 1
            support = chebyshev_polynomials(adj, num_supports - 1)
            placeholders = {
                'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
                'features': tf.sparse_placeholder(tf.float32),
                'labels': tf.placeholder(tf.float32, shape=(train_one_hot_labels.shape[1])),
                'dropout': tf.placeholder_with_default(0., shape=()),
                'weight': tf.placeholder(tf.float32),
                'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
            }
            train_precision, train_recall, \
            test_precision, test_recall, \
            train_f1, test_f1, \
            train_acc_classes, test_acc_classes = one_run(n_layers, placeholders,
                                                          support,
                                                          train_num_graphs,
                                                          train_sparse_features,
                                                          train_graph_weights,
                                                          train_one_hot_labels,
                                                          train_labels,
                                                          test_num_graphs,
                                                          test_sparse_features,
                                                          test_graph_weights,
                                                          test_one_hot_labels,
                                                          test_labels,
                                                          locality_sizes,
                                                          train_num_classes,
                                                          test_num_classes,
                                                          num_nodes)

            directory = base_path + '/logs/' + timestr + '/locality_' + str(locality_sizes) + '_{}layers'.format(n_layers)

            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(directory + '/' + 'train.txt', 'w+') as f:
                f.write('Precision: \n' + np.array2string(train_precision) + '\n\n')
                f.write('Recall: \n' + np.array2string(train_recall) + '\n\n')
                f.write('f1: \n' + np.array2string(train_f1) + '\n\n')
                f.write('confusion: \n' + np.array2string(train_acc_classes))

            with open(directory + '/' + 'test.txt', 'w+') as f:
                f.write('Precision: \n' + np.array2string(test_precision) + '\n\n')
                f.write('Recall: \n' + np.array2string(test_recall) + '\n\n')
                f.write('f1: \n' + np.array2string(test_f1) + '\n\n')
                f.write('confusion: \n' + np.array2string(test_acc_classes))
