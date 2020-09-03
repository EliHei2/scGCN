import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


flags = tf.app.flags
FLAGS = flags.FLAGS


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    print(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo())
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    print(sparse_to_tuple(adj_normalized))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, weight, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['weight']: weight})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def node_representations(support, sparse_features, one_hot_labels, placeholders, num_nodes, model, file_name):
    sess = tf.get_default_session()
    num_graphs = one_hot_labels.shape[0]
    # num_class = one_hot_labels.shape[1]
    num_layers = 3

    # layer_embeddings = np.zeros((num_layers, num_class, num_nodes))
    layer_embeddings = np.zeros((num_layers, num_graphs, num_nodes))
    # class_num_instance = np.zeros((num_class,))

    for i in range(num_graphs):
        feed_dict = construct_feed_dict(sparse_features[i], support, one_hot_labels[i], 1, placeholders)
        feed_dict.update({placeholders['dropout']: 0.})
        graph_embeddings = sess.run(model.activations[1:1 + num_layers], feed_dict=feed_dict)
        # class_idx = np.argmax(one_hot_labels[i])
        # class_num_instance[class_idx] += 1
        for layer in range(num_layers):
            # layer_embeddings[layer, class_idx, :] += np.mean(graph_embeddings[layer], axis=-1)
            layer_embeddings[layer, i, :] += np.mean(graph_embeddings[layer], axis=-1)

    # for j in range(num_class):
    #     layer_embeddings[:, j, :] /= class_num_instance[j]
    for layer in range(num_layers):
        embedding_df = pd.DataFrame(data=layer_embeddings[layer, :, :].T)
        embedding_df.to_csv('./models/pbmc/' + file_name + '_layer_{}.csv'.format(layer + 1))

def node_rep_fc(features, labels, weights, activations, file_name):
    sess = tf.get_default_session()
    num_graphs = labels.shape[0]
    num_nodes  = features.shape[1]
    num_layers = 2
    input_features = tf.placeholder(shape=[None, num_nodes], dtype=tf.float32)
    sample_weights = tf.placeholder(shape=[None,], dtype=tf.float32)
    input_labels = tf.placeholder(shape=[None,], dtype=tf.int64)
    layer_embeddings = np.zeros((num_layers, num_graphs, num_nodes))
    for i in range(num_graphs):
        print(i)
        print(labels[i:(i+1)].shape)
        # print(weights[i:(i+1)].shape)
        print(features[i:(i+1)].shape)
        feed_dict = dict()
        feed_dict.update({input_features: features[i:(i+1)], input_labels: labels[i:(i+1)], sample_weights: weights[i:(i+1)]})
        graph_embeddings = sess.run(activations[1:1 + num_layers], feed_dict=feed_dict)
        for layer in range(num_layers):
            layer_embeddings[layer, i, :] += np.mean(graph_embeddings[layer], axis=-1)

    for layer in range(num_layers):
        embedding_df = pd.DataFrame(data=layer_embeddings[layer, :, :].T)
        embedding_df.to_csv('./models/pbmc/' + file_name + '_layer_{}_FC.csv'.format(layer + 1))


def visualize_graph(adj, activations, num_sample, layout=None):
    idx = np.arange(adj.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num_sample]
    num_nodes = len(idx)
    adj = adj[idx, :]
    adj = adj[:, idx]
    acts = activations.loc[idx, :]

    graph = nx.Graph()
    graph.add_nodes_from(np.arange(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] != 0:
                graph.add_edge(i, j, weight=adj[i, j])

    if layout is None:
        layout = nx.spring_layout(graph, weight='weight', iterations=100, scale=1)

    cmap = plt.cm.seismic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=np.max(acts.values)))
    sm._A = []
    for i in range(len(acts.columns)):
        node_size = acts.loc[:, str(i)].values
        node_size /= np.max(node_size)
        node_size *= 50
        # print(node_size)
        node_size = 20
        nx.draw_networkx(graph, layout, node_size=node_size, width=0.3, node_color=acts.loc[:, str(i)], cmap=cmap,
                         with_labels=False)
        plt.colorbar(sm)
        plt.show()

    return layout
