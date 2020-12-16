from load_dataset import *
from utils import *
from model import *

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


flags.DEFINE_string('model', 'inception', 'Model string.')  # gcn, gcn_cheby, inception
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 5, 'Numbe:qr of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 9, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_bool('featureless', False, 'featureless')

# data directorys
base_path      = 'data_train/pbmc'

# read adj matrix
adj, num_nodes = load_adj(base_path)

# read in train data
## read labels
train_labels, train_one_hot_labels, train_num_graphs, num_classes,\
    train_nan_idx = load_classes(base_path, type='train')
train_class_dist  = \
    [train_labels.tolist().count(i) / train_num_graphs for i in range(num_classes)]
## read features
train_features    = load_features(base_path, type='train', is_binary=False)
train_features    = train_features[~train_nan_idx]
## shuffle data
idx               = np.arange(train_num_graphs)
np.random.shuffle(idx)
train_labels      = train_labels[idx]
train_one_hot_labels = train_one_hot_labels[idx, :]
train_features    = train_features[idx, :]
## for later analyses
with open("data_train/cross_hm/train_idx.txt", "w") as f:
    for s in idx:
        f.write(str(s) +"\n")
## sparse features
train_sparse_features = []
for i in range(train_num_graphs):
    train_sparse_features.append(\
        sparse_to_tuple(\
            sp.coo_matrix(\
                np.expand_dims(\
                    np.transpose(train_features[i, :])\
                , 1)
            )
        )
    )
## class weights
train_graph_weights = \
    [1 / train_class_dist[train_labels[i]] for i in range(train_num_graphs)]

# read in test data
## read labels
test_labels, test_one_hot_labels, test_num_graphs, num_classes,\
    test_nan_idx = load_classes(base_path, type='test')
test_class_dist  = \
    [test_labels.tolist().count(i) / test_num_graphs for i in range(num_classes)]
## read features
test_features    = load_features(base_path, type='test', is_binary=False)
test_features    = test_features[~test_nan_idx]
## shuffle data
idx              = np.arange(test_num_graphs)
np.random.shuffle(idx)
test_labels      = test_labels[idx]
test_one_hot_labels = test_one_hot_labels[idx, :]
test_features    = test_features[idx, :]
## for later analyses
with open("data_train/cross_hm/test_idx.txt", "w") as f:
    for s in idx:
        f.write(str(s) +"\n")
## sparse features
test_sparse_features = []
for i in range(test_num_graphs):
    test_sparse_features.append(\
        sparse_to_tuple(\
            sp.coo_matrix(\
                np.expand_dims(\
                    np.transpose(test_features[i, :])\
                , 1)
            )
        )
    )
## class weights
test_graph_weights  = \
    [1 / test_class_dist[test_labels[i]] for i in range(test_num_graphs)]

# specify the model
if FLAGS.model == 'gcn_cheby':
    locality = [7, 5, 3]  # locality sizes of different blocks
    num_supports = np.max(locality) + 1
    support = chebyshev_polynomials(adj, num_supports - 1)
elif FLAGS.model == 'inception':
    localityـsizes = [7, 5, 3]
    num_supports = np.max(localityـsizes) + 1
    support = chebyshev_polynomials(adj, num_supports - 1)
elif FLAGS.model == 'gcn':
    num_supports = 1
    support = [preprocess_adj(adj)]
else:
    raise NotImplementedError

# define placeholders
placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, \
            name='support_{}'.format(i)) for i in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32),
        'labels': tf.placeholder(tf.float32, shape=(train_one_hot_labels.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'weight': tf.placeholder(tf.float32),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

# define the model
if FLAGS.model == 'gcn_cheby':
    model = CheybyGCN(placeholders, input_dim=1,\
        num_class=num_classes, locality=locality, name='gcn_cheby')
elif FLAGS.model == 'inception':
    model = InceptionGCN2L(placeholders, input_dim=1,\
        num_class=num_classes, locality_sizes=localityـsizes,\
        is_pool=True, name='inception')
else:
    model = SimpleGCN(placeholders, input_dim=1, num_class=num_classes, name='simple')


with tf.Session() as sess:
    writer = tf.summary.FileWriter('.logs/final')
    writer.add_graph(sess.graph)
    print('HALOOO')
    sess.run(tf.global_variables_initializer())
    test_result = []
    sum_acc = 0
    sum_sum_loss = 0
    test_acc = 0
    # tf.summary.scalar('train_acc', sum_acc)
    # tf.summary.scalar('train_loss', sum_sum_loss)
    # tf.summary.scalar('test_acc', test_acc)
    summ = tf.summary.merge_all()
    for epoch in range(FLAGS.epochs):
        cnt = 0
        sum_loss = 0
        train_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
        test_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)

        for i in range(train_num_graphs):
            train_feed_dict = \
                construct_feed_dict(train_sparse_features[i],
                                    support,
                                    train_one_hot_labels[i],
                                    train_graph_weights[i],
                                    placeholders)
            train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            _, loss, acc, out= sess.run([model.opt_op,
                                          model.loss,
                                          model.accuracy,
                                          model.outputs],
                                         feed_dict=train_feed_dict)
            train_acc_classes[train_labels[i], np.argmax(out, 1)[0]] += 1
            cnt += acc
            sum_loss += loss
        sum_acc = cnt / float(train_num_graphs)
        sum_sum_loss = sum_loss / float(train_num_graphs)
        print('Epoch {}:'.format(epoch + 1),
              'acc={:.4f}, loss={:.4f}'.format(sum_acc, sum_sum_loss))
        # sum_acc = sess.run(sum_acc)
        # sum_sum_loss = sess.run(sum_sum_loss)

        cnt = 0
        for i in range(test_num_graphs):
            test_feed_dict = construct_feed_dict(test_sparse_features[i],
                                                 support,
                                                 test_one_hot_labels[i],
                                                 test_graph_weights[i],
                                                 placeholders)
            test_feed_dict.update({placeholders['dropout']: 0.})

            acc, out = sess.run([model.accuracy,
                                 model.outputs], feed_dict=test_feed_dict)
            test_acc_classes[test_labels[i], np.argmax(out, 1)[0]] += 1
            cnt += acc
        test_acc = cnt / float(test_num_graphs)
        test_result.append(test_acc)
        print('Test accuracy: {:.4f}'.format(test_acc))
        print('train confusion matrix: \n', train_acc_classes)
        print('test confusion matrix: \n', test_acc_classes)
        # test_acc = sess.run(test_acc)
        # summ = sess.run(summ)
        # writer.add_summary(summ, epoch)



    print("Optimization finished!")
    print("Start Saving Embeddings")
    node_representations(support, train_sparse_features, train_one_hot_labels, placeholders, num_nodes, model, 'train')
    node_representations(support, test_sparse_features, test_one_hot_labels, placeholders, num_nodes, model, 'test')

    model.save()
