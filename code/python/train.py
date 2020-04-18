from codes.GCN.load_dataset import *
from codes.GCN.utils import *
from codes.GCN.model import *

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


flags.DEFINE_string('model', 'inception', 'Model string.')  # gcn, gcn_cheby, inception
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 36, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 18, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 9, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_bool('featureless', False, 'featureless')


# base_path = './data/custom_data/'
train_path = "../../data/nestorowa/proc/"
test_path = "../../data/nestorowa/proc/"
adj, num_nodes = load_adj(test_path)
train_labels, train_one_hot_labels, train_num_graphs, num_classes, train_nan_idx = load_classes(train_path)
train_class_dist = [train_labels.tolist().count(i) / train_num_graphs for i in range(num_classes)]
train_features = load_features(train_path, is_binary=False)
print("****************")
print(train_features[1:10,1:10])
# train_features = train_features[~train_nan_idx]
train_class_idx = np.asarray([i for i, l in enumerate(train_labels) if train_class_dist[l] >= 0.0001])

print(train_class_idx)
train_labels = train_labels[train_class_idx]
print(train_labels)
train_one_hot_labels = train_one_hot_labels[train_class_idx]
train_features = train_features[train_class_idx]
num_classes -= np.sum(np.asarray(train_class_dist) < 0.0001).astype(int)
train_num_graphs = train_labels.shape[0]
# k = (train_one_hot_labels != 0).any(0)
train_one_hot_labels = train_one_hot_labels[:, (train_one_hot_labels != 0).any(0)]
# labels = np.argmax(train_one_hot_labels, axis=1)
print(num_classes)
# new_class_dist = [labels.tolist().count(i) / train_num_graphs for i in range(num_classes)]

# train_proportion = 0.75
train_num_graphs = train_num_graphs
idx = np.arange(train_num_graphs)
np.random.shuffle(idx)

# collecting test samples
test_labels, test_one_hot_labels, test_num_graphs, test_num_classes, test_nan_idx = load_classes(test_path)
test_class_dist = [test_labels.tolist().count(i) / test_num_graphs for i in range(test_num_classes)]
test_features = load_features(test_path, is_binary=False)
test_features = test_features[~test_nan_idx]
test_class_idx = np.asarray([i for i, l in enumerate(test_labels) if test_class_dist[l] >= 0.0001])
test_labels = test_labels[test_class_idx]
test_one_hot_labels = test_one_hot_labels[test_class_idx]
test_features = test_features[test_class_idx]
test_num_graphs = test_labels.shape[0]
# k = (test_one_hot_labels != 0).any(0)
test_one_hot_labels = test_one_hot_labels[:, (test_one_hot_labels != 0).any(0)]
labels = np.argmax(test_one_hot_labels, axis=1)
# new_class_dist = [labels.tolist().count(i) / test_num_graphs for i in range(test_num_classes)]

train_sparse_features = []
test_sparse_features = []
sparse_features = []



train_graph_weights = [1 / train_class_dist[train_labels[i]] for i in range(train_num_graphs)]
test_graph_weights = [1 / test_class_dist[train_labels[i]] for i in range(test_num_graphs)]

# for i in range(train_num_graphs):
#     sparse_features.append(sparse_to_tuple(sp.coo_matrix(np.expand_dims(np.transpose(train_features[i, :]), 1))))

for i in range(train_num_graphs):
    train_sparse_features.append(sparse_to_tuple(sp.coo_matrix(np.expand_dims(np.transpose(train_features[i, :]), 1))))
    # print(train_sparse_features[i])
    # print(train_one_hot_labels[i])

print(len(train_sparse_features))


for i in range(test_num_graphs):
    test_sparse_features.append(sparse_to_tuple(sp.coo_matrix(np.expand_dims(np.transpose(test_features[i, :]), 1))))
    # print(test_sparse_features[i])
    # print(test_one_hot_labels[i])
print(len(test_sparse_features))

if FLAGS.model == 'gcn_cheby':
    locality1 = 8
    locality2 = 7
    locality3 = 6
    locality = [locality1, locality2, locality3]  # locality sizes of different blocks
    num_supports = np.max(locality) + 1
    support = chebyshev_polynomials(adj, num_supports - 1)
elif FLAGS.model == 'inception':
    locality_sizes = [7, 5, 3]
    num_supports = np.max(locality_sizes) + 1
    support = chebyshev_polynomials(adj, num_supports - 1)
elif FLAGS.model == 'gcn':
    num_supports = 1
    support = [preprocess_adj(adj)]
else:
    raise NotImplementedError


placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32),
        'labels': tf.placeholder(tf.float32, shape=(train_one_hot_labels.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'weight': tf.placeholder(tf.float32),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

# model definition
if FLAGS.model == 'gcn_cheby':
    model = CheybyGCN(placeholders, input_dim=1, num_class=num_classes, locality=locality, name='gcn_cheby')
elif FLAGS.model == 'inception':
    model = InceptionGCN(placeholders, input_dim=1, num_class=num_classes,
                         locality_sizes=locality_sizes, is_pool=True, name='inception')
else:
    model = SimpleGCN(placeholders, input_dim=1, num_class=num_classes, name='simple')
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     test_result = []
#     for epoch in range(FLAGS.epochs):
#         # print('Starting epoch {}'.format(epoch + 1))
#         cnt = 0
#         sum_loss = 0
#         train_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
#         # print(len(train_acc_classes))
#         test_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
#         # print(len(test_acc_classes))
#         for i in range(train_num_graphs):
#             train_feed_dict = construct_feed_dict(train_sparse_features[i], support, train_one_hot_labels[i],
#                                                   train_graph_weights[i], placeholders)
#
#             # features, support, labels, weight, placeholders
#             # print(train_feed_dict.shape)
#             train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#
#             _, loss, acc, out = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs],
#                                          feed_dict=train_feed_dict)
#             train_acc_classes[train_labels[i], np.argmax(out, 1)[0]] += 1
#             # print('Graph {}: '.format(i + 1), 'Loss={}, '.format(loss), 'Acc={}'.format(acc))
#             # train_acc_classes[train_labels[i], prediction] += 1
#             cnt += acc
#             sum_loss += loss
#         print('Epoch {}:'.format(epoch + 1), 'acc={:.4f}, loss={:.4f}'.format(cnt / float(train_num_graphs),
#                                                                               sum_loss / float(train_num_graphs)))
#
#         cnt = 0
#         for i in range(test_num_graphs):
#             test_feed_dict = construct_feed_dict(test_sparse_features[i], support, test_one_hot_labels[i],
#                                                  1, placeholders)
#             # print(test_feed_dict.shape)
#             test_feed_dict.update({placeholders['dropout']: 0.})
#             # print(test_feed_dict.shape)
#
#             acc, out = sess.run([model.accuracy, model.outputs], feed_dict=test_feed_dict)
#             test_acc_classes[test_labels[i], np.argmax(out, 1)[0]] += 1
#             # test_acc_classes[test_labels[i], prediction] += 1
#             cnt += acc
#         test_acc = cnt / float(test_num_graphs)
#         test_result.append(test_acc)
#         print('Test accuracy: {:.4f}'.format(test_acc))
#         print('train confusion matrix: \n', train_acc_classes)
#         print('test confusion matrix: \n', test_acc_classes)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_result = []
    for epoch in range(FLAGS.epochs):
        # print('Starting epoch {}'.format(epoch + 1))
        cnt = 0
        sum_loss = 0
        train_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
        print(train_acc_classes)
        test_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
        for i in range(train_num_graphs):
            # print(placeholders)
            train_feed_dict = construct_feed_dict(train_sparse_features[i], support, train_one_hot_labels[i],
                                                  train_graph_weights[i], placeholders)
            train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            _, loss, acc, out = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs],
                                         feed_dict=train_feed_dict)
            train_acc_classes[train_labels[i], np.argmax(out, 1)[0]] += 1
            # print('Graph {}: '.format(i + 1), 'Loss={}, '.format(loss), 'Acc={}'.format(acc))
            # train_acc_classes[train_labels[i], prediction] += 1
            cnt += acc
            sum_loss += loss
        print('Epoch {}:'.format(epoch + 1), 'acc={:.4f}, loss={:.4f}'.format(cnt / float(train_num_graphs),
                                                                              sum_loss / float(train_num_graphs)))

        cnt = 0
        for i in range(test_num_graphs):
            # print(test_sparse_features[i].shape)
            # print(support.shape)
            # print(test_one_hot_labels[i].shape)
            # print(placeholders)
            test_feed_dict = construct_feed_dict(test_sparse_features[i], support, test_one_hot_labels[i], 1,
                                                 placeholders)
            test_feed_dict.update({placeholders['dropout']: 0.})

            acc, out = sess.run([model.accuracy, model.outputs], feed_dict=test_feed_dict)
            test_acc_classes[test_labels[i], np.argmax(out, 1)[0]] += 1
            # test_acc_classes[test_labels[i], prediction] += 1
            cnt += acc
        test_acc = cnt / float(test_num_graphs)
        test_result.append(test_acc)
        print('Test accuracy: {:.4f}'.format(test_acc))
        print('train confusion matrix: \n', train_acc_classes)
        print('test confusion matrix: \n', test_acc_classes)

    # print('Storing graph embedding')
    # embedding_level = 4
    # with open('./data/train_graph_embedding.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     header = ['id']
    #     for i in range(FLAGS.hidden3):
    #         header.append('emb_{}'.format(i))
    #     writer.writerow(header)
    #     embeddings = []
    #     for i in range(train_num_graphs):
    #         feed_dict = construct_feed_dict(train_sparse_features[i], support, train_one_hot_labels[i], 1, placeholders)
    #         feed_dict.update({placeholders['dropout']: 0.})
    #         embedding = sess.run(model.activations[embedding_level], feed_dict=feed_dict)
    #         row = [i + 1]
    #         for item in embedding.tolist()[0]:
    #             row.append(item)
    #         writer.writerow(row)
    #         embeddings.append(embedding.tolist()[0])
    #
    # with open('./data/test_graph_embeddings.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     header = ['id']
    #     for i in range(FLAGS.hidden3):
    #         header.append('emb_{}'.format(i))
    #     writer.writerow(header)
    #     embeddings = []
    #     for i in range(test_num_graphs):
    #         feed_dict = construct_feed_dict(test_sparse_features[i], support, test_one_hot_labels[i], 1, placeholders)
    #         feed_dict.update({placeholders['dropout']: 0.})
    #         embedding = sess.run(model.activations[embedding_level], feed_dict=feed_dict)
    #         row = [i + 1]
    #         for item in embedding.tolist()[0]:
    #             row.append(item)
    #         writer.writerow(row)
    #         embeddings.append(embedding.tolist()[0])
    #
    print("Optimization finished!")
    #
    # plt.plot(test_result)
    # plt.show()

    print("Start Saving Embeddings")
    node_representations(support, train_sparse_features, train_one_hot_labels, placeholders, num_nodes, model, 'train')
    node_representations(support, test_sparse_features, test_one_hot_labels, placeholders, num_nodes, model, 'test')

    model.save()
    # print('Plotting t-SNE')
    # embeddings = np.asarray(embeddings)
    # reduced_embedding = TSNE(n_components=2).fit_transform(embeddings)
    # color_names = ['b', 'g', 'r', 'y']
    # colors = [color_names[label] for label in labels]
    # num_samples = 5000
    # plt.scatter(reduced_embedding[:num_samples, 0], reduced_embedding[:num_samples, 1],
    #             marker='.',
    #             c=colors[:num_samples])
    # plt.show()
