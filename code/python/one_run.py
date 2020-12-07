from utils import *
from model import InceptionGCN3L, InceptionGCN2L

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def one_run(n_layers, placeholders, support,
            train_num_graphs, train_sparse_features, train_graph_weights, train_one_hot_labels, train_labels,
            test_num_graphs, test_sparse_features, test_graph_weights, test_one_hot_labels, test_labels,
            locality_sizes, train_num_classes, test_num_classes, num_nodes):
    # define the model
    if n_layers == 3:
        model = InceptionGCN3L(placeholders, input_dim=1, num_class=train_num_classes, locality_sizes=locality_sizes,
                               is_pool=True,
                               name='inception')
    else:
        model = InceptionGCN2L(placeholders, input_dim=1, num_class=train_num_classes, locality_sizes=locality_sizes,
                               is_pool=True,
                               name='inception')

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('.logs/final')
        writer.add_graph(sess.graph)
        # print('HALOOO')
        sess.run(tf.global_variables_initializer())
        test_result = []
        sum_acc = 0
        sum_sum_loss = 0
        test_acc = 0
        # tf.summary.scalar('train_acc', sum_acc)
        # tf.summary.scalar('train_loss', sum_sum_loss)
        # tf.summary.scalar('test_acc', test_acc)
        # summ = tf.summary.merge_all()
        for epoch in range(FLAGS.epochs):
            cnt = 0
            sum_loss = 0
            train_acc_classes = np.zeros((train_num_classes, train_num_classes), dtype=np.int32)

            for i in range(train_num_graphs):
                train_feed_dict = construct_feed_dict(train_sparse_features[i],
                                                      support,
                                                      train_one_hot_labels[i],
                                                      train_graph_weights[i],
                                                      placeholders)
                train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                _, loss, acc, out = sess.run([model.opt_op,
                                              model.loss,
                                              model.accuracy,
                                              model.outputs],
                                             feed_dict=train_feed_dict)
                # if np.max(out, 1)[0] < 0.5:
                #     train_acc_classes[train_labels[i], -1] += 1
                # else:
                train_acc_classes[train_labels[i], np.argmax(out, 1)[0]] += 1
                cnt += acc
                sum_loss += loss
            sum_acc = cnt / float(train_num_graphs)
            sum_sum_loss = sum_loss / float(train_num_graphs)
            print('Epoch {}:'.format(epoch + 1),
                  'acc={:.4f}, loss={:.4f}'.format(sum_acc, sum_sum_loss))
            # sum_acc = sess.run(sum_acc)
            # sum_sum_loss = sess.run(sum_sum_loss)

        test_acc_classes = np.zeros((test_num_classes, test_num_classes), dtype=np.int32)
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
            if np.max(out, 1)[0] < 0.5:
                test_acc_classes[test_labels[i], -1] += 1
            else:
                test_acc_classes[test_labels[i], np.argmax(out, 1)[0]] += 1
            cnt += acc
        test_acc = cnt / float(test_num_graphs)
        test_result.append(test_acc)
        print('Test accuracy: {:.4f}'.format(test_acc))
        # print('train confusion matrix: \n', train_acc_classes)
        # print('test confusion matrix: \n', test_acc_classes)
        train_precision = np.zeros((train_num_classes,))
        train_recall = np.zeros((train_num_classes,))
        test_precision = np.zeros((test_num_classes,))
        test_recall = np.zeros((test_num_classes,))
        train_f1 = np.zeros((train_num_classes,))
        test_f1 = np.zeros((test_num_classes,))
        for i in range(train_num_classes):
            train_precision[i] = train_acc_classes[i, i] / sum(train_acc_classes[:, i])
            train_recall[i] = train_acc_classes[i, i] / sum(train_acc_classes[i, :])
            train_f1[i] = (2 * train_recall[i] * train_precision[i]) / (train_recall[i] + train_precision[i])

        for i in range(test_num_classes):
            test_precision[i] = test_acc_classes[i, i] / sum(test_acc_classes[:, i])
            test_recall[i] = test_acc_classes[i, i] / sum(test_acc_classes[i, :])
            test_f1[i] = (2 * test_recall[i] * test_precision[i]) / (test_recall[i] + test_precision[i])
        # test_acc = sess.run(test_acc)
        # summ = sess.run(summ)
        # writer.add_summary(summ, epoch)

        print("Optimization finished!")
        # print("Start Saving Embeddings")
        # node_representations(support, train_sparse_features, train_one_hot_labels, placeholders, num_nodes, model,
        #                      'train')
        # node_representations(support, test_sparse_features, test_one_hot_labels, placeholders, num_nodes, model, 'test')
        #
        # model.save()

    return train_precision, train_recall, test_precision, test_recall, train_f1, test_f1, train_acc_classes, test_acc_classes
