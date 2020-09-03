
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from load_dataset import *
from utils import *
from model import *


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 30, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('batch_size', 256, 'Size of batch')
flags.DEFINE_integer('early_stopping', 10, 'Patience for early stopping over validation set')
flags.DEFINE_float('val_prop', 0.2, 'proportion of training set aside for validation')



# data directory
base_path      = 'data_train/pbmc'

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
with open("data_train/pbmc/train_idx_superCT.txt", "w") as f:
    for s in idx:
        f.write(str(s) +"\n")

# train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,
#                                                                           random_state=1,
#                                                                           test_size=FLAGS.val_prop,
#                                                                           stratify=train_labels)

train_class_dist = [np.sum(train_labels == i) for i in range(num_classes)]
train_weights = [1. / train_class_dist[label] for label in train_labels]
# val_class_dist = [np.sum(val_labels == i) for i in range(num_classes)]
# val_weights = [1. / val_class_dist[label] for label in val_labels]
n_features = int(train_features.shape[1])
n_graphs = int(train_features.shape[0])

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
idx               = np.arange(test_num_graphs)
np.random.shuffle(idx)
test_labels      = test_labels[idx]
test_one_hot_labels = test_one_hot_labels[idx, :]
test_features    = test_features[idx, :]
## for later analyses
with open("data_train/pbmc/test_idx_superCT.txt", "w") as f:
    for s in idx:
        f.write(str(s) +"\n")
test_class_dist = [np.sum(test_labels == i) for i in range(num_classes)]
test_weights = [1. / test_class_dist[label] for label in test_labels]


input_features = tf.placeholder(shape=[None, n_features], dtype=tf.float32)
sample_weights = tf.placeholder(shape=[None,], dtype=tf.float32)
labels = tf.placeholder(shape=[None,], dtype=tf.int64)
one_hot_labels = tf.one_hot(indices=labels, depth=num_classes)

# model
num_hidden_units = [FLAGS.hidden1, FLAGS.hidden2]
activations = dense_model(input_features, hidden_dims=num_hidden_units, out_dim=num_classes, model_name='superCT')
weighted_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=activations[-1],
                                                               onehot_labels=one_hot_labels,
                                                               weights=sample_weights))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(activations[-1], axis=1), labels), dtype=tf.float32))
out = tf.argmax(activations[-1], axis=1)
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
train_opt = optimizer.minimize(weighted_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train
    # cost_val = []
    for epoch in range(FLAGS.epochs):
        for batch in range(n_graphs // FLAGS.batch_size):

            batch_features = train_features[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
            batch_labels = train_labels[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
            batch_weights = train_weights[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
            # training
            train_feed_dict = dict()
            train_feed_dict.update({input_features: batch_features, labels: batch_labels, sample_weights: batch_weights})
            train_loss, train_acc, _ = sess.run([weighted_loss, acc, train_opt], feed_dict=train_feed_dict)

        # train stat
        train_feed_dict = dict()
        train_feed_dict.update({input_features: train_features, labels: train_labels, sample_weights: train_weights})
        train_loss, train_acc, train_out = sess.run([weighted_loss, acc, out], feed_dict=train_feed_dict)
        n_graphs = int(train_features.shape[0])
        train_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
        for i in range(n_graphs):
            train_acc_classes[train_labels[i], train_out[i]] += 1
        
        # val stat
        # val_feed_dict = dict()
        # val_feed_dict.update({input_features: val_features, labels: val_labels, sample_weights: val_weights})
        # val_loss, val_acc = sess.run([weighted_loss, acc], feed_dict=val_feed_dict)
        # cost_val.append(val_loss)

        # test stat
        test_feed_dict = dict()
        test_feed_dict.update({input_features: test_features, labels: test_labels, sample_weights: test_weights})
        test_loss, test_acc, test_out = sess.run([weighted_loss, acc, out], feed_dict=test_feed_dict)
        n_graphs = int(test_features.shape[0])
        test_acc_classes = np.zeros((num_classes, num_classes), dtype=np.int32)
        for i in range(n_graphs):
            test_acc_classes[test_labels[i], test_out[i]] += 1

        print('Epoch {}:'.format(epoch + 1))
        print("train: loss={:.3f}, acc={:.3f}".format(train_loss, train_acc))
        # print("val: loss={:.3f}, acc={:.3f}".format(val_loss, val_acc))
        print("test: loss={:.3f}, acc={:.3f}".format(test_loss, test_acc))
        print("---------------")
        print('train confusion matrix: \n', train_acc_classes)
        print('test confusion matrix: \n', test_acc_classes)

        # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #     print("Early stopping on epoch {}...".format(epoch + 1))
        #     break
    print('Optimization finished')
    print("Start Saving Embeddings")
    num_layers = 2
    # train embeddings
    n_graphs = int(train_features.shape[0])
    layer_embeddings = [np.zeros((n_graphs, 200)), np.zeros((n_graphs, 100))] 
    for i in range(n_graphs):
        feed_dict = dict()
        feed_dict.update({input_features: train_features[i:(i+1)], labels: train_labels[i:(i+1)], sample_weights: train_weights[i:(i+1)]})
        graph_embeddings = sess.run(activations[1:1 + num_layers], feed_dict=feed_dict)
        for layer in range(num_layers):
            layer_embeddings[layer][i, :] = graph_embeddings[layer]
    for layer in range(num_layers):
        embedding_df = pd.DataFrame(data=layer_embeddings[layer].T)
        embedding_df.to_csv('./models/pbmc/' + 'train' + '_layer_{}_FC.csv'.format(layer + 1))
    # test embeddings
    n_graphs = int(test_features.shape[0])
    layer_embeddings = [np.zeros((n_graphs, 200)), np.zeros((n_graphs, 100))] 
    for i in range(n_graphs):
        feed_dict = dict()
        feed_dict.update({input_features: test_features[i:(i+1)], labels: test_labels[i:(i+1)], sample_weights: test_weights[i:(i+1)]})
        graph_embeddings = sess.run(activations[1:1 + num_layers], feed_dict=feed_dict)
        for layer in range(num_layers):
            layer_embeddings[layer][i, :] = graph_embeddings[layer]
    for layer in range(num_layers):
        embedding_df = pd.DataFrame(data=layer_embeddings[layer].T)
        embedding_df.to_csv('./models/pbmc/' + 'test' + '_layer_{}_FC.csv'.format(layer + 1))
    # node_rep_fc(train_features, train_labels, train_weights, activations, 'train')
    # node_rep_fc(test_features, test_labels, test_weights, activations, 'test')

    
