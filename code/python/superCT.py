# PBMC: train
# CBMC: test

from codes.GCN.load_dataset import load_pbmc_cbmc
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from codes.GCN.model import dense_model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.4, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('batch_size', 256, 'Size of batch')
flags.DEFINE_integer('early_stopping', 10, 'Patience for early stopping over validation set')
flags.DEFINE_float('val_prop', 0.2, 'proportion of training set aside for validation')


# data loading and splitting
train_features, train_labels, train_cell_types, \
test_features, test_labels, test_cell_types, gene_names, n_class = load_pbmc_cbmc()


train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,
                                                                          random_state=1,
                                                                          test_size=FLAGS.val_prop,
                                                                          stratify=train_labels)

train_class_dist = [np.sum(train_labels == i) for i in range(n_class)]
val_class_dist = [np.sum(val_labels == i) for i in range(n_class)]
test_class_dist = [np.sum(test_labels == i) for i in range(n_class)]

train_class_dist /= np.sum(train_class_dist)
val_class_dist /= np.sum(val_class_dist)
test_class_dist /= np.sum(test_class_dist)

train_weights = [1. / train_class_dist[label] for label in train_labels]
val_weights = [1. / val_class_dist[label] for label in val_labels]
test_weights = [1. / test_class_dist[label] for label in test_labels]

n_features = int(train_features.shape[1])
n_graphs = int(train_features.shape[0])

input_features = tf.placeholder(shape=[None, n_features], dtype=tf.float32)
sample_weights = tf.placeholder(shape=[None,], dtype=tf.float32)
labels = tf.placeholder(shape=[None,], dtype=tf.int64)
one_hot_labels = tf.one_hot(indices=labels, depth=n_class)

# model
num_hidden_units = [FLAGS.hidden1, FLAGS.hidden2]
activations = dense_model(input_features, hidden_dims=num_hidden_units, out_dim=n_class, model_name='superCT')

# loss and accuracy
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activations[-1],
#                                                               labels=one_hot_labels))

weighted_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=activations[-1],
                                                               onehot_labels=one_hot_labels,
                                                               weights=sample_weights))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(activations[-1], axis=1), labels), dtype=tf.float32))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
train_opt = optimizer.minimize(weighted_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train
    cost_val = []
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
        train_loss, train_acc = sess.run([weighted_loss, acc], feed_dict=train_feed_dict)

        # val stat
        val_feed_dict = dict()
        val_feed_dict.update({input_features: val_features, labels: val_labels, sample_weights: val_weights})
        val_loss, val_acc = sess.run([weighted_loss, acc], feed_dict=val_feed_dict)
        cost_val.append(val_loss)

        # test stat
        test_feed_dict = dict()
        test_feed_dict.update({input_features: test_features, labels: test_labels, sample_weights: test_weights})
        test_loss, test_acc = sess.run([weighted_loss, acc], feed_dict=test_feed_dict)

        print('Epoch {}:'.format(epoch + 1))
        print("train: loss={:.3f}, acc={:.3f}".format(train_loss, train_acc))
        print("val: loss={:.3f}, acc={:.3f}".format(val_loss, val_acc))
        print("test: loss={:.3f}, acc={:.3f}".format(test_loss, test_acc))
        print("---------------")

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping on epoch {}...".format(epoch + 1))
            break

    print('Optimization finished')
