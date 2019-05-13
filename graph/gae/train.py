from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE, OptimizerHVAE
from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE, GCNModelHVAE, GCNModelHVAE_FF
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

tf.logging.set_verbosity(tf.logging.INFO)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('mode', 'train', 'Train/test switch')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('lr_adam', 0.01, 'Initial learning rate.')
flags.DEFINE_float('lr_rsgd', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('z_dim', 8, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('mmd_beta', 1e3, 'MMD lagrange multiplier.')
flags.DEFINE_boolean('sample_expmap', False, 'Use samples obtained via exponential map of Eucl. samples')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('expname', '', 'Short experiment tag')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

model_str = FLAGS.model
dataset_str = FLAGS.dataset
checkpoint_dir = "./checkpoints"
checkpoint_dir = os.path.join(checkpoint_dir, FLAGS.expname)
best_model_ckpt_dir = os.path.join(checkpoint_dir,'best_model')
if not os.path.exists(best_model_ckpt_dir):
  os.makedirs(best_model_ckpt_dir)

# Load data
adj, features = load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == 'gcn_hvae':
    model = GCNModelHVAE(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == 'gcn_hvae_ff':
    model = GCNModelHVAE_FF(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)
    elif 'gcn_hvae' in model_str:
        opt = OptimizerHVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           mmd_beta=FLAGS.mmd_beta)

def get_roc_score(sess, edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

# Initialize session
global_step = tf.train.get_or_create_global_step()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
# Construct feed dictionary for both train and test
feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)

# Train model 

saver = tf.train.Saver()
best_saver = tf.train.Saver()
saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=checkpoint_dir,
      save_steps=100,
      saver=saver)
hooks = [saver_hook]

## Training and validation ##

def train():
  with tf.train.SingularMonitoredSession(hooks=hooks,checkpoint_dir=checkpoint_dir) as session:
    gs = session.run(global_step)
    print("Global Step: ", gs)
    best_valid = 0
    for epoch in range(FLAGS.epochs):

        t = time.time()
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        #mmd,res1,res2,res3 = sess.run([opt.mmd_loss, opt.res1, opt.res2, opt.res3], feed_dict=feed_dict)
        outs = session.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr, ap_curr = get_roc_score(session, val_edges, val_edges_false)
        val_roc_score.append(roc_curr)

        if roc_curr>best_valid:
          print("Saving new best validation ROC score: ",roc_curr)
          best_valid = roc_curr
          best_model_ckpt = os.path.join(best_model_ckpt_dir,'best_model')
          best_saver.save(session.raw_session(), best_model_ckpt, global_step=global_step)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))

  print("Optimization Finished!")

def test():
  ## Testing ##
  # perform test: reset graph and load from best checkpoint 
  print("Testing best model")
  # tf.reset_default_graph()
  with tf.train.SingularMonitoredSession(hooks=[],checkpoint_dir=best_model_ckpt_dir) as session:
    roc_score, ap_score = get_roc_score(session, test_edges, test_edges_false)
    exp_string = [FLAGS.dataset + ',lr:{},do:{},beta:{},zdim:{}'.format(FLAGS.lr_adam,FLAGS.dropout,FLAGS.mmd_beta,FLAGS.hidden2)]
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))

    d = {'ROC': roc_score, 'AP': ap_score}
    import pandas as pd 
    from openpyxl import load_workbook
    df = pd.DataFrame(d,exp_string)
    sheetname = FLAGS.dataset
    writer = pd.ExcelWriter('results.xlsx')

    if os.path.isfile('results.xlsx'):
        writer.book = load_workbook('results.xlsx')
        writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
        df.to_excel(writer,sheetname,startrow=writer.sheets[sheetname].max_row,header=None)
    else:
        df.to_excel(writer,sheetname)
    writer.save()

if FLAGS.mode == 'train':
  train()
else:
  test()
