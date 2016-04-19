import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import math

num_labels = 2
batch_size = 128
feature_size = 101
dropout_p = 0.8
decay_rate = 0.9
num_steps = 3001 

# convert 0 to [1.0, 0.0], 1 to [0.0, 1.0]
def reformat(labels):
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels
    
# Define the structure of the neural network
# input: X, output: Y, relu: rectifier activation function
# t1 = W1 * X + b1
# t2 = relu(a)  
# Y = W2 * t2 + b2
# use dropout to avoid overfitting
def inference(data, hidden1_units, keep_prob):
    #Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([feature_size, hidden1_units],
                            stddev=1.0 / math.sqrt(float(feature_size))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name = 'biases')
        hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)
#        regularizers = (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))
    #Output
    with tf.name_scope('Output'):
        weights = tf.Variable(
        tf.truncated_normal([hidden1_units, num_labels],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([num_labels]),
            name='biases')
        hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
        logits = tf.matmul(hidden1_drop, weights) + biases
#        regularizers += (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))
          
    return logits 

# Define cross entropy as loss function
def lossFun(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

# Define stochastic gradient descent as optimizer
# Apply exponential decay to the learning rate
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
def training(loss, global_step):
    learning_rate = tf.train.exponential_decay(0.5, global_step, 10000, decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def main():
    pickle_file = 'data.pickle'

    # Load prepared data
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    pickle_file = 'data_edge.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        edge_dataset = save['test_dataset']
        edge_labels = save['test_labels']
        del save
        print('Edge set', edge_dataset.shape, edge_labels.shape)
        
    train_labels = reformat(train_labels)
    valid_labels = reformat(valid_labels)
    test_labels = reformat(test_labels)
    edge_labels = reformat(edge_labels)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, feature_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
    
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)
    
        # set hidden layer with 1024 units
        logits = inference(tf_train_dataset, 1024, keep_prob)
        loss = lossFun(logits, tf_train_labels)
        optimizer = training(loss, global_step)
    
        train_prediction = tf.nn.softmax(logits)
     
    
    correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
        
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
        
            if (step % 500 == 0):
                l, mini_accuracy = session.run([loss, accuracy], 
                                               feed_dict={tf_train_dataset: batch_data, 
                                                          tf_train_labels: batch_labels, keep_prob: 1.0})
            
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % mini_accuracy)
                print("Validation accuracy: %.1f%%" % session.run(accuracy, 
                                                                  feed_dict={tf_train_dataset: valid_dataset, 
                                                                             tf_train_labels: valid_labels, keep_prob: 1.0}))
        
            session.run([loss, optimizer], feed_dict={tf_train_dataset: batch_data, 
                                                      tf_train_labels: batch_labels, keep_prob: dropout_p})
        
        print("Test accuracy: %.1f%%" % 
              session.run(accuracy, feed_dict={tf_train_dataset: test_dataset, tf_train_labels: test_labels, keep_prob: 1.0}))

        [accuracy, correct_prediction] = session.run([accuracy, correct_prediction], 
                 feed_dict={tf_train_dataset: edge_dataset, tf_train_labels: edge_labels, keep_prob: 1.0})
        print("Edge accuracy: %.1f%%" % accuracy)

    session.close()
    
    
