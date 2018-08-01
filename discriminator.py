from __future__ import print_function
import tensorflow as tf
import dataset
import data_load
from graph_freezer import freezing_graph

def train_discriminator():
#    # Parameters
#    base_learning_rate = 0.0001
#    training_iters = 50000000
#    batch_size = 10000
#    display_step = 100
#    dropout = 0.8
#    
#    # Network Parameters
#    n_classes = 2 #Team 1 or team 2 win
#    
#    #Load HotsLogs data
#    train_matchups, train_results, test_matchups, test_results = data_load.get_data_discriminator(filterByMMR = True, averageMMR = 2500)
#    
#    #Verify data shape
#    print( train_matchups.shape )
#    print( train_results.shape )
#    
#    print( test_matchups.shape )
#    print( test_results.shape )
#    
#    #Make sure the shapes fit the neural network
#    X = train_matchups
#    Y = train_results
#    
#    test_x = test_matchups
#    test_y = test_results
#    
#    #Create datasets
#    trainSet = dataset.DataSet_discriminator( X, Y )
#    testSet = dataset.DataSet_discriminator( test_x, test_y )
#    
#    print('Dataset created')
#    
#    # tf Graph input
#    # Created 20 maps and 100 heroes to avoid having to change the model when a new map or hero is introduced
#    x = tf.placeholder(tf.float32, [None, 100], name = "x")
#    y = tf.placeholder(tf.float32, [None, n_classes])
#    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
#    
#    # Create model
#    # Heroes from both teams share the same set of weights for the first layers.
#    # Maps are added as a secondary input that is added to the first layer
#    # Siamese architecture: feature extraction to characterize the teams on the 4 first layers, followed by classification
#    def conv_net_siamese(x, weights, biases, dropout):
#        with tf.device('/gpu:0'):
#            teams = tf.reshape(x, [-1, weights['heroes_w'].get_shape().as_list()[0]])
#            teams = tf.matmul(teams, tf.abs(weights['heroes_w']))
#            fc1 = tf.add( teams, biases['b1'] )
#            fc1 = tf.nn.relu(fc1)
#            fc1 = tf.nn.dropout(fc1, dropout)
#            
#            fs2 = tf.add(tf.matmul(fc1, weights['heroes_w2']), biases['bw2'])
#            fs2 = tf.nn.relu(fs2)
#            fs2 = tf.nn.dropout(fs2, dropout)
#            
#            fs3 = tf.add(tf.matmul(fs2, weights['heroes_w3']), biases['bw3'])
#            fs3 = tf.nn.relu(fs3)
#            fs3 = tf.nn.dropout(fs3, dropout)
#            
#            fs4 = tf.add(tf.matmul(fs3, weights['heroes_w4']), biases['bw4'])
#            fs4 = tf.nn.relu(fs4)
#            fs4 = tf.nn.dropout(fs4, dropout)
#        
#            out = tf.add(tf.matmul(fs4, weights['out']), biases['out'])
#        return out
#    
#    # Store layers weight & bias
#    weights = {
#        'heroes_w': tf.get_variable( 'hw1', shape=[100, 1024], initializer=tf.contrib.layers.xavier_initializer()),
#        'heroes_w2': tf.get_variable( 'hw2', shape=[1024, 512], initializer=tf.contrib.layers.xavier_initializer()),
#        'heroes_w3': tf.get_variable( 'hw3', shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer()),
#        'heroes_w4': tf.get_variable( 'hw4', shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer()),
#        'out': tf.get_variable( 'wout', shape=[128, n_classes], initializer=tf.contrib.layers.xavier_initializer()),
#    }
#    
#    biases = {
#        'b1': tf.get_variable( 'b1', shape=[1024], initializer=tf.zeros_initializer()),
#        'bw2': tf.get_variable( 'bw2', shape=[512], initializer=tf.zeros_initializer()),
#        'bw3': tf.get_variable( 'bw3', shape=[256], initializer=tf.zeros_initializer()),
#        'bw4': tf.get_variable( 'bw4', shape=[128], initializer=tf.zeros_initializer()),
#        'out': tf.get_variable( 'bout', shape=[n_classes], initializer=tf.zeros_initializer()),
#    }
#    
#    # Construct model
#    pred = tf.nn.softmax(conv_net_siamese(x, weights, biases, keep_prob), name="y")
#    
#    # Define loss and optimizer
#    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=conv_net_siamese(x, weights, biases, keep_prob), labels=y)
#                                                                  + 0.001*tf.nn.l2_loss(weights['heroes_w'])
#                                                                  + 0.001*tf.nn.l2_loss(weights['heroes_w2'])
#                                                                  + 0.001*tf.nn.l2_loss(weights['heroes_w3'])
#                                                                  + 0.001*tf.nn.l2_loss(weights['heroes_w4'])
#                                                                  + 0.001*tf.nn.l2_loss(weights['out']))
#    optimizer = tf.train.AdamOptimizer(learning_rate=base_learning_rate).minimize(cost)
#    
#    # Evaluate model
#    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#    
#    # Initializing the variables
#    init = tf.global_variables_initializer()
#    
#    print('Parameters initialized')
#    
#    # Launch the graph
#    with tf.Session(graph = tf.get_default_graph()) as sess:
#        sess.run(init)
#        print('Session started')
#        step = 1
#        #If we want to resume training from a particular point
#        saver = tf.train.Saver()
#        
##        saver.restore(sess,'./disc_model/discriminator')
#        # Keep training until reach max iterations
#        while step * batch_size < training_iters:
#            batch_x, batch_y = trainSet.next_batch( batch_size )
#            test_x, test_y = testSet.next_batch( len(test_matchups) )
#            # Run optimization op (backprop)
#            with tf.device('/gpu:0'):
#                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
#            if step % display_step == 0:
#                # Calculate batch loss and accuracy
#                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                                                                  y: batch_y,
#                                                                  keep_prob: 1.0})
#                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                      "{:.5f}".format(acc))
#                
#                test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: test_x,
#                                                                  y: test_y,
#                                                                  keep_prob: 1.0})
#                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                      "{:.6f}".format(test_loss) + ", Testing Accuracy= " + \
#                      "{:.5f}".format(test_acc))
#        
#        
#            if step % (1000000 / batch_size) == 0:
#                #Now, save the graph
#                saver.save(sess, './disc_model/discriminator')
#
#            step += 1
#        
#        
#        print("Optimization Finished!")
#    
#    
#    
#        #Now, save the graph
#        
#        checkpoint_prefix = ("./discriminator_graph/discriminator_checkpoint")
#        saver.save(sess, checkpoint_prefix, global_step=0)
#        tf.train.write_graph(sess.graph_def, "./discriminator_graph", "discriminator_graph.pb")
        
        freezing_graph("discriminator")
        return 0
    
