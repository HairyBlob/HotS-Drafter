from __future__ import print_function
import tensorflow as tf
import dataset
import data_load

# Parameters
base_learning_rate = 0.0001
training_iters = 50000000
batch_size = 10000
display_step = 10

# Network Parameters
n_input = 100 * 2 # Number of possible heroes * number of teams
n_classes = 2 #Team 1 or team 2 win
dropout = 1 # Dropout, probability to keep units

#Load HotsLogs data
train_matchups, train_results, train_maps, test_matchups, test_results, test_maps = data_load.get_data()

#Verify data shape
print( train_matchups.shape )
print( train_results.shape )
print( train_maps.shape )

print( test_matchups.shape )
print( test_results.shape )
print( test_maps.shape )

#Make sure the shapes fit the neural network
X = train_matchups.reshape( -1, 2, 100 )
Y = train_results
maps = train_maps.reshape( -1, 2, 20 )

test_x = test_matchups.reshape( -1, 2, 100 )
test_y = test_results
test_maps = test_maps.reshape( -1, 2, 20 )

#Create datasets
trainSet = dataset.DataSet( X, maps, Y )
testSet = dataset.DataSet( test_x, test_maps, test_y )

print('Dataset created')

# tf Graph input
x = tf.placeholder(tf.float32, [None, 2, 100])
m = tf.placeholder(tf.float32, [None, 2, 20])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
# Heroes from both teams share the same set of weights for the first layer.
# Maps are added as a secondary input that is added to the first layer
# 4 layers, ReLu activation
def conv_net_siamese(x, maps, weights, biases, dropout):
    with tf.device('/gpu:0'):
        teams = tf.reshape(x, [-1, weights['heroes_w'].get_shape().as_list()[0]])
        teams = tf.matmul(teams, tf.abs(weights['heroes_w']))
        maps = tf.reshape(maps, [-1, weights['maps_w'].get_shape().as_list()[0]])
        teams = tf.multiply(teams, tf.matmul(maps, tf.abs(weights['maps_w'])))
        fc1 = tf.add( teams, biases['b1'] )
        fc1 = tf.nn.relu(fc1)
        
        fs2 = tf.add(tf.matmul(fc1, weights['heroes_w2']), biases['bw2'])
        fs2 = tf.nn.relu(fs2)
        fs2 = tf.nn.dropout(fs2, dropout)
        
        fs3 = tf.add(tf.matmul(fs2, weights['heroes_w3']), biases['bw3'])
        fs3 = tf.nn.relu(fs3)
        fs3 = tf.nn.dropout(fs3, dropout)
        
        fs4 = tf.add(tf.matmul(fs3, weights['heroes_w4']), biases['bw4'])
        fs4 = tf.nn.relu(fs4)
        fs4 = tf.nn.dropout(fs4, dropout)
    
        fc2 = tf.reshape(fs4, [-1, 200])
        fc2 = tf.add(tf.matmul(fc2, weights['w2']), biases['b2'])
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, dropout)
    
        fc3 = fc2
        fc3 = tf.add(tf.matmul(fc3, weights['w3']), biases['b3'])
        fc3 = tf.nn.relu(fc3)
        fc3 = tf.nn.dropout(fc3, dropout)
    
        fc4 = fc3
        fc4 = tf.add(tf.matmul(fc4, weights['w4']), biases['b4'])
        fc4 = tf.nn.relu(fc4)
        fc4 = tf.nn.dropout(fc4, dropout)
    
        out = tf.add(tf.matmul(fc4, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    'heroes_w': tf.get_variable( 'hw1', shape=[100, 100], initializer=tf.contrib.layers.xavier_initializer()),
    'heroes_w2': tf.get_variable( 'hw2', shape=[100, 100], initializer=tf.contrib.layers.xavier_initializer()),
    'heroes_w3': tf.get_variable( 'hw3', shape=[100, 100], initializer=tf.contrib.layers.xavier_initializer()),
    'heroes_w4': tf.get_variable( 'hw4', shape=[100, 100], initializer=tf.contrib.layers.xavier_initializer()),
    'maps_w': tf.get_variable( 'mw1', shape=[20, 100], initializer=tf.contrib.layers.xavier_initializer()),
    'w2': tf.get_variable( 'w2', shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer()),
    'w3': tf.get_variable( 'w3', shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer()),
    'w4': tf.get_variable( 'w4', shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable( 'wout', shape=[200, n_classes], initializer=tf.contrib.layers.xavier_initializer()),
}

biases = {
    'b1': tf.get_variable( 'b1', shape=[100], initializer=tf.contrib.layers.xavier_initializer()),
    'bw2': tf.get_variable( 'bw2', shape=[100], initializer=tf.contrib.layers.xavier_initializer()),
    'bw3': tf.get_variable( 'bw3', shape=[100], initializer=tf.contrib.layers.xavier_initializer()),
    'bw4': tf.get_variable( 'bw4', shape=[100], initializer=tf.contrib.layers.xavier_initializer()),
    'b2': tf.get_variable( 'b2', shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
    'b3': tf.get_variable( 'b3', shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
    'b4': tf.get_variable( 'b4', shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable( 'bout', shape=[n_classes], initializer=tf.contrib.layers.xavier_initializer()),
}

# Construct model
pred = conv_net_siamese(x, m, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=base_learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

print('Parameters initialized')

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    print('Session started')
    step = 1
    #If we want to resume training from a particular point
    saver = tf.train.Saver()
    
    saver.restore(sess,'C:\\Users\\Daniel\\.spyder-py3\\siamese_abs_big')
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_map, batch_y = trainSet.next_batch( batch_size )
        test_x, test_map, test_y = testSet.next_batch( batch_size )
        # Run optimization op (backprop)
        with tf.device('/gpu:0'):
            sess.run(optimizer, feed_dict={x: batch_x, m:batch_map, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              m: batch_map,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            
            test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: test_x,
                                                              m: test_map,
                                                              y: test_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(test_loss) + ", Testing Accuracy= " + \
                  "{:.5f}".format(test_acc))
    
    
        if step % (1000000 / batch_size) == 0:
            #Now, save the gr
            saver.save(sess, 'C:\\Users\\Daniel\\.spyder-py3\\siamese')
            #w2 = sess.run(weights['w2'])
            #print(w2)
            
            #w3 = sess.run(weights['w3'])
            #print(w3)
            
            #w4 = sess.run(weights['w4'])aph
            #print(w4)
            
        step += 1

    print("Optimization Finished!")


    #Now, save the graph
    saver.save(sess, 'C:\\Users\\Daniel\\.spyder-py3\\siamese')
    
