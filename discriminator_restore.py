from __future__ import print_function
import tensorflow as tf
import dataset
import data_load
from HotSInfo import Heroes
import matplotlib.pyplot as plt
import numpy as np

train_matchups, train_results, train_maps, test_matchups, test_results, test_maps = data_load.get_data_winrate_estimator(filterByMMR = True, averageMMR = 3000)

test_team = train_matchups.reshape( -1, 100 )
test_res = train_results

def discriminate(test_x, test_y):
    # Launch the graph
    mygraph = tf.Graph()
    with tf.Session(graph = mygraph) as sess:
        
        x = tf.placeholder(tf.float32, [None, 100], name = "x")
        keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
        
        # Create model
        # Heroes from both teams share the same set of weights for the first layers.
        # Maps are added as a secondary input that is added to the first layer
        # Siamese architecture: feature extraction to characterize the teams on the 4 first layers, followed by classification
        def conv_net_siamese(x, weights, biases, dropout):
            with tf.device('/gpu:0'):
                teams = tf.reshape(x, [-1, weights['heroes_w'].get_shape().as_list()[0]])
                teams = tf.matmul(teams, tf.abs(weights['heroes_w']))
                fc1 = tf.add( teams, biases['b1'] )
                fc1 = tf.nn.relu(fc1)
                fc1 = tf.nn.dropout(fc1, dropout)
                
                fs2 = tf.add(tf.matmul(fc1, weights['heroes_w2']), biases['bw2'])
                fs2 = tf.nn.relu(fs2)
                fs2 = tf.nn.dropout(fs2, dropout)
                
                fs3 = tf.add(tf.matmul(fs2, weights['heroes_w3']), biases['bw3'])
                fs3 = tf.nn.relu(fs3)
                fs3 = tf.nn.dropout(fs3, dropout)
                
                fs4 = tf.add(tf.matmul(fs3, weights['heroes_w4']), biases['bw4'])
                fs4 = tf.nn.relu(fs4)
                fs4 = tf.nn.dropout(fs4, dropout)
            
                out = tf.add(tf.matmul(fs4, weights['out']), biases['out'])
            return out
        
        # Store layers weight & bias
        weights = {
            'heroes_w': tf.get_variable( 'hw1', shape=[100, 1024], initializer=tf.contrib.layers.xavier_initializer()),
            'heroes_w2': tf.get_variable( 'hw2', shape=[1024, 512], initializer=tf.contrib.layers.xavier_initializer()),
            'heroes_w3': tf.get_variable( 'hw3', shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer()),
            'heroes_w4': tf.get_variable( 'hw4', shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable( 'wout', shape=[128, 2], initializer=tf.contrib.layers.xavier_initializer()),
        }
        
        biases = {
            'b1': tf.get_variable( 'b1', shape=[1024], initializer=tf.zeros_initializer()),
            'bw2': tf.get_variable( 'bw2', shape=[512], initializer=tf.zeros_initializer()),
            'bw3': tf.get_variable( 'bw3', shape=[256], initializer=tf.zeros_initializer()),
            'bw4': tf.get_variable( 'bw4', shape=[128], initializer=tf.zeros_initializer()),
            'out': tf.get_variable( 'bout', shape=[2], initializer=tf.zeros_initializer()),
        }
        
        # Construct model
        pred = tf.nn.softmax(conv_net_siamese(x, weights, biases, keep_prob), name="y")

        print('Session started')
        #If we want to resume training from a particular point
        saver = tf.train.Saver()
    
        saver.restore(sess,'C:\\Users\\Daniel\\.spyder-py3\\discriminator')
        prediction = sess.run(pred, feed_dict={x: test_x, keep_prob: 1.0})
        acc = 0
        i = 0
        counts = 0
        for game in test_y:
            if prediction[i][0] - prediction[i + 1][0] > 0.8 and game[0] == 1:
                acc +=1
                counts += 1
            elif prediction[i][0] - prediction[i + 1][0] > 0.8 and game[0] == 0:
                counts += 1
            elif prediction[i + 1][0] - prediction[i][0] > 0.8 and game[1] == 1:
                acc +=1
                counts += 1
            elif prediction[i + 1][0] - prediction[i][0] > 0.8 and game[1] == 0:
                counts += 1
            i+=2
        acc /= counts
        print("Accuracy:", \
        acc)
        print(counts)
    
        return prediction
    
    
team1 = [0]*100
team2 = [0]*100
holder = []
holder.append(team1)
holder.append(team2)
discriminate(test_team, test_res)