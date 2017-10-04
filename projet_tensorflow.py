from __future__ import print_function
import tensorflow as tf
import numpy as np
import csv
import os

# Parameters
base_learning_rate = 0.001
training_iters = 50000000
batch_size = 10000
display_step = 10

# Network Parameters
n_input = 100 * 2 # Number of possible heroes * number of teams
n_classes = 2 #Team 1 or team 2 win
dropout = 1 # Dropout, probability to keep units

#Dataset class adapted from the mnist example. Mostly useful for its minibatch method
class DataSet(object):

    def __init__(self,
           teams,
           maps,
           labels):
        assert teams.shape[0] == labels.shape[0], (
        'teams.shape: %s labels.shape: %s' % (teams.shape, labels.shape))
        self._num_examples = teams.shape[0]
        self._teams = teams
        self._maps = maps
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def teams(self):
        return self._teams
    
    @property
    def maps(self):
        return self._maps

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffling=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffling:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._teams = self.teams[perm0]
            self._maps = self.maps[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            teams_rest_part = self._teams[start:self._num_examples]
            maps_rest_part = self._maps[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffling:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._teams = self.teams[perm]
                self._maps = self.maps[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            teams_new_part = self._teams[start:end]
            maps_new_part = self._maps[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((teams_rest_part, teams_new_part), axis=0) , np.concatenate((maps_rest_part, maps_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._teams[start:end], self._maps[start:end], self._labels[start:end]

#Load the HotsLogs data.
def get_data():
#    if os.path.exists('games_matchup.npy'):
#        games_matchup = np.load('games_matchup.npy')
#        games_result = np.load('games_result.npy')
#        maps = np.load('maps.npy')
#        
#        test_games_matchup = np.load('test_games_matchup.npy')
#        test_games_result = np.load('test_games_result.npy')
#        test_maps = np.load('test_maps.npy')
#    else:
    games_matchup = []
    games_id = []
    games_result = []
    maps = []
    
    test_games_matchup = []
    test_games_result = []
    test_maps = []
    
    game_winner = [0]*100
    game_loser = [0]*100     
    
    with open( 'C:/Users/Daniel/Favorites/Downloads/HOTSLogs_Data_Export_Current_ana/ReplayCharacters.csv' ) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        idx = 1
        mmr = 0
        for row in reader:
            #if we want to discriminate by mmr
            stringmmr = row[5]
            for i in stringmmr.split():
                try:
                    mmr += int(i)
                except ValueError:
                    pass
                
            if int(row[4]) == 1:
                game_winner[int(row[2]) - 1] = 1
                
            elif int(row[4]) == 0:
                game_loser[int(row[2]) - 1] = 1
            
            #Training set
            if idx % 10 == 0 and idx > 500000:
                games_id.append( row[0] )
                
                #games must have 5 winners and 5 losers
                assert sum(game_winner) == 5, ('not 5 winners')
                game_winner_arr = np.array( game_winner )
                assert sum(game_loser) == 5, ('not 5 losers')
                game_loser_arr = np.array( game_loser )
                
                #For each game we double the training set by mirroring the matchup and results
                game_matchupA = []
                game_matchupA.append(game_winner_arr)
                game_matchupA.append(game_loser_arr)
                games_result.append([1,0])
                
                game_matchupB = []
                game_matchupB.append(game_loser_arr)
                game_matchupB.append(game_winner_arr)
                games_result.append([0,1])
                
                game_matchupC = []
                game_matchupC.append(game_loser_arr)
                game_matchupC.append(game_loser_arr)
                games_result.append([0.5,0.5])
                
                game_matchupD = []
                game_matchupD.append(game_winner_arr)
                game_matchupD.append(game_winner_arr)
                games_result.append([0.5,0.5])
                    
                    
                games_matchup.append( game_matchupA )
                games_matchup.append( game_matchupB )
                games_matchup.append( game_matchupC )
                games_matchup.append( game_matchupD )

                game_winner = [0]*100
                game_loser = [0]*100
                mmr = 0
            
            #Testing set
            elif idx % 10 == 0 and idx <= 500000:
                games_id.append(row[0])
                
                assert sum(game_winner) == 5, ('not 5 winners')
                game_winner_arr = np.array( game_winner )
                assert sum(game_loser) == 5, ('not 5 losers')
                game_loser_arr = np.array( game_loser )
                
                game_matchupA = []
                game_matchupA.append(game_winner_arr)
                game_matchupA.append(game_loser_arr)
                test_games_result.append([1,0])
                
                game_matchupB = []
                game_matchupB.append(game_loser_arr)
                game_matchupB.append(game_winner_arr)
                test_games_result.append([0,1])
                    
                test_games_matchup.append( game_matchupA )
                test_games_matchup.append( game_matchupB )

                game_winner = [0]*100
                game_loser = [0]*100
                mmr = 0
            
            #if we screen by mmr, reset the game arrays
            elif idx % 10 == 0:
                game_winner = [0]*100
                game_loser = [0]*100
                mmr = 0  
                
            idx += 1
#                if idx >1000000:
#                    break
    
    #Load the maps associated with the the matchups
    with open( 'C:/Users/Daniel/Favorites/Downloads/HOTSLogs_Data_Export_Current_ana/Replays.csv' ) as mapsfile:
        mapsreader = csv.reader(mapsfile, delimiter=',')
        next(mapsreader)
        idx = 0
        print( len(games_id) )
        for row in mapsreader:
            if( row[0] == games_id[idx]):
                actual_map = [0]*13
                actual_map1 = [0]*13
                if int( row[2] ) == 1001:
                    actual_map[0] = 1
                    actual_map1[0] = 1
                
                elif int( row[2] ) == 1002:
                    actual_map[1] = 1
                    actual_map1[1] = 1
                
                elif int( row[2] ) == 1003:
                    actual_map[2] = 1
                    actual_map1[2] = 1
                
                elif int( row[2] ) == 1004:
                    actual_map[3] = 1
                    actual_map1[3] = 1
                
                elif int( row[2] ) == 1005:
                    actual_map[4] = 1
                    actual_map1[4] = 1
                
                elif int( row[2] ) == 1006:
                    actual_map[5] = 1
                    actual_map1[5] = 1
                
                elif int( row[2] ) == 1007:
                    actual_map[6] = 1
                    actual_map1[6] = 1
                
                elif int( row[2] ) == 1008:
                    actual_map[7] = 1
                    actual_map1[7] = 1
                
                elif int( row[2] ) == 1009:
                    actual_map[8] = 1
                    actual_map1[8] = 1
                
                elif int( row[2] ) == 1010:
                    actual_map[9] = 1
                    actual_map1[9] = 1
                
                elif int( row[2] ) == 1012:
                    actual_map[10] = 1
                    actual_map1[10] = 1
                
                elif int( row[2] ) == 1013:
                    actual_map[11] = 1
                    actual_map1[11] = 1
                
                elif int( row[2] ) == 1016:
                    actual_map[12] = 1
                    actual_map1[12] = 1
                #Testing set
                #Maps are added twice to match the number of teams ( 2 team per game )
                if idx < len(test_games_result) / 2:
                    map_copy = []
                    map_copy.append(actual_map)
                    map_copy.append(actual_map1)
                    test_maps.append( map_copy )
                    test_maps.append( map_copy )
                #Training Set
                else:
                    map_copy = []
                    map_copy.append(actual_map)
                    map_copy.append(actual_map1)
                    maps.append( map_copy )
                    maps.append( map_copy )
                    maps.append( map_copy )
                    maps.append( map_copy )
                idx += 1
                if len(games_result) == len(maps):
                    break
                
#    np.save('games_matchup.npy',  np.array(games_matchup))
#    np.save('games_result.npy',  np.array(games_result))
#    np.save('maps.npy',  np.array(maps))
#    np.save('test_games_matchup.npy',  np.array(test_games_matchup))
#    np.save('test_games_result.npy',  np.array(test_games_result))
#    np.save('test_maps.npy',  np.array(test_maps))
    return np.array(games_matchup), np.array(games_result), np.array(maps), np.array(test_games_matchup), np.array(test_games_result), np.array(test_maps)
            
#Load HotsLogs data
train_matchups, train_results, train_maps, test_matchups, test_results, test_maps = get_data()

print( train_matchups.shape )
print( train_results.shape )
print( train_maps.shape )

print( test_matchups.shape )
print( test_results.shape )
print( test_maps.shape )

#Make sure the shapes fit the neural network
X = train_matchups.reshape( -1, 2, 100 )
Y = train_results
maps = train_maps.reshape( -1, 2, 13 )

test_x = test_matchups.reshape( -1, 2, 100 )
test_y = test_results
test_maps = test_maps.reshape( -1, 2, 13 )

#Create datasets
trainSet = DataSet( X, maps, Y )
testSet = DataSet( test_x, test_maps, test_y )

print('Dataset created')

# tf Graph input
x = tf.placeholder(tf.float32, [None, 2, 100])
m = tf.placeholder(tf.float32, [None, 2, 13])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
# Heroes from both teams share the same set of weights for the first layer.
# Maps are added as a secondary input that is added to the first layer
# 4 layers, ReLu activation
def conv_net(x, maps, weights, biases, dropout):

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
    'maps_w': tf.get_variable( 'mw1', shape=[13, 100], initializer=tf.contrib.layers.xavier_initializer()),
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
pred = conv_net(x, m, weights, biases, keep_prob)

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
with tf.Session() as sess:
    sess.run(init)
    print('Session started')
    step = 1
    #If we want to resume training from a particular point
    saver = tf.train.Saver()
    
    saver.restore(sess,'C:\\Users\\Daniel\\.spyder-py3\\siamese_abs')
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_map, batch_y = trainSet.next_batch( batch_size )
        test_x, test_map, test_y = testSet.next_batch( batch_size )
        # Run optimization op (backprop)
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
            
            print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_x,
                                          m: test_map,
                                          y: test_y,
                                          keep_prob: 1.}))
    
    
        if step % (1000000 / batch_size) == 0:
            saver = tf.train.Saver()
            #Now, save the gr
            saver.save(sess, 'C:\\Users\\Daniel\\.spyder-py3\\siamese_abs')
            #w2 = sess.run(weights['w2'])
            #print(w2)
            
            #w3 = sess.run(weights['w3'])
            #print(w3)
            
            #w4 = sess.run(weights['w4'])aph
            #print(w4)
            
        step += 1

    print("Optimization Finished!")
    
    saver = tf.train.Saver()
    #Now, save the graph
    saver.save(sess, 'C:\\Users\\Daniel\\.spyder-py3\\siamese_abs')
    
