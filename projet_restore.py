from __future__ import print_function
import tensorflow as tf
from HotSInfo import Heroes, Maps, create_game
from mcts.mcts import MCTS
from mcts.default_policies import random_terminal_roll_out
from mcts.backups import monte_carlo
from mcts.tree_policies import UCB1
from mcts.graph import StateNode
import draftBoard
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from copy import deepcopy

# Network Parameters
n_input = 100 * 2 # MNIST data input (img shape: 128*128)
n_classes = 2 # dog or cat
dropout = 1 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 2, 100])
m = tf.placeholder(tf.float32, [None, 2, 20])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
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
pred = tf.nn.softmax(conv_net_siamese(x, m, weights, biases, keep_prob))

# Initializing the variables
init = tf.global_variables_initializer()

print('Parameters initialized')

#IF YOU WANT TO ESTIMATE THE WINRATE OF A SPECIFIC GAME
myGame, myMap = create_game( Heroes.AURIEL.value, 
                             Heroes.SONYA.value, 
                             Heroes.LEORIC.value, 
                             Heroes.SYLVANAS.value, 
                             Heroes.STUKOV.value,
                             Heroes.RAGNAROS.value, 
                             Heroes.MALFURION.value, 
                             Heroes.KELTHUZAD.value, 
                             Heroes.DEHAKA.value, 
                             Heroes.SAMURO.value,
                             Maps.BRAXIS.value)


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print('Session started')

    saver = tf.train.Saver()
    saver.restore(sess,'C:\\Users\\Daniel\\.spyder-py3\\siamese')
    
    w = sess.run(tf.abs(weights['heroes_w']))
    tf.get_default_graph().finalize()
    #Evaluation function to pass to the MCTS. Has to be defined outside of DraftBoard to prevent graph modification that can slow execution
    def evalFunc(games, mapss):
        result = sess.run(pred, feed_dict={x: games,
                                  m: mapss,
                                  keep_prob: 1.})
        return result
    
    #add picked heroes
    team1 = [0]*100
#    team1[Heroes.AURIEL.value] = 1
#    team1[Heroes.SONYA.value] = 1
#    team1[Heroes.LEORIC.value] = 1
#    team1[Heroes.SYLVANAS.value] = 1
#    team1[Heroes.STUKOV.value] = 1
    team2 = [0]*100
#    team2[Heroes.RAGNAROS.value] = 1
#    team2[Heroes.MALFURION.value] = 1
#    team2[Heroes.KELTHUZAD.value] = 1
#    team2[Heroes.DEHAKA.value] = 1
#    team2[Heroes.SAMURO.value] = 1
    pool = [1]*72
    #Need to ban CHOGALL because he doesn't behave the same way in draft as other heroes.
    #Still needs to be implemented
    pool[Heroes.CHO.value] = 0
    pool[Heroes.GALL.value] = 0
    #Add picked or banned heroes here
    pool[Heroes.AURIEL.value] = 0
    pool[Heroes.RAGNAROS.value] = 0
    pool[Heroes.MALFURION.value] = 0
    pool[Heroes.SONYA.value] = 0
    pool[Heroes.LEORIC.value] = 0
    pool[Heroes.ANUBARAK.value] = 0
    pool[Heroes.ANA.value] = 0
    pool[Heroes.KELTHUZAD.value] = 0
    pool[Heroes.DEHAKA.value] = 0
    pool[Heroes.SYLVANAS.value] = 0
#    pool[Heroes.GULDAN.value] = 0
#    pool[Heroes.ZAGARA.value] = 0
#    pool[Heroes.SAMURO.value] = 0
    mymap = [0]*20
    mymap[10] = 1
    #Inputs are: team1, team2, unpickable heroes, map, pick phase (1-15), team to pick
    boardState = draftBoard.DraftState(team1, team2, pool, mymap, 14, 2, evalFunc)
    algo = MCTS(tree_policy=UCB1(c=1.41), 
        default_policy=random_terminal_roll_out,
        backup=monte_carlo)
    
    #TO GET THE 4 BEST PICKS ACCORDING TO THE MCTS
#    result = algo(StateNode(None,boardState), n = 10000)
#    for hero in result:
#        print(Heroes(hero.action).name)
    #print( Heroes(draftBoard.random_search(boardState, 40000)).name )
    
    #TO MAP THE HEROES CLUSTERING
#    w = w[:-28]
#    linkage = scipy.cluster.hierarchy.linkage(w, method='ward')
#    hero_label = []
#    for a in range(0,72):
#        hero_label.append(Heroes(a).name)
#
#    plt.title('RMSD Average linkage hierarchical clustering')
#    _ = scipy.cluster.hierarchy.dendrogram(linkage, labels = hero_label, count_sort='descendent', leaf_font_size = 12)

    #TO PREDICT TO RESULT OF A GAME
    #print( weights['heroes_w'] ) 
    
#    print("Predicted:", \
#    sess.run(pred, feed_dict={x: myGame,
#                                 m: myMap,
#                                 keep_prob: 1.}))
