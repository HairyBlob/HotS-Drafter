from __future__ import print_function
from enum import Enum
import tensorflow as tf
import numpy as np         # dealing with arrays
import random 
from mcts.mcts import MCTS
from mcts.default_policies import random_terminal_roll_out
from mcts.backups import monte_carlo
from mcts.tree_policies import UCB1
from mcts.graph import StateNode

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from copy import deepcopy
# Parameters
base_learning_rate = 0.0001
training_iters = 30000000
batch_size = 5000
display_step = 10

class Heroes(Enum):
    ABATHUR = 0
    ANUBARAK = 1
    ARTHAS = 2
    AZMODAN = 3
    BRIGHTWING = 4
    CHEN = 5
    DIABLO = 6
    ETC = 7
    FALSTAD = 8
    GAZLOWE = 9
    ILIDAN = 10
    JAINA = 11
    JOHANNA = 12
    KAELTHAS = 13
    KERRIGAN = 14
    KHARAZIM = 15
    LEORIC = 16
    LILI = 17
    MALFURION = 18
    MURADIN = 19
    MURKY = 20
    NAZEEBO = 21
    NOVA = 22
    RAYNOR = 23
    REHGAR = 24
    HAMMER = 25
    SONYA = 26
    STITCHES = 27
    SYLVANAS = 28
    TASSADAR = 29
    BUTCHER = 30
    VIKINGS = 31
    THRALL = 32
    TYCHUS = 33
    TYRAEL = 34
    TYRANDE = 35
    UTHER = 36
    VALLA = 37
    ZAGARA = 38
    ZERATUL = 39
    REXXAR = 40
    MORALES = 41
    ARTANIS = 42
    CHO = 43
    GALL = 44
    LUNARA = 45
    GREYMANE = 46
    LIMING = 47
    XUL = 48
    DEHAKA = 49
    TRACER = 50
    CHROMIE = 51
    MEDIVH = 52
    GULDAN = 53
    AURIEL = 54
    ALARAK = 55
    ZARYA = 56
    SAMURO = 57
    VARIAN = 58
    RAGNAROS = 59
    ZULJIN = 60
    VALEERA = 61
    LUCIO = 62
    PROBIUS = 63
    CASSIA = 64
    GENJI = 65
    DVA = 66
    MALTHAEL = 67
    STUKOV = 68
    GARROSH = 69
    KELTHUZAD = 70
    ANA = 71
    
class Maps(Enum):
    BATTLEFIELD = 0 
    BLACKHEART = 1
    CURSED = 2
    DRAGON = 3
    GARDEN = 4
    HAUNTED = 5
    INFERNAL = 6
    SKY = 7
    TOMB = 8
    TOWERS = 9
    BRAXIS = 10
    WARHEAD = 11
    HANAMURA = 12
    VOLSKAYA = 13
    
def create_game( hero1, hero2, hero3, hero4, hero5, hero6, hero7, hero8, hero9, hero10, MAP):
    match = []
    matchup = []
    myMap = []
    team1 = [0]*100
    team1[hero1] = 1
    team1[hero2] = 1
    team1[hero3] = 1
    team1[hero4] = 1
    team1[hero5] = 1
    team2 = [0]*100
    team2[hero6] = 1
    team2[hero7] = 1
    team2[hero8] = 1
    team2[hero9] = 1
    team2[hero10] = 1
    matchup.append(team1)
    matchup.append(team2)
    match.append(matchup)
    
    mapHack = []
    maps = [0]*13
    maps[MAP] = 1
    myMap.append(maps)
    myMap.append(maps)
    mapHack.append(myMap)
    
    return match, mapHack

# Network Parameters
n_input = 100 * 2 # MNIST data input (img shape: 128*128)
n_classes = 2 # dog or cat
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 2, 100])
m = tf.placeholder(tf.float32, [None, 2, 13])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
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

# Initializing the variables
init = tf.global_variables_initializer()

print('Parameters initialized')

myGame, myMap = create_game( Heroes.DEHAKA.value, 
                             Heroes.FALSTAD.value, 
                             Heroes.TYRAEL.value, 
                             Heroes.LIMING.value, 
                             Heroes.REHGAR.value,
                             Heroes.ZERATUL.value, 
                             Heroes.GARROSH.value, 
                             Heroes.BRIGHTWING.value, 
                             Heroes.VALLA.value, 
                             Heroes.UTHER.value,
                             Maps.SKY.value)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print('Session started')

    saver = tf.train.Saver()
    saver.restore(sess,'C:\\Users\\Daniel\\.spyder-py3\\siamese_abs')
    
    w = sess.run(tf.abs(weights['heroes_w']))
    #np.savetxt("heroes.csv", np.array(w), delimiter=";")
    
    class DraftState:

        def __init__(self, teamA, teamB, hero_Pool, theMap, thePhase, playerjp):
            self.phase = thePhase # At the root pretend the player just moved is p2 - p1 has the first move
            self.team1 = teamA
            self.team2 = teamB
            self.heroes = hero_Pool
            self.actions = self.GetMoves()
            self.map = theMap
            self.playerJustMoved = playerjp
    
        def perform(self, move):
            """ Update a state by carrying out the given move.
                Must update playerJustMoved.
            """
            action = deepcopy(self)
            
            if action.phase == 1:
                action.heroes[move] = 0
                action.playerJustMoved = 1
            if action.phase == 2:
                action.heroes[move] = 0
                action.playerJustMoved = 2
            if action.phase == 3:
                action.team1[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 1
            if action.phase == 4:
                action.team2[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 2
            if action.phase == 5:
                action.team2[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 2
            if action.phase == 6:
                action.team1[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 1
            if action.phase == 7:
                action.team1[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 1
            if action.phase == 8:
                action.heroes[move] = 0
                action.playerJustMoved = 2
            if action.phase == 9:
                action.heroes[move] = 0
                action.playerJustMoved = 1
            if action.phase == 10:
                action.team2[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 2
            if action.phase == 11:
                action.team2[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 2
            if action.phase == 12:
                action.team1[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 1
            if action.phase == 13:
                action.team1[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 1
            if action.phase == 14:
                action.team2[move] = 1
                action.heroes[move] = 0
                action.playerJustMoved = 2
                
            action.phase = action.phase + 1
            
            return action
        
        def GetMoves(self):
            """ Get all possible moves from this state.
            """
            heroes_left = []
            if self.phase == 15:
                return heroes_left
            else:
                i = 0
                for hero in self.heroes:
                    if hero == 1:
                        heroes_left.append(i)
                    i += 1
                return heroes_left
#            
#        def DoRandomRollout(self):
#            """ Get to end state
#            """
#            heroes_left = self.GetMoves()
#            picks_left1 = 5 - np.sum(self.team1)
#            picks_left2 = 5 - np.sum(self.team2)
#            picks_left = picks_left1 + picks_left2
#            picks = random.sample( heroes_left, picks_left )
#            a = 0
#            for pick in picks:
#                if a < picks_left1:
#                    self.team1[picks[a]] = 1
#                else:
#                    self.team2[picks[a]] = 1
#                    
#                a += 1
#            
#            self.phase = 15
#          
        def is_terminal(self):
            if self.phase == 15:
                return True
            else:
                return False
        
        def reward(self, parent, action):
            """ Get the game result from the viewpoint of playerjm. 
            """
            assert self.phase == 15
            game = []
            games = []
            game.append(self.team1)
            game.append(self.team2)
            games.append(game)
            maps = []
            mapss = []
            maps.append(self.map)
            maps.append(self.map)
            mapss.append(maps)
            result = sess.run(tf.nn.softmax(pred), feed_dict={x: games,
                                         m: mapss,
                                         keep_prob: 1.})
            #if playerjm == 1:
            return result[0,1]
            #else:
            #    return result[0,1]
    
    team1 = [0]*100
    team1[Heroes.STUKOV.value] = 1
    team1[Heroes.MEDIVH.value] = 1
    team1[Heroes.DIABLO.value] = 1
    team1[Heroes.CASSIA.value] = 1
    team1[Heroes.STITCHES.value] = 1
    team2 = [0]*100
    team2[Heroes.BUTCHER.value] = 1
    team2[Heroes.XUL.value] = 1
    team2[Heroes.KERRIGAN.value] = 1
    team2[Heroes.GARROSH.value] = 1
    team2[Heroes.CHROMIE.value] = 1
    pool = [1]*72
    pool[43] = 0
    pool[44] = 0
    pool[Heroes.MALTHAEL.value] = 0
    pool[Heroes.ANUBARAK.value] = 0
    pool[Heroes.STUKOV.value] = 0
    pool[Heroes.BUTCHER.value] = 0
    pool[Heroes.ALARAK.value] = 0
    pool[Heroes.UTHER.value] = 0
    pool[Heroes.KERRIGAN.value] = 0
    pool[Heroes.GARROSH.value] = 0
    pool[Heroes.CASSIA.value] = 0
    pool[Heroes.STITCHES.value] = 0
#    pool[Heroes.TRACER.value] = 0
#    pool[Heroes.DIABLO.value] = 0
#    pool[Heroes.TYRANDE.value] = 0
    mymap = [0]*13
    mymap[3] = 1
    boardState = DraftState(team1, team2, pool, mymap, 14, 1)
    
    algo = MCTS(tree_policy=UCB1(c=1.41), 
            default_policy=random_terminal_roll_out,
            backup=monte_carlo)
    
    #print( algo(StateNode(None, boardState), n = 500) )

#    w = w[:-28]
#    linkage = scipy.cluster.hierarchy.linkage(w, method='ward')
#    hero_label = []
#    for a in range(0,72):
#        hero_label.append(Heroes(a).name)
#
#    plt.title('RMSD Average linkage hierarchical clustering')
#    _ = scipy.cluster.hierarchy.dendrogram(linkage, labels = hero_label, count_sort='descendent', leaf_font_size = 12)

#        
    #print( weights['heroes_w'] ) 
    
    print("Predicted:", \
    sess.run(tf.nn.softmax(pred), feed_dict={x: myGame,
                                 m: myMap,
                                 keep_prob: 1.}))
