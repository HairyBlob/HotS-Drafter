from __future__ import print_function
import tensorflow as tf
import numpy as np
from HotSInfo import Heroes, Maps, create_game
from mcts.mcts import MCTS
from mcts.default_policies import random_terminal_roll_out
from mcts.backups import monte_carlo
from mcts.tree_policies import UCB1
from mcts.graph import StateNode
import draftBoard
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy

class ImportDiscGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:            
            self.activation = self.graph.get_operation_by_name('y').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data, "Placeholder_1:0": 1.0})
    
class ImportWRGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:            
            self.activation = self.graph.get_operation_by_name('y').outputs[0]
    
    def run(self, data, maps):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data, "m:0": maps, "Placeholder_1:0": 1.0})

init = tf.global_variables_initializer()

print('Parameters initialized')

#IF YOU WANT TO ESTIMATE THE WINRATE OF A SPECIFIC GAME
myGame, myMap = create_game( Heroes.MALFURION.value, 
                             Heroes.MURADIN.value, 
                             Heroes.GENJI.value, 
                             Heroes.CHROMIE.value,   
                             Heroes.ZERATUL.value,
                             Heroes.FALSTAD.value, 
                             Heroes.DEHAKA.value, 
                             Heroes.GARROSH.value, 
                             Heroes.LIMING.value,   
                             Heroes.DECKARD.value,
                             Maps.DRAGON.value)


# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    print('Session started')

    WR_model = ImportWRGraph('C:\\Users\\Daniel\\.spyder-py3\\siamese_L2_deep')
    
    tf.get_default_graph().finalize()
    model = ImportDiscGraph('C:\\Users\\Daniel\\.spyder-py3\\discriminator')
    #Evaluation function to pass to the MCTS. Has to be defined outside of DraftBoard to prevent graph modification that can slow execution
    def evalFunc(games, mapss):
        result = WR_model.run(games, mapss)
        discrimination = model.run(np.array(games).reshape(-1, 100))
#        print(result)
#        print(discrimination)
        result[0,0] += discrimination[0,0]/4
        result[0,1] += discrimination[1,0]/4
        return result
    
    #add picked heroes
    team1 = [0]*100
    team1[Heroes.MALFURION.value] = 1
    team1[Heroes.GARROSH.value] = 1
    team1[Heroes.DEHAKA.value] = 1
    team1[Heroes.GENJI.value] = 1
    team1[Heroes.GREYMANE.value] = 1
    team2 = [0]*100
    team2[Heroes.DECKARD.value] = 1
    team2[Heroes.RAGNAROS.value] = 1
    team2[Heroes.FENIX.value] = 1
    team2[Heroes.THRALL.value] = 1
#    team2[Heroes.UTHER.value] = 1
    pool = [1]*79
    #Need to ban CHOGALL because he doesn't behave the same way in draft as other heroes.
    #Still needs to be implemented
    pool[Heroes.CHO.value] = 0
    pool[Heroes.GALL.value] = 0
    #Add picked or banned heroes here
    pool[Heroes.MALFURION.value] = 0
    pool[Heroes.DECKARD.value] = 0
    pool[Heroes.GARROSH.value] = 0
    pool[Heroes.DEHAKA.value] = 0
    pool[Heroes.GENJI.value] = 0
    pool[Heroes.GREYMANE.value] = 0
    pool[Heroes.FENIX.value] = 0
#    pool[Heroes.MURKY.value] = 0
    pool[Heroes.THRALL.value] = 0
#    pool[Heroes.ABATHUR.value] = 0
#    pool[Heroes.VARIAN.value] = 0
#    pool[Heroes.ZAGARA.value] = 0
#    pool[Heroes.SAMURO.value] = 0
    mymap = [0]*20
    mymap[Maps.INFERNAL.value] = 1
    #Inputs are: team1, team2, unpickable heroes, map, pick phase (1-15), team to pick
#    boardState = draftBoard.DraftState(team1, team2, pool, mymap, 11, 2, evalFunc)
#    algo = MCTS(tree_policy=UCB1(c=1.41), 
#        default_policy=random_terminal_roll_out,
#        backup=monte_carlo)
###    
##    #TO GET THE 4 BEST PICKS ACCORDING TO THE MCTS
#    result = algo(StateNode(None,boardState), n = 1000)
#    for hero in result:
#        print(Heroes(hero.action).name)
#    print( Heroes(draftBoard.random_search(boardState, 40000)).name )
    
#    TO MAP THE HEROES CLUSTERING
#    w = w[:-21]
#    linkage = scipy.cluster.hierarchy.linkage(w, method='ward')
#    hero_label = []
#    for a in range(0,79):
#        hero_label.append(Heroes(a).name)
#
#    plt.title('RMSD Average linkage hierarchical clustering')
#    _ = scipy.cluster.hierarchy.dendrogram(linkage, labels = hero_label, count_sort='descendent', leaf_font_size = 12)

    #TO PREDICT TO RESULT OF A GAME
    
    print("Predicted:", \
    WR_model.run(myGame, myMap))
#
#    print("Predicted:", \
#    model.run(np.array(myGame).reshape(-1, 100)))