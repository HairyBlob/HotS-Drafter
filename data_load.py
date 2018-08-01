# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:25:43 2017

@author: Daniel
"""
import numpy as np
import csv
import random
import tensorflow as tf

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
        return self.sess.run(self.activation, feed_dict={"x:0": data, "keep_prob:0": 1.0})

#Load the HotsLogs data.
def get_data_winrate_estimator( dataAugment = False, filterByMMR = False, averageMMR = 2500):
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
    test_id = []
    train_id = []
    games_result = []
    maps = []
    
    test_games_matchup = []
    test_games_result = []
    test_maps = []
    
    game_winner = [0]*100
    game_loser = [0]*100     
    
    with open( 'C:/Users/gosek/Downloads/HOTSLogs_Data_Export_Current_Yrel/ReplayCharacters.csv' ) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        idx = 1
        mmr = 0
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
#            model = ImportDiscGraph('C:\\Users\\gosek\\.spyder-py3\\discriminator')
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
                
                #Testing set
                if idx % 10 == 0 and idx % 100 == 0 and (not filterByMMR or mmr > averageMMR*10):
                    
                    assert sum(game_winner) == 5, ('not 5 winners')
                    game_winner_arr = np.array( game_winner )
                    assert sum(game_loser) == 5, ('not 5 losers')
                    game_loser_arr = np.array( game_loser )
                    
#                    if (model.run(np.array(game_winner).reshape(-1,100))[0,0] > 0.5 and model.run(np.array(game_loser).reshape(-1,100))[0,0] > 0.5):
                    test_id.append( idx/10 -1 )
                    
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
                    
                    #Training set
                elif idx % 10 == 0 and (not filterByMMR or mmr > averageMMR*10):
                    train_id.append( idx/10 -1 )
                    
                    #games must have 5 winners and 5 losers
                    assert sum(game_winner) == 5, ('not 5 winners')
                    game_winner_arr = np.array( game_winner )
                    assert sum(game_loser) == 5, ('not 5 losers')
                    game_loser_arr = np.array( game_loser )
                    
    #                For each game we double the training set by mirroring the matchup and results
#                    if (model.run(np.array(game_winner).reshape(-1,100))[0,0] < 0.5 and model.run(np.array(game_loser).reshape(-1,100))[0,0] > 0.5):
#                        game_matchupA = []
#                        game_matchupA.append(game_loser_arr)
#                        game_matchupA.append(game_winner_arr)
#                        games_result.append([1,0])
#                        
#                        game_matchupB = []
#                        game_matchupB.append(game_winner_arr)
#                        game_matchupB.append(game_loser_arr)
#                        games_result.append([0,1])
#                            
#                        games_matchup.append( game_matchupA )
#                        games_matchup.append( game_matchupB )
                    
#                    else:
                    game_matchupA = []
                    game_matchupA.append(game_winner_arr)
                    game_matchupA.append(game_loser_arr)
                    games_result.append([1,0])
                    
                    game_matchupB = []
                    game_matchupB.append(game_loser_arr)
                    game_matchupB.append(game_winner_arr)
                    games_result.append([0,1])
                        
                    games_matchup.append( game_matchupA )
                    games_matchup.append( game_matchupB )
                            
                    #We also force symmetry in the network by training on identical team compositions (50% winrate)
                    if dataAugment:
                        game_matchupC = []
                        game_matchupC.append(game_loser_arr)
                        game_matchupC.append(game_loser_arr)
                        games_result.append([0.5,0.5])
                        
                        game_matchupD = []
                        game_matchupD.append(game_winner_arr)
                        game_matchupD.append(game_winner_arr)
                        games_result.append([0.5,0.5])
                        games_matchup.append( game_matchupC )
                        games_matchup.append( game_matchupD )
    
                    game_winner = [0]*100
                    game_loser = [0]*100
                    mmr = 0
                
                
                #if we screen by mmr, reset the game arrays
                elif idx % 10 == 0:
                    game_winner = [0]*100
                    game_loser = [0]*100
                    mmr = 0  
                    
                idx += 1
    #            if idx >2000000:
    #                break
    
    #Load the maps associated with the the matchups
    with open( 'C:/Users/gosek/Downloads/HOTSLogs_Data_Export_Current_Yrel/Replays.csv' ) as mapsfile:
        mapsreader = csv.reader(mapsfile, delimiter=',')
        next(mapsreader)
        train_idx = 0
        test_idx = 0
        idx = 0
        print( len(test_id) )
        for row in mapsreader:
            if( idx == test_id[test_idx] or idx == train_id[train_idx] ):
                actual_map = [0]*20
                actual_map1 = [0]*20
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
                    
                elif int( row[2] ) == 1019:
                    actual_map[13] = 1
                    actual_map1[13] = 1
                elif int( row[2] ) == 1022:
                    actual_map[14] = 1
                    actual_map1[14] = 1
                #Testing set
                #Maps are added twice to match the number of teams ( 2 team per game )
                if idx == test_id[test_idx]:
                    map_copy = []
                    map_copy.append(actual_map)
                    map_copy.append(actual_map1)
                    test_maps.append( map_copy )
                    test_maps.append( map_copy )
                    if test_idx != len(test_id) - 1:
                        test_idx += 1
                #Training Set
                if idx == train_id[train_idx]:
                    map_copy = []
                    map_copy.append(actual_map)
                    map_copy.append(actual_map1)
                    maps.append( map_copy )
                    maps.append( map_copy )
                    if dataAugment:
                        maps.append( map_copy )
                        maps.append( map_copy )
                    if train_idx != len(train_id) - 1:
                        train_idx += 1
                if len(games_result) == len(maps):
                    break
            idx += 1
                
#    np.save('games_matchup.npy',  np.array(games_matchup))
#    np.save('games_result.npy',  np.array(games_result))
#    np.save('maps.npy',  np.array(maps))
#    np.save('test_games_matchup.npy',  np.array(test_games_matchup))
#    np.save('test_games_result.npy',  np.array(test_games_result))
#    np.save('test_maps.npy',  np.array(test_maps))
    return np.array(games_matchup), np.array(games_result), np.array(maps), np.array(test_games_matchup), np.array(test_games_result), np.array(test_maps)
            

def generate_random_team():
    team = [0]*100 
    heroes = random.sample(range(0,78), 5)
    others = random.sample(range(0,76), 3)
    if 43 not in heroes or 44 not in heroes:
        for x in range(0,5):
            team[heroes[x]] = 1
    else:
        team[43] = 1
        team[44] = 1
        for x in range(0,3):
            if others[x] < 43:
                team[others[x]] = 1
            else:
                team[others[x]+2] = 1
    assert sum(team) == 5
    return np.array(team)

#Load the HotsLogs data.
def get_data_discriminator( filterByMMR = False, averageMMR = 2500):
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
    games_result = []
    
    test_games_matchup = []
    test_games_result = []
    
    game_winner = [0]*100
    game_loser = [0]*100    
    
    gameID = 0
        
    #Load the maps associated with the the matchups
    with open( 'C:/Users/gosek/Downloads/HOTSLogs_Data_Export_Current_Yrel/Replays.csv' ) as mapsfile:
        mapsreader = csv.reader(mapsfile, delimiter=',')
        next(mapsreader)
        for row in mapsreader:
            if int(row[1]) == 4:
                gameID = int(row[0])
                print(gameID)
                break
    
    with open( 'C:/Users/gosek/Downloads/HOTSLogs_Data_Export_Current_Yrel/ReplayCharacters.csv' ) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        idx = 1
        mmr = 0
        reached = False
        for row in reader:
            if not reached and int(row[0]) != gameID:
                continue
            elif int(row[0]) == gameID:
                reached = True
                print("reached!")
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
            
            #Testing set
            if idx % 10 == 0 and idx % 100 == 0 and (not filterByMMR or mmr > averageMMR*10):
                
                assert sum(game_winner) == 5, ('not 5 winners')
                game_winner_arr = np.array( game_winner )
                assert sum(game_loser) == 5, ('not 5 losers')
                game_loser_arr = np.array( game_loser )
                
                false_game_1 = generate_random_team()
                false_game_2 = generate_random_team()

                test_games_result.append([1,0])
                test_games_result.append([1,0])
                test_games_result.append([0,1])
                test_games_result.append([0,1])
                    
                test_games_matchup.append( game_winner_arr )
                test_games_matchup.append( game_loser_arr )
                test_games_matchup.append( false_game_1 )
                test_games_matchup.append( false_game_2 )

                game_winner = [0]*100
                game_loser = [0]*100
                mmr = 0
                
                #Training set
            elif idx % 10 == 0 and (not filterByMMR or mmr > averageMMR*10):                
                #games must have 5 winners and 5 losers
                assert sum(game_winner) == 5, ('not 5 winners')
                game_winner_arr = np.array( game_winner )
                assert sum(game_loser) == 5, ('not 5 losers')
                game_loser_arr = np.array( game_loser )
                
                false_game_1 = generate_random_team()
                false_game_2 = generate_random_team()
                
                games_result.append([1,0])
                games_result.append([1,0])
                games_result.append([0,1])
                games_result.append([0,1])
                    
                games_matchup.append( game_winner )
                games_matchup.append( game_loser )
                games_matchup.append( false_game_1 )
                games_matchup.append( false_game_2 )
                
                game_winner = [0]*100
                game_loser = [0]*100
                mmr = 0
            
            
            #if we screen by mmr, reset the game arrays
            elif idx % 10 == 0:
                game_winner = [0]*100
                game_loser = [0]*100
                mmr = 0  
                
            idx += 1
#            if idx >2000000:
#                break

    return np.array(games_matchup), np.array(games_result), np.array(test_games_matchup), np.array(test_games_result)
            
