# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:25:43 2017

@author: Daniel
"""
import numpy as np
import csv

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
    
    with open( 'ReplayCharacters.csv' ) as csvfile:
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
            if idx % 10 == 0 and idx > 500000 and mmr > 18000:
                games_id.append( row[0] )
                
                #games must have 5 winners and 5 losers
                assert sum(game_winner) == 5, ('not 5 winners')
                game_winner_arr = np.array( game_winner )
                assert sum(game_loser) == 5, ('not 5 losers')
                game_loser_arr = np.array( game_loser )
                
#                For each game we double the training set by mirroring the matchup and results
                game_matchupA = []
                game_matchupA.append(game_winner_arr)
                game_matchupA.append(game_loser_arr)
                games_result.append([1,0])
                
                game_matchupB = []
                game_matchupB.append(game_loser_arr)
                game_matchupB.append(game_winner_arr)
                games_result.append([0,1])
                
                #We also force symmetry in the network by training on identical team compositions (50% winrate)
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
            elif idx % 10 == 0 and idx <= 500000 and mmr > 18000:
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
#            if idx >1000000:
#                break
    
    #Load the maps associated with the the matchups
    with open( 'Replays.csv' ) as mapsfile:
        mapsreader = csv.reader(mapsfile, delimiter=',')
        next(mapsreader)
        idx = 0
        print( len(games_id) )
        for row in mapsreader:
            if( row[0] == games_id[idx]):
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
            