# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:29:51 2017

@author: Daniel
"""
from copy import deepcopy
import numpy as np
import random 

class DraftState:

        def __init__(self, teamA, teamB, hero_Pool, theMap, thePhase, playerjp, evalfunc):
            self.phase = thePhase 
            self.team1 = teamA
            self.team2 = teamB
            self.heroes = hero_Pool
            self.actions = self.GetMoves()
            self.map = theMap
            self.player = playerjp
            self.evaluation = evalfunc
    
        def perform(self, move):
            """ Update a state by carrying out the given move.
                Must update playerJustMoved.
            """
            action = deepcopy(self)
            
            if action.phase == 1:
                action.heroes[move] = 0
            if action.phase == 2:
                action.heroes[move] = 0
            if action.phase == 3:
                action.team1[move] = 1
                action.heroes[move] = 0
            if action.phase == 4:
                action.team2[move] = 1
                action.heroes[move] = 0
            if action.phase == 5:
                action.team2[move] = 1
                action.heroes[move] = 0
            if action.phase == 6:
                action.team1[move] = 1
                action.heroes[move] = 0
            if action.phase == 7:
                action.team1[move] = 1
                action.heroes[move] = 0
            if action.phase == 8:
                action.heroes[move] = 0
            if action.phase == 9:
                action.heroes[move] = 0
            if action.phase == 10:
                action.team2[move] = 1
                action.heroes[move] = 0
            if action.phase == 11:
                action.team2[move] = 1
                action.heroes[move] = 0
            if action.phase == 12:
                action.team1[move] = 1
                action.heroes[move] = 0
            if action.phase == 13:
                action.team1[move] = 1
                action.heroes[move] = 0
            if action.phase == 14:
                action.team2[move] = 1
                action.heroes[move] = 0
                
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
            
        def DoRandomRollout(self):
            """ Get to end state
            """
            heroes_left = self.GetMoves()
            picks_left1 = 5 - np.sum(self.team1)
            picks_left2 = 5 - np.sum(self.team2)
            picks_left = picks_left1 + picks_left2
            picks = random.sample( heroes_left, picks_left )
            a = 0
            for pick in picks:
                if a < picks_left1:
                    self.team1[picks[a]] = 1
                else:
                    self.team2[picks[a]] = 1
                    
                a += 1
            
            self.phase = 15
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
            result = self.evaluation( games, mapss )
            if self.player == 1:
                return result[0,0]
            elif self.player == 2:
                return result[0,1]
            

def random_search(boardState, itNb):
    heroes = [0]*100
    visits = [0]*100
    for i in range(itNb):
        print(i)
        board = deepcopy(boardState)
        board.DoRandomRollout()
        reward = board.reward()
        for idx, Hero in enumerate(board.team1):
            if Hero == 1 and boardState.team1[idx] != 1:
                visits[idx] += 1
                if boardState.player == 1:
                    heroes[idx] += reward
                elif boardState.player == 2:
                    heroes[idx] += 1 - reward
        for idx, Hero in enumerate(board.team2):
            if Hero == 1 and boardState.team2[idx] != 1:
                visits[idx] += 1
                if boardState.player == 2:
                    heroes[idx] += reward
                elif boardState.player == 1:
                    heroes[idx] += 1 - reward
    for idx, Hero in enumerate(heroes):
        if visits[idx] != 0:
            Hero /= visits[idx]
    return heroes.index(max(heroes))