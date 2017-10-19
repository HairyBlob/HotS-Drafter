# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:29:24 2017

@author: Daniel
"""
from enum import Enum

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
    JUNKRAT = 72
    
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
    team1[Heroes(hero1).value] = 1
    team1[Heroes(hero2).value] = 1
    team1[Heroes(hero3).value] = 1
    team1[Heroes(hero4).value] = 1
    team1[Heroes(hero5).value] = 1
    team2 = [0]*100
    team2[Heroes(hero6).value] = 1
    team2[Heroes(hero7).value] = 1
    team2[Heroes(hero8).value] = 1
    team2[Heroes(hero9).value] = 1
    team2[Heroes(hero10).value] = 1
    matchup.append(team1)
    matchup.append(team2)
    match.append(matchup)
    
    mapHack = []
    maps = [0]*20
    maps[MAP] = 1
    myMap.append(maps)
    myMap.append(maps)
    mapHack.append(myMap)
    
    return match, mapHack
