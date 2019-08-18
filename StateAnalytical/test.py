import json
from itertools import product
import numpy as np
from scipy.stats import t
from itertools import permutations
from sympy import Symbol, simplify, core
from sympy.utilities.iterables import multiset_permutations
import pickle
import csv
import glob

def createStrategies(nb_states):
    """
    Generate all possible combination of strategies, depending on the number of states
    :return: list of all strategies in the form [T_c,T_d, action state_0, ..., action state_maxState]
            transition = 1 = Left ; action = 1 = C
    """

    action_choice = list(list(item) for item in product("CD", repeat=nb_states))
    state_change = list(list(item) for item in product("LR", repeat=2))
    strats = []
    for action in action_choice:
        action_c_tr = []
        for i in range (len(action)):

            if action[i] == "C":
                action_c_tr.append(1)
            else:
                action_c_tr.append(0)

        for state_c in state_change:
            state_c_tr = []
            if state_c[0] == "L":
                state_c_tr.append(1)
            else:
                state_c_tr.append(0)
            if state_c[1] == "L":
                state_c_tr.append(1)
            else:
                state_c_tr.append(0)
            list_c_tr = [state_c_tr, action_c_tr]
            strat = [item for sublist in list_c_tr for item in sublist]
            strats.append(strat)
    return strats

def reducedStrategies(strategies):

    all_c = strategies[0]
    all_d = all_c[:2] + strategies[-1][2:]
    reduced_strats = [all_c]
    for strat in strategies:
        if not (hasOnlyOneAction(strat) or hasOnlyOneDirection(strat)):
            reversed_strat = []
            for i in range (2):
                reversed_strat.append((strat[i]+1)%2)   #Bit flip for transitions : LR -> RL
            for i in range (len(strat)-1, 1, -1):       #Reverse actions : CDD -> DDC
                reversed_strat.append(strat[i])
            if reversed_strat not in reduced_strats:
                reduced_strats.append(strat)
    reduced_strats.append(all_d)
    return reduced_strats

def hasOnlyOneDirection(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible transition direction, False otherwise
    """
    directions = strat[:2]
    if 1 not in directions or 0 not in directions:
        return True
    return False

def hasOnlyOneAction(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible action choice, False otherwise
    """
    actions = strat[2:]
    if 1 not in actions or 0 not in actions:
        return True
    return False

def getFreqFromFile(filename, strats):
    n = len(strats)
    fix_mat = np.zeros((n, n))

    f = open(filename, "r")
    for line in f.readlines():
        res_strat, fix_probs = json.loads(line)
        index_strat = strats.index(res_strat)
        fix_mat[index_strat] = fix_probs
    return fix_mat


def getStationaryFromFile(filename):
    f = open(filename, "r")
    stationary = []
    for line in f.readlines():
        index = line.index(": ")
        stationary_i = line[index+1:]
        if "-" in stationary_i:
            print(stationary_i)
        stationary.append(float(stationary_i))

    r = 1 - sum(stationary)  # Handle loss of precision
    print(r)
    print(sum(stationary))
    stationary[0] += r
    print(sum(stationary))
    f.close()
    return stationary


def buildExpr():

    print("start")
    x = Symbol('x')
    y = Symbol('y')
    alpha = Symbol("α")
    weight = alpha*2
    print(x*weight)
    expression = 0
    expression += x
    expression += (1-alpha)*y
    expression += alpha*x
    print(expression)
    print(simplify(expression))
    return expression

def displayableStrat(strat):
    """
    Transforms a binary strategy array into a human readable strategy array
    :param strat: strategy array
    :return: human readable strategy array
    """
    displayable_strat = []
    for i in range (len(strat)):
        if strat[i] == 1:
            if i < 2:
                displayable_strat.append("L")
            else:
                displayable_strat.append("C")
        else:
            if i < 2:
                displayable_strat.append("R")
            else:
                displayable_strat.append("D")
    return displayable_strat

n = 5
print(np.identity(n, dtype = float))