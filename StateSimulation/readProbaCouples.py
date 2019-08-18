import json
import numpy as np
from itertools import product

def loadProbaCouples(filename):
    f = open(filename, "r")
    proba_c_matrix = json.load(f)
    f.close()
    return np.asarray(proba_c_matrix)

def createStrategies():
    """
    Generate all possible combination of strategies, depending on the number of states
    :return: list of all strategies in the form [T_c,T_d, action state_0, ..., action state_maxState]
            transition = 1 = Left ; action = 1 = C
    """
    #strats = list(list(int(elem) for elem in items) for items in product("10", repeat=self.getNbStates() + 2))

    action_choice = list(list(item) for item in product("CD", repeat=2))
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
            #strats.append([state_c, action])
    return strats

strats = ["LLCC", "LRCC", "RLCC", "RRCC", "LLCD", "LRCD", "RLCD", "RRCD", "LLDC","LRDC", "RLDC", "RRDC", "LLDD", "LRDD", "RLDD", "RRDD"]

print(strats)
filename = "proba_couples_matrix.txt"

proba_c_matrix = loadProbaCouples(filename)

print("Probabilities of cooperation for pair of strategies:")
for i in range(len(proba_c_matrix)):
    print("---------------------")

    for j in range(i, len(proba_c_matrix)):
        print("Strat : ", strats[i], " (",proba_c_matrix[i, j], ") against strat : ", strats[j], " (",proba_c_matrix[j, i], ")")

print(proba_c_matrix)