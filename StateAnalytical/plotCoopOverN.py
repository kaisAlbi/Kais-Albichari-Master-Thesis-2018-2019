import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import csv

from matplotlib.ticker import PercentFormatter


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


def getStationaryFromFile(filename):
    f = open(filename, "r")
    stationary = []
    for line in f.readlines():
        index = line.index(": ")
        stationary_i = line[index + 1:]
        stationary.append(float(stationary_i))

    r = 1 - sum(stationary)  # Handle loss of precision
    #print(r)
    #print(sum(stationary))
    stationary[0] += r
    #print(sum(stationary))
    f.close()
    return stationary

def getCoopRatioOneState(game, strategies):
    coop_ratio = 0
    stationary_filename = "stationaryDistrib/" + game + "/Groups/Reduced/1state/stationary_1st_anygroupsize.txt"
    stationary = getStationaryFromFile(stationary_filename)
    for i in range (len(stationary)):
        if strategies[i][2] == 1:
            coop_ratio += stationary[i]
    return coop_ratio



def getCoopRatioTwoStates(game, strategies, N_values):
    coop_ratio = np.zeros(len(N_values))
    for i in range (len(N_values)):
        group_size = N_values[i]
        stationary_filename = "stationaryDistrib/" + game + "/Groups/Reduced/2states/stationary_2st_" + str(group_size) + "groupsize.txt"
        stationary = getStationaryFromFile(stationary_filename)
        print(stationary)
        for j in range (len(stationary)):
            cur_strat = strategies[j]
            if hasOnlyOneAction(cur_strat):
                if cur_strat[2] == 1:
                    coop_ratio[i] += stationary[j]
            elif hasOnlyOneDirection(cur_strat):
                if cur_strat[0] == 1 and cur_strat[2] == 1:
                    coop_ratio[i] += stationary[j]
                elif cur_strat[0] == 0 and cur_strat[-1] == 1:
                    coop_ratio[i] += stationary[j]
            else:
                coop_ratio[i] += (stationary[j]/2)
    return coop_ratio


def plotCoopRatioN(game, N_values):
    strat_1state = reducedStrategies(createStrategies(1))
    strat_2states = reducedStrategies(createStrategies(2))
    coop_one_state = getCoopRatioOneState(game, strat_1state)
    coop_two_states = getCoopRatioTwoStates(game, strat_2states, N_values)

    print("c one state :",coop_one_state)
    print("c two states :", coop_two_states)
    x = N_values
    markers = [".", "s", "+", "D", "^", "*", "p"]

    fig, ax = plt.subplots()
    plt.title(game)
    ax.set_ylabel("Cooperation")
    x_label = "group size"
    ax.set_xlabel(x_label)
    plt.ylim(-0.1, 1.1)
    ax.plot(x, coop_two_states, marker= "x", label="2states")
    ax.plot(x, [coop_one_state]*len(x), marker=".", label="1 state")

    legend = ax.legend(loc='best', bbox_to_anchor=(0.85, 0.7))
    plt.show()


def getCoopOverNFromFiles(game, nb_states):
    """
    Read series of files to get the cooperation ratio in the right files depending on the game and the number of
    internal states considered
    :param game: dilemma considered
    :param nb_states: number of internal states
    :return: matrix containing the cooperation ratio for each number of internal state, and each different T value
    considered
    """
    folder = "EvalCoop/"+game+"/Different_N/"
    n_subfolders = ["N1/", "N2/", "N6/", "N10/", "N20/"]

    coop_for_states_diff_n = np.zeros((nb_states, len(n_subfolders)))
    for i in range (len(n_subfolders)):
        cur_filename = folder + n_subfolders[i] + "coop_evolution.csv"

        with open(cur_filename, "r") as f:
            reader = csv.DictReader(f)  # read rows into a dictionary format
            for row in reader:
                state = int(row["NbStates"])
                coop_ratio = float(row["Cooperation"])
                coop_for_states_diff_n[state-1, i] = coop_ratio
    return coop_for_states_diff_n



def showCoopOverN(game, nb_states, N_values):
    """
    Show the evolution of cooperation in function of the number of rounds N and number of internal states values, given a
    specified dilemma
    :param game: dilemma considered
    :param nb_states: number of internal states
    """
    coop_for_states_diff_n = getCoopOverNFromFiles(game, nb_states)

    x = N_values
    markers = [".", "s", "+", "D", "^"]

    fig, ax = plt.subplots()
    plt.title(game)
    ax.set_ylabel("Cooperation")
    x_label = "Group size"
    ax.set_xlabel(x_label)
    plt.ylim(-0.1, 1.1)

    ax.plot(x, coop_for_states_diff_n[0], marker = markers[0], label = "1 state")
    for i in range (1, nb_states):
        ax.plot(x, coop_for_states_diff_n[i], marker = markers[i], label = str(i+1)+" states")

    legend = ax.legend(loc='upper right')

    plt.show()


N_values = [1,2,6,10,20]
game = "PD"
nb_states = 5
showCoopOverN(game, nb_states, N_values)