import numpy as np
import json
from itertools import product
import matplotlib.pyplot as plt

def createStrategies(nb_states):
    """
    Generate all possible combination of strategies, depending on the number of signals
    :return: list of all strategies
    """
    action_choice = list(list(item) for item in product("CD", repeat=nb_states))
    state_change = list(list(item) for item in product("LR", repeat=2))
    strats = []
    for action in action_choice:
        for state_c in state_change:
            strats.append([state_c, action])
    return strats


def getStationaryFromFile(filename):
    f = open(filename, "r")
    stationary = []
    for line in f.readlines():
        index = line.index(": ")
        stationary_i = line[index + 1:]
        stationary.append(float(stationary_i))

    r = 1 - sum(stationary)  # Handle loss of precision
    stationary[0] += r

    f.close()
    return stationary


def getFreqFromFile(filename, strats):
    n = len(strats)
    fix_mat = np.zeros((n, n))

    f = open(filename, "r")
    for line in f.readlines():
        res_strat, fix_probs = json.loads(line)
        index_strat = strats.index(res_strat)
        fix_mat[index_strat] = fix_probs
    return fix_mat


def stationaryDistrib(Z, strats, freq_mat):
    """
    Calculates the transition matrix and the stationary distributions of the strategies.
    :return: transition matrix and stationary distributions
    """
    n = len(strats)
    norm_fact = 1 / float(n - 1)
    transitions = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        transitions[i, i] = 1
        for j in range(n):
            if i != j:
                trans_val = freq_mat[i, j] * norm_fact
                transitions[i, j] = trans_val
                transitions[i, i] -= trans_val
    val, vect = np.linalg.eig(transitions.transpose())
    j_stationary = np.argmin(abs(val - 1.0))  # look for the element closest to 1 in the list of eigenvalues
    p_stationary = abs(vect[:, j_stationary].real)  # the, is essential to access the matrix by column
    p_stationary /= p_stationary.sum()  # normalize

    return transitions, p_stationary


def showStationaryDistrib(game, strats, stationary):
    n = len(strats)
    x = [i for i in range(n)]

    fig = plt.figure()
    plt.title(game)
    ax = fig.add_subplot(111)
    ax.set_ylabel("stationary distribution")

    plt.xticks(x, ["".join(map(str, strats[i][0]))+"".join(map(str, strats[i][1])) for i in range(n)], rotation='vertical')
    for i in range (n):
        plt.bar(x[i], stationary[i])


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def getTotalDefectCoop(strats, stationary):
    defect_coop = np.zeros(2)
    for i in range(len(strats)):
        action = strats[i][1][0]
        if action == "C":
            defect_coop[1] += stationary[i]
        else:
            defect_coop[0] += stationary[i]
    return defect_coop


def showCoopDefectRatio(strats, stationary):
    x = [i for i in range(2)]  # 2 because one for defection and one for cooperation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("stationary distribution")
    defect_coop = getTotalDefectCoop(strats, stationary)
    colors = ["red", "blue"]
    ax.set_ylim(0, 1)
    plt.xticks(x, ["Defection", "Cooperation"])
    for i in range(2):
        plt.bar(x[i], defect_coop[i], color=colors[i])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def storeAllStationaries(game):
    Z = 150
    stationary_folder = "stationaryDistrib/"+game+"/"
    fix_probs_folder = "fixProbs/"+game+"/"
    subfolder = ["T0/", "T025/", "T05/", "T075/", "T1/"]
    if game == "PD":
        subfolder = ["T1/", "T125/", "T15/", "T175/", "T2/"]

    for nb_states in range (2,6):
        strats = createStrategies(nb_states)
        for cur_subfolder in subfolder:
            cur_fix_probs_folder =  fix_probs_folder + cur_subfolder
            filename = cur_fix_probs_folder + "fixprobs_" + str(nb_states) + "st_0.8_alpha.txt"

            fix_mat = getFreqFromFile(filename,strats)
            trans_mat, stationary = stationaryDistrib(Z, strats, fix_mat)
            stationary_filename = stationary_folder + cur_subfolder + "stationary_"+ str(nb_states) + "st.txt"
            f = open(stationary_filename, "w")
            for i in range (len(strats)):
                line = str(strats[i]) + " : " + str(round(stationary[i], 8))
                f.write(line + "\n")
            f.close()


if __name__ == '__main__':



    Z = 150
    nb_states = 2
    game = "PD"
    #storeAllStationaries(game)
    strats = createStrategies(nb_states)
    #filename = "fixProbs/SH/fixprobs_" + str(nb_states) + "st_0.8_alpha.txt"
    filename = "fixProbs/"+game+"/Rounds/fixprobs_" + str(nb_states) + "st_0.8_alpha.txt"
    #filename = "egt_1st_pd.txt"
    fix_mat = getFreqFromFile(filename, strats)

    #transition_matrix, fix_probs, strats, stationary_dist = load(filename)

    trans_mat, stationary = stationaryDistrib(Z, strats, fix_mat)
    #print(trans_mat)
    for i in range(len(strats)):
        print(strats[i], " : ", round(stationary[i], 8))
    showStationaryDistrib(game, strats, stationary)
    #showCoopDefectRatio(strats, stationary)
