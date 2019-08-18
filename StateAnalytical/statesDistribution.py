import numpy as np
import csv
from AnalyticalExprGroups import createStrategies, reducedStrategies


def getStatesDistribFromFile(filename, strats, nb_states):
    n_strats = len(strats)
    nb_run = 0
    states_dist = np.zeros((n_strats*nb_states), dtype=np.float64)
    print(filename)
    with open(filename, "r") as f:
        reader = csv.reader(f)
        c = 1
        for line in reader:
            for i in range (len(line)):
                states_dist[i] += float(line[i])
            nb_run += 1
            c += 1
        states_dist = np.divide(states_dist, nb_run)
    states_dist = states_dist.reshape((n_strats, nb_states))
    return states_dist

def getStationaryFromFile(filename):
    stationary = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            index = line.index(": ")
            stationary_i = line[index + 1:]
            stationary.append(float(stationary_i))
    return stationary


def getStatesDistribution(game):
    states_dist = []
    param_folder = "Different_T/"
    if game == "SH":
        param_folder += "T05/"
    else:
        param_folder += "T15/"
    folder = "EvalCoop/"+game+ "/"+param_folder
    for nb_states in range(2, 6):
        strats = reducedStrategies(createStrategies(nb_states))
        stationary_filename = "stationaryDistrib/" + game + "/Groups/" + param_folder + "stationary_" + str(nb_states) + "st.txt"
        stationary = getStationaryFromFile(stationary_filename)

        eval_coop_filename = folder + "eval_coop_" + str(nb_states) + "st.csv"
        states_dist_by_strat = getStatesDistribFromFile(eval_coop_filename, strats, nb_states)
        print("states dist for "+str(nb_states)+ " states : ")
        print(states_dist_by_strat)
        states_dist.append(np.sum(states_dist_by_strat, axis=0))
        print(np.sum(states_dist_by_strat, axis=0))
    return states_dist

def plotStatesDist(game, states_dist):
    n = len(states_dist)


game = "SH"
param_folder = "Different_T/T05/"
states_dist = getStatesDistribution("SH")
plotStatesDist(game, states_dist)