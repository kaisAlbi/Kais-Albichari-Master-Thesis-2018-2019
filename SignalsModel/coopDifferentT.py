import glob
import json


def coopDifferentT(game):
    """
    Compute the evolution of cooperation, given a game, for different temptation to defect values and different number
    of signals, and store them into different files
    :param game: considered dilemma
    """
    if game == "SH":
        for sig in range(1, 6):
            loc = "SH_diff_T/"
            coop_ratio = []
            for filename in glob.glob(loc + "*_" + str(sig) + "_*.txt"):
                f = open(filename, "r")
                transition_matrix, fix_probs, strats, stationary_dist = json.load(f)
                coop_ratio.append(getCoopRatio(strats, stationary_dist))

            fopen = open(loc+"coop_ratio_"+str(sig)+".txt", "w")
            json.dump(coop_ratio, fopen)
            fopen.close()
    else:
        for sig in range(1, 7):
            loc = "PD_diff_T/"
            coop_ratio = []
            for filename in glob.glob(loc + "*_" + str(sig) + "_*.txt"):
                f = open(filename, "r")
                transition_matrix, fix_probs, strats, stationary_dist = json.load(f)
                coop_ratio.append(getCoopRatio(strats, stationary_dist))
            fopen = open(loc+"coop_ratio_"+str(sig)+".txt", "w")
            json.dump(coop_ratio, fopen)
            fopen.close()



def getCoopRatio(strats, stationary_dist):
    """
    Compute the cooperation ratio of a population, given a list of strategies and their corresponding strationary distributions
    :param strats: strategies
    :param stationary_dist: stationary distributions
    :return: cooperation ratio
    """
    coop_ratio = 0
    for i in range (len(strats)):
        signal = strats[i][0]
        if strats[i][signal+1] == "C":
            coop_ratio += stationary_dist[i]
    return coop_ratio

coopDifferentT("SH")
coopDifferentT("PD")

