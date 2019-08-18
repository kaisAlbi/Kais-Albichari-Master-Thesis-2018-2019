from analyzePayoffsDistrib import createStrategies, displayableStrat
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def coopRatio(proba_c_per_strat, strats, stationary):

    coop_defect = np.zeros(2)
    for i in range (len(strats)):
        coop_defect[0] += proba_c_per_strat[i] * stationary[i]
        coop_defect[1] += (1 - proba_c_per_strat[i]) * stationary[i]
    return coop_defect

def getProbaPerStrat(proba_couples_filename, strats):
    n = len(strats)
    proba_c_per_strat = np.zeros(n)
    with open(proba_couples_filename, "r") as f:
        proba_c_pairs = np.asarray(json.load(f))
        for i in range(n):
            proba_c_i = 0
            for j in range(n):
                proba_c_i += proba_c_pairs[i, j]
            proba_c_i /= n
            proba_c_per_strat[i] = proba_c_i
    return proba_c_per_strat

def getStationaryFromFile(filename):
    stationary = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            index = line.index(": ")
            stationary_i = line[index + 1:]
            stationary.append(float(stationary_i))
    return stationary


def plotCoopDefect(game, proba_c_per_strat, strats, stationary):
    fig = plt.figure()
    plt.title(game)
    ax = fig.add_subplot(111)
    ax.set_ylabel("cooperation ratio")
    x = [i for i in range(2)]  # 2 because one for defection and one for cooperation
    coop_defect = coopRatio(proba_c_per_strat, strats, stationary)
    print(coop_defect)
    colors = ["blue", "red"]
    ax.set_ylim(0, 1)
    plt.xticks(x, ["Cooperation", "Defection"])
    for i in range(2):
        plt.bar(x[i], coop_defect[i], color=colors[i])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

def plotCoopEvolStates(game, max_nb_states):
    coop_for_states = np.zeros(max_nb_states)
    for m in range (max_nb_states):
        strats = createStrategies(m+1)
        proba_c_filename = "ProbaCouples/proba_couples_matrix_" + str(m+1) + "st.txt"
        proba_c_per_strat = getProbaPerStrat(proba_c_filename, strats)
        stationary_filename = "stationaryDistrib/" + game + "/Rounds/stationary_" + str(m+1) + "st.txt"
        stationary = getStationaryFromFile(stationary_filename)
        coop_for_states[m] = coopRatio(proba_c_per_strat, strats, stationary)[0]

    x = np.linspace(1,max_nb_states, max_nb_states)

    fig, ax = plt.subplots()
    plt.title(game)
    plt.xticks(x, [int(x[i]) for i in range (len(x))])
    for i in range(len(x)):
        plt.bar(x[i], coop_for_states[i])
    ax.set_ylabel("Cooperation ratio")
    ax.set_xlabel("Number of States")
    plt.show()

def coopRatioDiffTrans(game, rho_list, nb_states, strats):
    n_config = len(rho_list)
    coop_every_config = np.zeros((n_config, n_config))
    for i in range (len(rho_list)):
        for j in range (len(rho_list)):
            index = i * len(rho_list) + j
            stationary_filename = "stationaryDistrib/" + game + "/DiffTrans/stationary_" + str(nb_states) + "st_" + str(
                index) + ".txt"
            proba_filename = "ProbaCouples/" + game + "/DiffTrans/proba_couples_matrix_" + str(nb_states) + "st_" + str(
                index) + ".txt"
            stationary = getStationaryFromFile(stationary_filename)
            proba_c_per_strat = getProbaPerStrat(proba_filename, strats)
            coop_defect = coopRatio(proba_c_per_strat, strats, stationary)
            coop_every_config[i, j] = coop_defect[0]
    return coop_every_config


def plotEveryConfigCoop(game, rho_list, nb_states, strats):
    coop_every_config= coopRatioDiffTrans(game, rho_list, nb_states, strats)

    x = [float(rho) for rho in rho_list]
    markers = [".", "s", "+", "D", "^", "*", "x", "p", "2", "8", ">"]

    fig, ax = plt.subplots()
    plt.title(game)
    ax.set_ylabel("Cooperation")
    x_label = "P(going state 0)"
    ax.set_xlabel(x_label)
    color = iter(cm.rainbow(np.linspace(0, 1, len(rho_list))))
    for i in range (len(rho_list)):
        cur_color = next(color)
        ax.plot(x, coop_every_config[i], color=cur_color, marker=markers[i], label="P(going state 1) = "+str(rho_list[i]))

    legend = ax.legend(loc='best')#, bbox_to_anchor=(0.85, 0.7))

    plt.show()

if __name__ == '__main__':
    game = "PD"
    nb_states = 3
    #strats = createStrategies(nb_states)
    rho_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #plotEveryConfigCoop(game, rho_list, nb_states, strats)

    plotCoopEvolStates(game, 3)

    #stationary_filename = "stationaryDistrib/" + game + "/Rounds/stationary_" + str(nb_states) + "st.txt"
    #proba_couples_filename = "ProbaCouples/" + game + "/proba_couples_matrix_" + str(nb_states) + "st.txt"
    #proba_c_per_strat = getProbaPerStrat(proba_couples_filename, strats)

    #stationary = getStationaryFromFile(stationary_filename)
    #for i in range (len(strats)):
    #    print(displayableStrat(strats[i]), " cooperate with a probability of ", proba_c_per_strat[i])
    #plotCoopDefect(game, proba_c_per_strat, strats, stationary)