import numpy as np
import csv
import matplotlib.pyplot as plt

def getCoopOverTFromFiles(game, nb_states):
    """
    Read series of files to get the cooperation ratio in the right files depending on the game and the number of
    internal states considered
    :param game: dilemma considered
    :param nb_states: number of internal states
    :return: matrix containing the cooperation ratio for each number of internal state, and each different T value
    considered
    """
    folder = "EvalCoop/"+game+"/Different_T/"
    t_subfolders = ["T0/", "T025/", "T05/", "T075/", "T1/"]
    if game == "PD":
        t_subfolders = ["T1/", "T125/", "T15/", "T175/", "T2/"]

    coop_for_states_diff_t = np.zeros((nb_states, len(t_subfolders)))
    for i in range (len(t_subfolders)):
        cur_filename = folder + t_subfolders[i] + "coop_evolution.csv"

        with open(cur_filename, "r") as f:
            reader = csv.DictReader(f)  # read rows into a dictionary format
            for row in reader:
                state = int(row["NbStates"])
                coop_ratio = float(row["Cooperation"])
                coop_for_states_diff_t[state-1, i] = coop_ratio
    return coop_for_states_diff_t



def showCoopOverT(game, nb_states):
    """
    Show the evolution of cooperation in function of T values and number of internal states values, given a
    specified dilemma
    :param game: dilemma considered
    :param nb_states: number of internal states
    """
    coop_for_states_diff_t = getCoopOverTFromFiles(game, nb_states)

    x = [float(j / 100) for j in range (0,125,25)]
    markers = [".", "s", "+", "D", "^"]
    if game == "PD":
        x = [float(j / 100) for j in range (100,225,25)]

    fig, ax = plt.subplots()
    plt.title(game)
    ax.set_ylabel("Cooperation")
    x_label = "T(=-S)"
    if game == "PD":
        x_label = "T(=1-S)"
    ax.set_xlabel(x_label)
    ax.plot(x, coop_for_states_diff_t[0], marker=markers[0], label= "1 state")
    for i in range (1, nb_states):
        ax.plot(x, coop_for_states_diff_t[i], marker = markers[i], label = str(i+1)+" states")

    if game == "SH":
        plt.axvline(x=0.5, linestyle='--', color="black")  # half limit

    legend = ax.legend(loc='lower left')

    plt.show()

game = "SH"
nb_states = 5

showCoopOverT(game, nb_states)