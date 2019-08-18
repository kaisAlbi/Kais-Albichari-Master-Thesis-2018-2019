import numpy as np
import csv
import matplotlib.pyplot as plt

def getCoopOverNFromFiles(game, nb_states):
    """
    Read series of files to get the cooperation ratio in the right files depending on the game and the number of
    internal states considered
    :param game: dilemma considered
    :param nb_states: number of internal states
    :return: matrix containing the cooperation ratio for each number of internal state, and each different T value
    considered
    """
    folder = "DifferentN/EvalCoop/"+game+"/"
    N_subfolders = ["N1/", "N5/", "N10/", "N15/"]

    coop_for_states_diff_t = np.zeros((nb_states, len(N_subfolders)))
    for i in range (len(N_subfolders)):
        cur_filename = folder + N_subfolders[i] + "coop_evolution.csv"

        with open(cur_filename, "r") as f:
            reader = csv.DictReader(f)  # read rows into a dictionary format
            for row in reader:
                state = int(row["NbStates"])
                coop_ratio = float(row["Cooperation"])
                coop_for_states_diff_t[state-1, i] = coop_ratio
    return coop_for_states_diff_t



def showCoopOverN(game, nb_states):
    """
    Show the evolution of cooperation in function of the number of rounds N and number of internal states values, given a
    specified dilemma
    :param game: dilemma considered
    :param nb_states: number of internal states
    """
    coop_for_states_diff_n = getCoopOverNFromFiles(game, nb_states)

    x = [1,5,10,15]

    fig, ax = plt.subplots()
    plt.title(game)
    ax.set_ylabel("Cooperation")
    markers = [".", "s", "+"]
    x_label = "number of rounds N"
    ax.set_xlabel(x_label)
    ax.plot(x, coop_for_states_diff_n[0], marker=markers[0], label= "1 state")
    for i in range (1,nb_states):
        ax.plot(x, coop_for_states_diff_n[i], marker = markers[i], label = str(i+1)+" states")

    legend = ax.legend(loc='left')

    plt.show()


if __name__ == '__main__':

    game = "SH"
    nb_states = 3

    showCoopOverN(game, nb_states)