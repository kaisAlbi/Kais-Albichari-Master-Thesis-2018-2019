import csv
import matplotlib.pyplot as plt
from math import pow
import numpy as np

def getCoopForStatesFromFiles(game):
    filename = "EvalCoop/"+game+"/coop_evolution.csv"
    coop_ratios = []
    nb_strats = []
    f = open(filename, "r")
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader:
        for (key,value) in row.items():
            if key == "Cooperation":
                coop_ratios.append(float(value))
            if key == "NbStates":
                nb_strat = pow(2, 2 + int(value))
                nb_strats.append(int(nb_strat))
    f.close()
    return coop_ratios, nb_strats

def showCoopOverStates(game):
    coop_ratios, nb_strats = getCoopForStatesFromFiles(game)
    fig = plt.figure()
    plt.title(game)
    ax = fig.add_subplot(111)
    ax.set_ylabel("Cooperation")
    ax.set_xlabel("number of states")
    for i in range (len(coop_ratios)):
        if i == 0:
            plt.bar(1, coop_ratios[i], color = "black")
        else:
            plt.bar(i+1, coop_ratios[i], color = "orange")

    plt.xticks(np.arange(0, len(coop_ratios)+1, step=1))
    patches = ax.patches
    plt.ylim(0,1)
    for i in range (len(patches)):
        patch = patches[i]

        text = str(nb_strats[i])+" strategies"
        text_size = len(text)/70
        ax.text(patch.get_xy()[0], patch.get_height()+ text_size, text, fontsize=10, rotation=45)
        if i == 0:
            text = "Conventional "+game
            ax.text(patch.get_xy()[0]+0.2, patch.get_height()/1.5, text, fontsize=10, rotation="vertical", color="white")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()



game = "SH"
#getCoopForStatesFromFiles(game)
showCoopOverStates(game)
#showRatio()