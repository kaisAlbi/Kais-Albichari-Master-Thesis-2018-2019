import matplotlib.pyplot as plt
import json
import numpy as np

def getCoopOverTFromFiles(game):
    """
    Read the cooperation evolution from .txt files
    :param game: considered dilemma
    :return: cooperation levels for the three models: signals, internal states, empathy
    """
    coop_over_t = []
    for model_name in ["Signals", "States", "Empathy"]:
        filename = "ModelComparison/" + game + "/" + model_name + "/coop_ratio_3.txt"
        f = open(filename, "r")
        coop_ratio = json.load(f)
        coop_over_t.append(coop_ratio)

    return coop_over_t


def showCoopOverT(game):
    """
    Plot a graph with the evolution of cooperation levels for three models: signals, internal states, empathy
    for respectively 3 signals, 3 states, and the combination of 3 signals with 3 states
    :param game: considered dilemma
    """
    y1, y2, y3= getCoopOverTFromFiles(game)
    x_1 =  [float(j / 100) for j in range (0,105,5)]
    x_2_3 = [float(j / 100) for j in range (0,125,25)]
    if game == "PD":
        x_1 = [float(j / 100) for j in range(100, 205, 5)]
        x_2_3 = [float(j / 100) for j in range (100,225,25)]
    print(x_1)
    print(x_2_3)
    fig, ax = plt.subplots()
    plt.suptitle("Evolution of cooperation")
    plt.title("for 3 signals, 3 states and combination of both")
    ax.set_ylabel("Cooperation")
    x_label = "T(=-S)"
    if game == "PD":
        x_label = "T(=1-S)"
    ax.set_xlabel(x_label)
    ax.plot(x_1, y1, marker=".", label="Signals model", color="black")
    ax.plot(x_2_3, y2, marker="s", label="States model", color="green")
    ax.plot(x_2_3, y3, marker="v", label="Empathy model", color="blue")

    if game == "SH":
        plt.axvline(x=0.5, linestyle='--', color="black")      #half limit

    else:
        plt.axvline(x=1.25, linestyle="--", color="black")
    legend = ax.legend(loc='upper right')

    plt.show()

showCoopOverT("SH")