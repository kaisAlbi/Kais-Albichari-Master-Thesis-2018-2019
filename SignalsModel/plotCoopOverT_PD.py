import matplotlib.pyplot as plt
import json

def getCoopOverTFromFiles():
    coop_over_t = []
    for sig in range (1,7):
        filename = "PD_diff_T/coop_ratio_" + str(sig) + ".txt"
        f = open(filename, "r")
        coop_ratio = json.load(f)
        coop_over_t.append(coop_ratio)

    return coop_over_t


def showCoopOverT():
    y1, y2, y3, y4, y5, y6 = getCoopOverTFromFiles()
    x =  [float(j / 100) for j in range (100,205,5)]
    fig, ax = plt.subplots()
    ax.set_ylabel("Cooperation")
    ax.set_xlabel("T(=1-S)")
    ax.plot(x, y1, marker=".", label="1 signal", color="black")
    ax.plot(x, y2, marker="s", label="2 signal", color="green")
    ax.plot(x, y3, marker="+", label="3 signal", color="blue")
    ax.plot(x, y4, marker="D", label="4 signal", color="orange")
    ax.plot(x, y5, marker="^", label="5 signal", color="grey")
    ax.plot(x, y6, marker="v", label="6 signal", color="red")

    legend = ax.legend(loc='best', bbox_to_anchor=(0.85, 0.7))

    plt.show()

showCoopOverT()