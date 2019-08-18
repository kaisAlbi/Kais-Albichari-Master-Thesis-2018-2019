import matplotlib.pyplot as plt
import json

def getCoopOverTFromFiles():
    coop_over_t = []
    for sig in range (1,6):
        filename = "SH_diff_T/coop_ratio_" + str(sig) + ".txt"
        f = open(filename, "r")
        coop_ratio = json.load(f)
        coop_over_t.append(coop_ratio)

    return coop_over_t


def showCoopOverT():
    y1, y2, y3, y4, y5 = getCoopOverTFromFiles()
    x =  [float(j / 100) for j in range (0,105,5)]
    fig, ax = plt.subplots()
    ax.set_ylabel("Cooperation")
    ax.set_xlabel("T")
    ax.plot(x, y1, marker=".", label="1 signal", color="black")
    ax.plot(x, y2, marker="s", label="2 signal", color="green")
    ax.plot(x, y3, marker="v", label="3 signal", color="blue")
    ax.plot(x, y4, marker="D", label="4 signal", color="orange")
    ax.plot(x, y5, marker="^", label="5 signal", color="red")
    plt.axvline(x=0.5, linestyle='--', color="black")      #half limit

    legend = ax.legend(loc='lower left')

    plt.show()



showCoopOverT()
