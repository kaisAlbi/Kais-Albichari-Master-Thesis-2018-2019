import SignalsModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def showSignalsDistribution(egt_model):
    nb_signals = egt_model.getNbSignals()
    stationary = egt_model.getStationaryDistrib()
    strats = egt_model.getStrategies()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("stationary distribution")
    x = [i for i in range(nb_signals)]
    ys = np.zeros(nb_signals)
    for i in range(len(strats)):
        signal = strats[i][0]
        ys[signal] += stationary[i]
    ax.set_ymargin(1)
    color = iter(cm.rainbow(np.linspace(0, 1, nb_signals)))
    plt.xticks(x, ["Signal " + str(i) for i in range(nb_signals)])
    for i in range(nb_signals):
        c = next(color)
        plt.bar(x[i], ys[i], color=c)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def showStationaryDistrib(egt_model):
    strats = egt_model.getStrategies()
    n = len(strats)
    x = [i for i in range(n)]
    stationary = egt_model.getStationaryDistrib()
    nb_signals = egt_model.getNbSignals()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("stationary distribution")

    color = iter(cm.rainbow(np.linspace(0, 1, egt_model.getNbSignals())))
    plt.xticks(x, ["".join(map(str, strats[i])) for i in range(n)], rotation='vertical')
    for i in range(nb_signals):
        c = next(color)
        group_size = int(len(strats) / nb_signals)
        start = int(i * group_size)
        for j in range(start, start + group_size):
            m = max(stationary)
            delta_m = 0.0001
            if j in [k for k, l in enumerate(stationary) if (l > m - delta_m and l < m + delta_m)]:
                plt.bar(x[j], stationary[j], color=c, hatch="xx")
            else:
                plt.bar(x[j], stationary[j], color=c)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def getTotalDefectCoop(strats, stationary):
    defect_coop = np.zeros(2)
    for i in range(len(strats)):
        signal = strats[i][0]
        if strats[i][signal + 1] == "C":
            defect_coop[1] += stationary[i]
        else:
            defect_coop[0] += stationary[i]
    return defect_coop


def showCoopDefectRatio(egt_model):
    strats = egt_model.getStrategies()
    x = [i for i in range(2)]  # 2 because one for defection and one for cooperation
    stationary = egt_model.getStationaryDistrib()
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

R, S, T, P = 1., -0.5, 0.5, 0.       #SH first config
Z = 150 #Population size
beta = 0.05
egt_2sig= SignalsModel.EGTModel(R, S, T, P, Z, beta, 2)

showSignalsDistribution(egt_2sig)
showStationaryDistrib(egt_2sig)
showCoopDefectRatio(egt_2sig)