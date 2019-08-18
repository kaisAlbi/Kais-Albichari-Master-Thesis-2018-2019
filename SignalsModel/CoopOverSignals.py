import json
import matplotlib.pyplot as plt

def getCoopRatio(strats, stationary_dist):
    coop_ratio = 0
    for i in range (len(strats)):
        signal = strats[i][0]
        if strats[i][signal+1] == "C":
            coop_ratio += stationary_dist[i]
    return coop_ratio

def getCoopForSigFromFiles():
    coop_over_t = []
    nb_strats = []
    for sig in range (1,7):
        filename = "coopOverSignals/" + str(sig) + "sig.txt"
        f = open(filename, "r")
        transition_matrix, fix_probs, strats, stationary_dist = json.load(f)
        coop_ratio = getCoopRatio(strats,stationary_dist)
        coop_over_t.append(coop_ratio)
        nb_strats.append(len(strats))
    return coop_over_t, nb_strats

def getTransitionFromFiles():
    """
    Method used to read the transition probabilities from files, depending on the signals
    Then calculates the ratio between the accumulated transition probabilities of monomorphic states with the same
    signal and monomorphic states with different signals
    :return: array of ratios
    """
    trans_ratio_over_t = []
    for sig in range (1,6):
        filename = "coopOverSignals/" + str(sig) + "sig.txt"
        f = open(filename, "r")
        transition_matrix, fix_probs, strats, stationary_dist = json.load(f)
        trans_same_sig = 0
        trans_diff_sig = 0
        for i in range (len(fix_probs)):
            for j in range (len(fix_probs[i])):
                if  i!=j:
                    if strats[i][0] == strats[j][0]:        #Same sig
                        trans_same_sig += transition_matrix[i][j]
                    else:
                        trans_diff_sig += transition_matrix[i][j]
        trans_ratio_over_t.append(trans_diff_sig/trans_same_sig)
    return trans_ratio_over_t

def showCoopOverSig():
    coop_over_t, nb_strats = getCoopForSigFromFiles()

    y1, y2, y3, y4, y5, y6 = coop_over_t
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("Cooperation")
    ax.set_xlabel("number of signals")
    plt.bar(1,y1,color="black")
    plt.bar(2, y2,color="orange")
    plt.bar(3, y3,color="orange")
    plt.bar(4, y4,color="orange")
    plt.bar(5, y5,color="orange")
    plt.bar(6, y6,color="orange")
    patches = ax.patches
    for i in range (len(patches)):
        patch = patches[i]
        if i!=0:
            text = str(nb_strats[i])+" strategies"
            text_size = len(text)/80
            ax.text(patch.get_xy()[0], patch.get_height()+ text_size, text, fontsize=10, rotation=45)
        else:
            text = "Conventional SH \n with 2 strategies"
            ax.text(patch.get_xy()[0]+0.2, patch.get_height()/2, text, fontsize=10, rotation="vertical", color="white")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def showRatio():
    ratio1, ratio2, ratio3, ratio4, ratio5 = getTransitionFromFiles()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("ratio")
    ax.set_xlabel("number of signals")
    plt.bar(1, ratio1, color="blue")
    plt.bar(2, ratio2,color="blue")
    plt.bar(3, ratio3,color="blue")
    plt.bar(4, ratio4,color="blue")
    plt.bar(5, ratio5,color="blue")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

showCoopOverSig()
#showRatio()