import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import json
import time
import networkx as nx
from matplotlib.lines import Line2D

class EGTModel:
    def __init__(self, payoffs_filename, pop_size, beta, nb_states):
        self.Z = pop_size
        self.beta = beta
        self.nb_states = nb_states
        self.strategies = self.createStrategies()
        self.payoff_pairs = self.loadPayoffs(payoffs_filename)

        self.transition_proba, self.fix_probs, self.stationary = self.stationaryDistrib()

    def getNbStates(self):
        return self.nb_states

    def getStrategies(self):
        return self.strategies

    def getBeta(self):
        return self.beta

    def getPopSize(self):
        return self.Z

    def getFixProbs(self):
        return self.fix_probs

    def getTransitionMatrix(self):
        return self.transition_proba

    def getStationaryDistrib(self):
        return self.stationary


    def loadPayoffs(self, filename):
        f = open(filename, "r")
        payoff_pairs = json.load(f)
        return np.asarray(payoff_pairs)

    def createStrategies(self):
        """
        Generate all possible combination of strategies, depending on the number of states
        :return: list of all strategies in the form [T_c,T_d, action state_0, ..., action state_maxState]
                transition = 1 = Left ; action = 1 = C
        """

        action_choice = list(list(item) for item in product("CD", repeat=self.getNbStates()))
        state_change = list(list(item) for item in product("LR", repeat=2))
        strats = []
        for action in action_choice:
            action_c_tr = []
            for i in range (len(action)):

                if action[i] == "C":
                    action_c_tr.append(1)
                else:
                    action_c_tr.append(0)

            for state_c in state_change:
                state_c_tr = []
                if state_c[0] == "L":
                    state_c_tr.append(1)
                else:
                    state_c_tr.append(0)
                if state_c[1] == "L":
                    state_c_tr.append(1)
                else:
                    state_c_tr.append(0)
                list_c_tr = [state_c_tr, action_c_tr]
                strat = [item for sublist in list_c_tr for item in sublist]
                strats.append(strat)
        return strats


    def fermiDistrib(self, first_fitness, second_fitness, positive):
        """
        :param first_payoff: payoff obtained by the first agent after the interaction
        :param second_payoff: payoff obtained by the second agent after the interaction
        :param positive: boolean value used to calculate probability increase and decrease (T +-)
        :return: probability that the first agent imitates the second ond
        """
        if positive:
            return 1. / (1. + np.exp(-self.getBeta() * (first_fitness - second_fitness)))
        else:
            return 1. / (1. + np.exp(self.getBeta() * (first_fitness - second_fitness)))


    def probIncDec(self, n_A, p_inv_inv, p_inv_res, p_res_res, p_res_inv):
        """
        :param n_A: number of invaders
        :param inv_strat: invaders' strategy
        :param res_strat: residents' strategy
        :return: probability to change the number of k invaders (by +- one at each time step)
        """
        result_inv = (n_A * p_inv_inv + (self.Z - n_A) * p_inv_res) / float(self.Z)
        result_res = (n_A * p_res_inv + (self.Z - n_A) * p_res_res) / float(self.Z)

        tmp = ((self.Z-n_A) / Z) * (n_A / Z)
        inc = np.clip(tmp * self.fermiDistrib(result_res, result_inv, False), 0., 1.)
        dec = np.clip(tmp * self.fermiDistrib(result_res, result_inv, True), 0., 1.)
        return [inc, dec]


    def fixationProba(self, res_index, inv_index):
        """
        :param res_index: resident index
        :param inv_index: invader index
        :return: fixation probability of the invader in a population of residents
        """

        p_inv_inv = self.payoff_pairs[inv_index, inv_index]
        p_inv_res = self.payoff_pairs[inv_index, res_index]
        p_res_res = self.payoff_pairs[res_index, res_index]
        p_res_inv = self.payoff_pairs[res_index, inv_index]

        result = 0.
        for i in range(0, self.Z):
            mul = 1.
            for j in range(1, i + 1):
                inc, dec = self.probIncDec(j, p_inv_inv, p_inv_res, p_res_res, p_res_inv)
                lambda_j = np.float(dec / float(inc))
                mul *= lambda_j
            result += mul

        return np.clip(1. / result, 0., 1.)

    def transitionMatrix(self):
        """
        Compute the fixation probability for each pair invader-resident of strategies and build the fixation probabilities
        matrix and the transition matrix
        :return: transition matrix and fixation probabilities matrix
        """
        strats = self.getStrategies()
        n = len(strats)
        drift = 1 / float(self.Z)
        norm_fact = 1 / float((n - 1))
        fix_probs = np.zeros((n, n))
        transitions = np.zeros((n, n))
        for i in range(n):
            start_time = time.time()
            transitions[i, i] = 1
            for j in range(n):
                if i != j:
                    f_proba = self.fixationProba(i,j)
                    fix_probs[i, j] = f_proba
                    trans_value = f_proba * norm_fact
                    transitions[i, j] = trans_value
                    transitions[i, i] -= trans_value
            print("transitions values calculations for resident strat ", strats[i], " took --- %s seconds---" % (time.time() - start_time))
        return [transitions, fix_probs]



    def stationaryDistrib(self):
        """
        Calculate the transition matrix, and based on that matrix, the stationary distribution of each strategy
        :return: transition matrix, fixation probabilities matrix, stationary distribution
        """
        t, f = self.transitionMatrix()
        val, vect = np.linalg.eig(t.transpose())
        j_stationary = np.argmin(abs(val - 1.0))  # look for the element closest to 1 in the list of eigenvalues
        p_stationary = abs(vect[:, j_stationary].real)  # the, is essential to access the matrix by column
        p_stationary /= p_stationary.sum()  # normalize
        return t, f, p_stationary


def makeXTicks(strats):
    """
    Transforms the binary strategies into 'x_ticks' to plot them on a graph
    :param strats: array of binary strategies
    :return: array of transformed strategies
    """
    x_ticks = []
    for strat in strats:
        x_tick = []
        for i in range (len(strat)):
            if i == 0 or i == 1:
                if strat[i] == 1:
                    x_tick.append("L")
                else:
                    x_tick.append("R")
            else:
                if strat[i] == 1:
                    x_tick.append("C")
                else:
                    x_tick.append("D")
        x_ticks.append(x_tick)
    return ["".join(map(str, x_ticks[i])) for i in range(len(strats))]

def showStationaryDistrib(game, strats, stationary):
    """
    Plot a bar graph showing the stationary distribution
    :param game: evolutionary game being played
    :param strats: array of binary strategies
    :param stationary: stationary distribution array
    """
    n = len(strats)
    x = [i for i in range(n)]

    fig = plt.figure()
    plt.title(game)
    ax = fig.add_subplot(111)
    ax.set_ylabel("stationary distribution")

    plt.xticks(x, makeXTicks(strats), rotation='vertical')
    for i in range (n):
        plt.bar(x[i], stationary[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def plotMultipleStationary(game, setStrats, setStationary):
    n= len(setStrats)

    nb_strats = len(setStrats[0])
    x = [i for i in range(nb_strats)]

    fig = plt.figure()
    for index in range (n):
        ax = plt.subplot(n/2, n/2, index+1)
        strats = setStrats[index]
        plt.xticks(x, makeXTicks(strats), rotation='vertical')
        plt.ylabel("stationary distribution")
        plt.title(game)
        for i in range(n):
            plt.bar(x[i], setStationary[index][i])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.show()

    plt.show()



def displayableStrat(strat):
    """
    Transforms a binary strategy array into a human readable strategy array
    :param strat: strategy array
    :return: human readable strategy array
    """
    displayable_strat = []
    for i in range (len(strat)):
        if strat[i] == 1:
            if i < 2:
                displayable_strat.append("L")
            else:
                displayable_strat.append("C")
        else:
            if i < 2:
                displayable_strat.append("R")
            else:
                displayable_strat.append("D")
    return displayable_strat


def getStationaryFromFile(filename):
    f = open(filename, "r")
    stationary = []
    for line in f.readlines():
        index = line.index(": ")
        stationary_i = line[index + 1:]
        stationary.append(float(stationary_i))

    r = 1 - sum(stationary)  # Handle loss of precision
    stationary[0] += r
    f.close()
    return stationary

def storeStationary(filename, strats, stationary):

    with open(filename, "w") as f:
        for i in range(len(strats)):
            line = "".join(map(str, strats[i])) + " : " + str(stationary[i])
            if i < (len(strats) - 1):
                line += "\n"
            f.write(line)

def hasOnlyOneDirection(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible transition direction, False otherwise
    """
    directions = strat[:2]
    if 1 not in directions or 0 not in directions:
        return True
    return False

def hasOnlyOneAction(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible action choice, False otherwise
    """
    actions = strat[2:]
    if 1 not in actions or 0 not in actions:
        return True
    return False

def getColorsForStrats(strats):
    colors = []
    for i in range(len(strats)):
        if hasOnlyOneAction(strats[i]):
            if strats[i][2] == 1:
                colors.append("blue")
            else:
                colors.append("red")
        else:
            if hasOnlyOneDirection(strats[i]):
                if strats[i][0] == 1:
                    if strats[i][2] == 1:
                        colors.append("blue")
                    else:
                        colors.append("red")
                else:
                    if strats[i][-1] == 1:
                        colors.append("blue")
                    else:
                        colors.append("red")
            else:

                colors.append("purple")
    return colors



def showTransitionForStrategies(game, egt_model):
    strats_mat = egt_model.getStrategies()
    n = len(strats_mat)
    strats = makeXTicks(strats_mat)
    fix_probs = egt_model.getFixProbs()
    transition = egt_model.getTransitionMatrix()
    stationary = egt_model.getStationaryDistrib()
    drift = 1 / egt_model.getPopSize()
    G = nx.DiGraph(directed=True)
    ncolors = getColorsForStrats(strats_mat)
    G.add_nodes_from(strats)
    for j in range(n):
        for i in range(n):
            if fix_probs[i, j] > drift:
                G.add_edge(strats[i], strats[j], weight=fix_probs[i, j])
    """PD
    eselect_6 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > 6 * drift and (u, v))]
    eselect_4 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > 4 * drift and (u, v) and (u, v) not in eselect_6)]
    #eselect_3 = [(u, v) for (u, v, d) in G.edges(data=True) if
    #             (d['weight'] > 3 * drift and (u, v) and (u, v) not in eselect_6 and (u,v) not in eselect_4)]
    #eselect_2 = [(u, v) for (u, v, d) in G.edges(data=True) if
    #             (d['weight'] > 2 * drift and (u, v) not in eselect_6 and (u, v) not in eselect_4 and (u, v) not in eselect_3)]
    #eselect_1 = [(u, v) for (u, v, d) in G.edges(data=True) if
    #             (d['weight'] > drift and (u, v) not in eselect_6 and (u, v) not in eselect_4 and (u, v) not in eselect_3 and (u, v) not in eselect_2)]

    """

    #SH
    eselect_4 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > 4 * drift and (u, v))]
    eselect_2 = [(u, v) for (u, v, d) in G.edges(data=True) if
                          (d['weight'] > 2 * drift and (u,v) not in eselect_4)]
    eselect_1 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > drift and (u,v) not in eselect_4 and (u,v) not in eselect_2)]

    eselect_labels = dict(((u, v), float("{0:.6f}".format(d['weight'])))
                          for (u, v, d) in G.edges(data=True))

    fig = plt.figure(figsize=(10, 10), dpi=150)
    plt.title(game)
    nodes_labels = {}
    for i in range(len(strats)):
        nodes_labels["".join(map(str, strats[i]))] = "".join(
            map(str, strats[i] + "\n" + str(round(stationary[i] * 100, 2)) + "%"))
    #strats = [LLCC, LRCC, RLCC, RRCC, LLCD, LRCD, RLCD, RRCD, LLDC, LRDC, RLDC, RRDC, LLDD, LRDD, RLDD, RRDD]
    pos = {strats[0]: np.array([5, 25]), strats[1]: np.array([10, 25]), strats[2]: np.array([15, 25]), strats[3]: np.array([20, 25]),
           strats[4]: np.array([0, 20]), strats[5]: np.array([-2, 15]), strats[6]: np.array([-2, 10]) ,strats[7]: np.array([0, 5]),
           strats[8]: np.array([25, 20]), strats[9]: np.array([27, 15]), strats[10]: np.array([27, 10]), strats[11]: np.array([25, 5]),
           strats[12]: np.array([5, 0]), strats[13]: np.array([10, 0]), strats[14]: np.array([15, 0]),strats[15]: np.array([20, 0])}
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=ncolors, with_labels=True)

    # edges
    """PD
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_6,
                                            width=4, arrowsize=40, edge_color="black")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_4,
                                            width=3, arrowsize=30, edge_color="grey")
    """

    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_4,
                                            width=3, arrowsize=30, edge_color="grey")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_2,
                                            width=2, arrowsize=20, edge_color="blue")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_1,
                                            width=0.5, arrowsize=10, edge_color="orange")



    # node labels

    nx.draw_networkx_labels(G, pos, nodes_labels,
                            font_size=8, font_weight='bold', font_color='white')

    # edge labels
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=eselect_labels, font_size=14)
    #plt.legend(numpoints=1)

    #SH
    legend_elements = [Line2D([0], [0], marker='$\leftarrow$', color='orange', lw=1, label='$>'+str("\\")+'rho_N$',
                              markerfacecolor='orange', markersize=25),
                       Line2D([0], [0], marker='$\leftarrow$', color='blue', lw=1, label='$>2' + str("\\") + 'rho_N$',
                              markerfacecolor='blue', markersize=25),
                       Line2D([0], [0], marker='$\leftarrow$', color='grey',lw=1, label='$>4'+str("\\")+'rho_N$',
                              markerfacecolor='grey', markersize=25),

                       ]


    """PD
    legend_elements = [Line2D([0], [0], marker='$\leftarrow$', color='grey',lw=1, label='$>4'+str("\\")+'rho_N$',
                              markerfacecolor='grey', markersize=25),
                       Line2D([0], [0], marker='$\leftarrow$', color='black', lw=1, label='$>6' + str("\\") + 'rho_N$',
                              markerfacecolor='black', markersize=25)
                       ]
    """
    plt.legend(handles=legend_elements, loc='upper right')

    plt.axis('off')
    plt.show()  # display


if __name__ == '__main__':
    game = "SH"

    Z = 150 #Population size
    beta = 0.05
    nb_states = 2
    rounds = 10

    #rho = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    #for i in range(len(rho)):

    #    for j in range(len(rho)):
    #        index = i * len(rho) + j
    #        payoffs_filename = "PayoffPairs/" + game + "/DiffTrans/payoffs_pairs_" + str(nb_states) + "st_" + str(
    #            index) + ".txt"
            #trans_proba = [rho[i], rho[j]]
    #        egt_model = EGTModel(payoffs_filename, Z, beta, nb_states)
    #        stationary_filename = "stationaryDistrib/" + game + "/DiffTrans/stationary_" + str(nb_states) + "st_" + str(
    #            index) + ".txt"
    #        storeStationary(stationary_filename, egt_model.getStrategies(), egt_model.getStationaryDistrib())
    #    print("line ", i, " done")

    #payoffs_filename = "PayoffPairs/Simplified/" + game + "/payoffs_pairs_" + str(nb_states) + "st.txt"
    payoffs_filename = "PayoffPairs/" + game + "/" + str(rounds) + "rounds/payoffs_pairs_" + str(nb_states) + "st.txt"
    #payoffs_filename = "PayoffPairs/" + game + "/TitForTat/payoffs_pairs_" + str(nb_states) + "st_10rounds_1cheater.txt"
    #payoffs_filename = game + "_payoffs_pairs_" + str(nb_states) + "st.txt"
    egt_2states = EGTModel(payoffs_filename, Z, beta, nb_states)
    stationary = egt_2states.getStationaryDistrib()
    strats = egt_2states.getStrategies()
    #showTransitionForStrategies(game, egt_2states)
    for i in range(len(strats)):
        print(strats[i], " : ", round(stationary[i], 8))
    #showStationaryDistrib(game, strats, stationary)
    showTransitionForStrategies(game, egt_2states)

    #strat_ll = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0]]
    #strat_lr = [[1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0]]
    #strat_rl = [[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0]]
    #strat_rr = [[0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]

    #strat_cc = [[1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]]
    #strat_cd = [[1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]]
    #strat_dc = [[1, 1, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]]
    #strat_dd = [[1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    #set_strats = [strat_cc, strat_cd, strat_dc, strat_dd]#[strat_ll, strat_lr, strat_rl, strat_rr]#, strat_cc, strat_cd, strat_dc, strat_dd]
    #stationary_ll = getStationaryFromFile("stationaryDistrib/SH/Subset_Analysis/stationary_LL.txt")
    #stationary_lr = getStationaryFromFile("stationaryDistrib/SH/Subset_Analysis/stationary_LR.txt")
    #stationary_rl = getStationaryFromFile("stationaryDistrib/SH/Subset_Analysis/stationary_RL.txt")
    #stationary_rr = getStationaryFromFile("stationaryDistrib/SH/Subset_Analysis/stationary_RR.txt")
    #stationary_cc = getStationaryFromFile("stationaryDistrib/SH/Subset_Analysis/stationary_CC.txt")
    #stationary_cd = getStationaryFromFile("stationaryDistrib/SH/Subset_Analysis/stationary_CD.txt")
    #stationary_dc = getStationaryFromFile("stationaryDistrib/SH/Subset_Analysis/stationary_DC.txt")
    #stationary_dd = getStationaryFromFile("stationaryDistrib/SH/Subset_Analysis/stationary_DD.txt")
    #set_stationary = [stationary_cc, stationary_cd, stationary_dc, stationary_dd]

    #plotMultipleStationary(game, set_strats, set_stationary)