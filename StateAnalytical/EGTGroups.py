import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pickle
import networkx as nx
from matplotlib.lines import Line2D

class EGTModel:
    def __init__(self, fit_diff_filename, pop_size, group_size, beta, nb_states):
        self.Z = pop_size
        self.N = group_size
        self.beta = beta
        self.nb_states = nb_states
        strategies = self.createStrategies()
        self.strategies = self.reducedStrategies(strategies)
        self.fit_diff_matrix = self.loadFitDiff(fit_diff_filename)

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


    def loadFitDiff(self, filename):
        f = open(filename, "rb")
        fit_diff_matrix = np.asarray(pickle.load(f,encoding='latin1'))
        f.close()
        return fit_diff_matrix

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

    def reducedStrategies(self, strategies):

        all_c = strategies[0]
        all_d = all_c[:2] + strategies[-1][2:]
        reduced_strats = [all_c]
        for strat in strategies:
            if not (hasOnlyOneAction(strat) or hasOnlyOneDirection(strat)):
                reversed_strat = []
                for i in range(2):
                    reversed_strat.append((strat[i] + 1) % 2)  # Bit flip for transitions : LR -> RL
                for i in range(len(strat) - 1, 1, -1):  # Reverse actions : CDD -> DDC
                    reversed_strat.append(strat[i])
                if reversed_strat not in reduced_strats:
                    reduced_strats.append(strat)
        reduced_strats.append(all_d)
        return reduced_strats

    def fermiDistrib(self, fit_diff, increase):
        """
        :param first_payoff: payoff obtained by the first agent after the interaction
        :param second_payoff: payoff obtained by the second agent after the interaction
        :param positive: boolean value used to calculate probability increase and decrease (T +-)
        :return: probability that the first agent imitates the second ond
        """
        if increase:
            #print("A - B : ", fit_diff[0])
            return 1. / (1. + np.exp(-self.getBeta() * fit_diff))
        else:
            #print("A + B : ", fit_diff[1])
            return 1. / (1. + np.exp(self.getBeta() * fit_diff))


    def probIncDec(self, n_A, fit_diff):
        """
        :param n_A: number of invaders
        :param inv_strat: invaders' strategy
        :param res_strat: residents' strategy
        :return: probability to change the number of k invaders (by +- one at each time step)
        """
        tmp = ((self.Z - n_A) / self.Z) * (n_A / self.Z)
        inc = np.clip(tmp * self.fermiDistrib(fit_diff, True), 0., 1.)
        dec = np.clip(tmp * self.fermiDistrib(fit_diff, False), 0., 1.)
        return [inc, dec]


    def fixationProba(self, res_index, inv_index):
        """
        :param res_index: resident index
        :param inv_index: invader index
        :return: fixation probability of the invader in a population of residents
        """

        fit_diff_array = self.fit_diff_matrix[res_index, inv_index]
        #print("fit diff array : ", fit_diff_array)
        result = 0.
        for i in range(0, self.Z):
            #if i > 0:
                #print("fit_diff_array[i-1] : ", fit_diff_array[i-1])
            mul = 1.
            for j in range(1, i + 1):
                inc, dec = self.probIncDec(j, fit_diff_array[j-1])
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
        norm_fact = 1 / float((n - 1))
        fix_probs = np.zeros((n, n))
        transitions = np.zeros((n, n))
        for i in range(n):
            transitions[i, i] = 1
            for j in range(n):
                if i != j:
                    f_proba = self.fixationProba(j,i)
                    fix_probs[i, j] = f_proba
                    trans_value = f_proba * norm_fact
                    transitions[i, j] = trans_value
                    transitions[i, i] -= trans_value
        print("transition matrix computed")
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

def showStationaryDistrib(game, N, strats, stationary):
    """
    Plot a bar graph showing the stationary distribution
    :param game: evolutionary game being played
    :param strats: array of binary strategies
    :param stationary: stationary distribution array
    """
    n = len(strats)
    x = [i for i in range(n)]

    fig = plt.figure()
    plt.title(game + " - group size : " + str(N))
    #plt.title(game + " - any group size")

    ax = fig.add_subplot(111)
    ax.set_ylabel("stationary distribution")

    plt.xticks(x, makeXTicks(strats), rotation='vertical')
    for i in range (n):
        plt.bar(x[i], stationary[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
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
    print("fix_probs : ")
    print(fix_probs)
    transition = egt_model.getTransitionMatrix()
    print("transition : ")
    print(transition)
    stationary = egt_model.getStationaryDistrib()
    drift = 1 / egt_model.getPopSize()
    print("drift = ", drift)
    print("2drift = ", 2 * drift)
    print("4drift = ", 4 * drift)
    print("6drift = ", 6 * drift)
    G = nx.DiGraph(directed=True)
    #G = nx.cycle_graph(len(strats_mat))
    ncolors = getColorsForStrats(strats_mat)
    G.add_nodes_from(strats)
    for j in range(n):
        for i in range(n):
            #print(fix_probs[i, j])
            if fix_probs[i, j] > drift:
                print("ok")
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
    for (u,v,d) in G.edges(data=True):
        print(u)
        print(v)
        print(d)
    eselect_4 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > 4 * drift)]
    eselect_3 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > 3 * drift and (u, v) not in eselect_4)]
    eselect_2 = [(u, v) for (u, v, d) in G.edges(data=True) if
                          (d['weight'] > 2 * drift and (u,v) not in eselect_4 and (u,v) not in eselect_3)]
    eselect_1 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > drift and (u,v) not in eselect_4 and (u,v) not in eselect_3 and (u,v) not in eselect_2)]

    eselect_labels = dict(((u, v), float("{0:.6f}".format(d['weight'])))
                          for (u, v, d) in G.edges(data=True))

    #edrift = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 1.0]
    print("eselect")
    print(eselect_4)
    #print(eselect_2)
    #print(eselect_1)
    print("eselect_labels")
    print(eselect_labels)
    print("edrift")
    #print(edrift)

    fig = plt.figure(figsize=(10, 10), dpi=150)
    plt.title(game)
    nodes_labels = {}
    for i in range(len(strats)):
        nodes_labels["".join(map(str, strats[i]))] = "".join(
            map(str, strats[i] + "\n" + str(round(stationary[i] * 100, 2)) + "%"))
    print(strats)
    pos = {strats[0]: np.array([2,2]), strats[1]: np.array([0,0]), strats[2]: np.array([4,0]), strats[3]: np.array([2,-2])}
    #pos = {strats[0]: np.array([5, 25]), strats[1]: np.array([10, 25]), strats[2]: np.array([15, 25]), strats[3]: np.array([20, 25]),
    #       strats[4]: np.array([0, 20]), strats[5]: np.array([-2, 15]), strats[6]: np.array([-2, 10]) ,strats[7]: np.array([0, 5]),
    #       strats[8]: np.array([25, 20]), strats[9]: np.array([27, 15]), strats[10]: np.array([27, 10]), strats[11]: np.array([25, 5]),
    #       strats[12]: np.array([5, 0]), strats[13]: np.array([10, 0]), strats[14]: np.array([15, 0]),strats[15]: np.array([20, 0])}
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=ncolors, with_labels=True)

    # edges
    """PD
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_6,
                                            width=4, arrowsize=40, edge_color="black")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_4,
                                            width=3, arrowsize=30, edge_color="grey")
    """

    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_3,
                                            width=3, arrowsize=60, edge_color="grey")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_2,
                                            width=2, arrowsize=50, edge_color="blue")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_1,
                                            width=0.5, arrowsize=40, edge_color="orange")



    # node labels

    nx.draw_networkx_labels(G, pos, nodes_labels,
                            font_size=10, font_weight='bold', font_color='white')

    # edge labels
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=eselect_labels, font_size=14)
    #plt.legend(numpoints=1)

    #SH
    legend_elements = [Line2D([0], [0], marker='$\leftarrow$', color='orange', lw=1, label='$>'+str("\\")+'rho_N$',
                              markerfacecolor='orange', markersize=25),
                        Line2D([0], [0], marker='$\leftarrow$', color='blue', lw=1, label='$>2' + str("\\") + 'rho_N$',
                              markerfacecolor='blue', markersize=25),

                       Line2D([0], [0], marker='$\leftarrow$', color='grey', lw=1, label='$>3' + str("\\") + 'rho_N$',
                              markerfacecolor='grey', markersize=25)
                       ]


    """PD
    legend_elements = [Line2D([0], [0], marker='$\leftarrow$', color='grey',lw=1, label='$>4'+str("\\")+'rho_N$',
                              markerfacecolor='grey', markersize=25),
                       Line2D([0], [0], marker='$\leftarrow$', color='black', lw=1, label='$>6' + str("\\") + 'rho_N$',
                              markerfacecolor='black', markersize=25)
                       ]
    """
    plt.legend(handles=legend_elements, loc='best', frameon=False)

    plt.axis('off')
    plt.show()  # display


#    print(strats[i], " : ", round(stationary[i], 8))





if __name__ == '__main__':

    game = "PD"
    Z = 150
    N = 150

    beta = 0.05
    nb_states = 2
    #fit_diff_filename = "FitDiff/PD/Reduced/T15/fit_diff_1st_groupsize150.pickle"

    #fit_diff_filename = "FitDiff/" + game + "/analytical_groups/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"
    fit_diff_filename = "FitDiff/" + game + "/Reduced/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"
    #fit_diff_filename = "FitDiff/" + game + "/analytical_groups/fit_diff_LRCD_RLCD_groupsize" + str(N) + ".pickle"
    #fit_diff_filename = "FitDiff/" + game + "/test/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"

    egt_2states = EGTModel(fit_diff_filename, Z, N, beta, nb_states)
    stationary = egt_2states.getStationaryDistrib()
    strats = egt_2states.getStrategies()
    # showTransitionForStrategies(game, egt_2states)
    for i in range(len(strats)):
        #print(strats[i], " : ", round(stationary[i], 8))
        print(strats[i], " : ", stationary[i])
    showTransitionForStrategies(game, egt_2states)
    #print(game, " - N=", N, " - Z=", Z)
    #showStationaryDistrib(game, N, strats, stationary)



    """
    for nb_states in [2]:#[1,2]:
        N_array = [1,2,6,10,20,75,150]
        for n in N_array:
            fit_diff_filename = "FitDiff/" + game + "/Reduced/fit_diff_" + str(nb_states) + "st_groupsize" + str(
                n) + ".pickle"
            egt_states = EGTModel(fit_diff_filename, Z, n, beta, nb_states)
            stationary = egt_states.getStationaryDistrib()
            strats = egt_states.getStrategies()
            stationary_filename = "stationaryDistrib/"+game+"/Groups/Reduced/"+str(nb_states)+"states/stationary_" + str(nb_states) + "st_"+str(n)+"groupsize.txt"
            if nb_states == 1:
                stationary_filename = "stationaryDistrib/"+game+"/Groups/Reduced/"+str(nb_states)+"state/stationary_" + str(nb_states) + "st_"+str(n)+"groupsize.txt"
            for i in range(len(strats)):
                print(strats[i], " : ", round(stationary[i], 8))
            storeStationary(stationary_filename, strats, stationary)
            #showStationaryDistrib(game, n, strats, stationary)
    """
    """
    list_nb_states = [1]
    t_folder = ["T0", "T025", "T05", "T075", "T1"]
    for nb_states in list_nb_states:
        for folder in t_folder:
            fit_diff_filename = "FitDiff/SH/"+folder+"/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"
            #fit_diff_filename = "FitDiff/SH/Reduced/" + folder + "/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"
            egt_states = EGTModel(fit_diff_filename, Z, N, beta, nb_states)
            stationary = egt_states.getStationaryDistrib()
            strats = egt_states.getStrategies()
            stationary_filename = "stationaryDistrib/SH/Groups/" + folder + "/stationary_" + str(nb_states) + "st.txt"

            storeStationary(stationary_filename, strats, stationary)

        t_folder = ["T1", "T125", "T15", "T175", "T2"]
        for folder in t_folder:
            fit_diff_filename = "FitDiff/PD/" + folder + "/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"
            #fit_diff_filename = "FitDiff/PD/Reduced/" + folder + "/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"
            egt_states = EGTModel(fit_diff_filename, Z, N, beta, nb_states)
            stationary = egt_states.getStationaryDistrib()
            strats = egt_states.getStrategies()
            stationary_filename = "stationaryDistrib/PD/Groups/" + folder + "/stationary_" + str(nb_states) + "st.txt"

            storeStationary(stationary_filename, strats, stationary)
    """
