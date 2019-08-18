import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import networkx as nx
from matplotlib.lines import Line2D
from StrategyBuilder import loadStrategies, displayableStrat, hasOnlyOneDirection, hasOnlyOneAction


class EmpathyModel:
    def __init__(self, fit_diff_filename, strats, pop_size, beta, nb_states):
        self.Z = pop_size
        self.beta = beta
        self.nb_states = nb_states
        self.strategies = strats
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
        """
        Load a matrix containing every fitness difference for every pair of strategies
        :param filename: name of the file in which the matrix lies
        :return: matrix containing fitness differences for every pair of strategies
        """
        f = open(filename+".pickle", "rb")
        fit_diff_matrix = np.asarray(pickle.load(f,encoding='latin1'))
        f.close()
        return fit_diff_matrix


    def fermiDistrib(self, fit_diff, increase):
        """
        :param fit_diff: difference of the fitness values of the first and the second agents considered
        :param increase: boolean value used to calculate probability increase and decrease (T +-)
        :return: probability that the first agent imitates the second ond
        """
        if increase:
            return 1. / (1. + np.exp(-self.getBeta() * fit_diff))
        else:
            return 1. / (1. + np.exp(self.getBeta() * fit_diff))


    def probIncDec(self, n_A, fit_diff):
        """
        :param n_A: number of invaders
        :param fit_diff: difference of the fitness values of the first and the second agents considered
        :return: probability to change the number of n_A invaders (by +- one at each time step)
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
        result = 0.
        for i in range(0, self.Z):
            mul = 1.
            for j in range(1, i + 1):
                inc, dec = self.probIncDec(j, fit_diff_array[j-1])
                lambda_j = np.float(dec / float(inc))
                mul *= lambda_j
            result += mul

        return np.clip(1. / result, 0., 1.)

    def transitionMatrix(self):
        """
        Compute the fixation probability for each pair invader-resident of strategies and build the fixation
        probabilities matrix and the transition matrix
        :return: transition matrix and fixation probabilities matrix
        """
        strats = self.getStrategies()
        n = len(strats)
        norm_fact = 1 / float((n - 1))
        fix_probs = np.zeros((n, n))
        transitions = np.zeros((n, n))
        for i in range(n):
            start_time = time.time()
            transitions[i, i] = 1
            for j in range(n):
                if i != j:
                    f_proba = self.fixationProba(j, i)
                    fix_probs[i, j] = f_proba
                    trans_value = f_proba * norm_fact
                    transitions[i, j] = trans_value
                    transitions[i, i] -= trans_value

            print("transitions values calculations for resident strat ", strats[i], " (strat ", i+1, "/", len(strats),") took --- %s seconds---" % (time.time() - start_time))
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
    :param N: number of rounds for which the game was played
    :param strats: array of binary strategies
    :param stationary: stationary distribution array
    """
    n = len(strats)
    x = [i for i in range(n)]
    fig = plt.figure()
    plt.title(game + " - " + str(N) + " rounds")
    #plt.title(game + " - any number of rounds")

    ax = fig.add_subplot(111)
    ax.set_ylabel("stationary distribution")
    strats_ticks = [displayableStrat(strat) for strat in strats]
    plt.xticks(x, strats_ticks, rotation='vertical')
    for i in range (n):
        plt.bar(x[i], stationary[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

def getColorsForStrats(strats):
    """
    Associate a color to a strategy, depending on whether it is always cooperative (blue), always defective (red),
    or can be both (purple)
    :param strats: considered strategy
    :return: color associated to this strategy
    """
    colors = []
    for i in range(len(strats)):
        if hasOnlyOneAction(strats[i]):
            if strats[i][1][0][0] == 1:
                colors.append("blue")
            else:
                colors.append("red")
        else:
            if hasOnlyOneDirection(strats[i]):
                if strats[i][0][0] == 1:
                    if strats[i][1][0][0] == 1:
                        colors.append("blue")
                    else:
                        colors.append("red")
                else:
                    if strats[i][1][-1][0] == 1:
                        colors.append("blue")
                    else:
                        colors.append("red")
            else:
                colors.append("purple")
    return colors


def showTransitionForStrategies(game, egt_model):
    """
    Plot a node graph in which arrows A->B denote that a mutant B in a population of As will fixate with a probability
    of at least 1/Z (Z = population size)
    :param game: considered game
    :param egt_model: object containing an analysed model
    """
    strats_mat = egt_model.getStrategies()
    n = len(strats_mat)
    strats = [displayableStrat(strat) for strat in strats_mat]
    fix_probs = egt_model.getFixProbs()
    stationary = egt_model.getStationaryDistrib()
    drift = 1 / egt_model.getPopSize()

    G = nx.DiGraph(directed=True)
    ncolors = getColorsForStrats(strats_mat)
    G.add_nodes_from(strats)
    for j in range(n):
        for i in range(n):
            if fix_probs[i, j] > drift:
                G.add_edge(strats[i], strats[j], weight=fix_probs[i, j])


    eselect_2 = [(u, v) for (u, v, d) in G.edges(data=True) if
                          (d['weight'] > 2 * drift and (u,v))]
    eselect_1 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > drift and (u,v) not in eselect_2)]

    fig = plt.figure(figsize=(10, 10), dpi=150)
    plt.title(game + " - any number of rounds")
    nodes_labels = {}
    for i in range(len(strats)):
        nodes_labels["".join(map(str, strats[i]))] = "".join(
            map(str, strats[i]))
        print("".join(map(str, strats[i]+ "\n" + str(round(stationary[i] * 100, 2)) + "%")))
    #strats = [LLC, LLD, LRC, LRD, RLC, RLD, RRC, RRD]
    pos = {strats[0]: np.array([0, 5]), strats[1]: np.array([0, 0]), strats[2]: np.array([5, 5]), strats[3]: np.array([5, 0]),
           strats[4]: np.array([10, 5]), strats[5]: np.array([10, 0]), strats[6]: np.array([15, 5]) ,strats[7]: np.array([15, 0])}
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1300, node_color=ncolors, with_labels=True)

    # edges
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_2,
                                            width=2, arrowsize=30, edge_color="blue")
    if game == "SH":

        nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_1,
                                                width=2, arrowsize=10, edge_color="orange")

    # node labels

    nx.draw_networkx_labels(G, pos, nodes_labels,
                            font_size=10, font_weight='bold', font_color='white')


    legend_elements = [Line2D([0], [0], marker='$\leftarrow$', color='blue', lw=1, label='$>2' + str("\\") + 'rho_N$',
                              markerfacecolor='blue', markersize=25)]
    if game == "SH":
        legend_elements = [Line2D([0], [0], marker='$\leftarrow$', color='orange', lw=1, label='$>' + str("\\") + 'rho_N$',
                              markerfacecolor='orange', markersize=25)]


    #plt.legend(handles=legend_elements, loc='best', frameon=False)

    plt.axis('off')
    plt.show()  # display

def getStationaryFromFile(filename):
    """
    Retrieve the stationary distributions values from a file
    :param filename: name of the file in which the stationary distributions are stored
    :return: stationary distributions
    """
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
    """
    Store the stationary distributions together with the associated strategies in a file
    :param filename: name of the file in which the stationary distributions are stored
    :param strats: strategies
    :param stationary: stationary distributions
    """
    with open(filename, "w") as f:
        for i in range(len(strats)):
            line = "".join(map(str, strats[i])) + " : " + str(stationary[i])
            if i < (len(strats) - 1):
                line += "\n"
            f.write(line)


def showMostFavouredStationaryDistrib(game, N, strats, stationary):
    """
    Plot a bar graph showing the stationary distribution
    :param game: evolutionary game being played
    :param N: number of rounds for which the game was played
    :param strats: array of binary strategies
    :param stationary: stationary distribution array
    """
    if nb_states == 2:
        if game == "SH":
            most_favoured_strats_ids = [1, 5, 9, 13, 23, 46, 56, 57, 58, 59]
            colors = ["green"] * 4 + ["orange"] * 2 + ["purple"] * 4
            most_favoured_strats = [strats[i] for i in most_favoured_strats_ids]
            most_favoured_stationary = [stationary[i] for i in most_favoured_strats_ids]
        else:
            most_favoured_strats_ids = [1, 5, 9, 13, 16, 24, 32, 33, 56, 57, 58, 59]
            colors = ["green"] * 4 + ["red", "orange"] * 2 + ["purple"] * 4
            most_favoured_strats = [strats[i] for i in most_favoured_strats_ids]
            most_favoured_stationary = [stationary[i] for i in most_favoured_strats_ids]
        n = len(most_favoured_strats)
        x = [i for i in range(n)]
        fig = plt.figure()
        plt.title(game + " - " + str(N) + " rounds")
        #plt.title(game + " - any number of rounds")

        ax = fig.add_subplot(111)
        ax.set_ylabel("stationary distribution")
        strats_ticks = [displayableStrat(strat) for strat in most_favoured_strats]
        plt.xticks(x, strats_ticks, rotation='vertical')
        for i in range (n):
            plt.bar(x[i], most_favoured_stationary[i], color = colors[i])
        ax.set_ylim(0,0.04)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()
    else:
        showStationaryDistrib(game, N, strats, stationary)


if __name__ == '__main__':
    game = "PD"
    Z = 150
    N = 15

    beta = 0.05
    nb_states = 2

    strats_filename = "Strategies/" + str(nb_states) + "_st_strats"
    strats = loadStrategies(strats_filename)

    #fit_diff_filename = "FitDiff/" + game + "/" + str(nb_states) + "states/fit_diff_" + str(N) + "_rounds_alpha08"


    #empathy_model = EmpathyModel(fit_diff_filename, strats, Z, beta, nb_states)
    #stationary = empathy_model.getStationaryDistrib()
    #strats = empathy_model.getStrategies()
    stationary_fn = "stationaryDistributions/" + game + "/stationary_" + str(nb_states) + "st_" + str(N) + "rounds.txt"
    stationary = getStationaryFromFile(stationary_fn)
    #stationary_fn = "stationaryDistributions/" + game + "/stationary_" + str(nb_states) + "st_" + str(N) + "rounds.txt"
    #storeStationary(stationary_fn, strats, stationary)

    showMostFavouredStationaryDistrib(game, N, strats, stationary)
    #showStationaryDistrib(game, N, strats, stationary)
    #showTransitionForStrategies(game, empathy_model)