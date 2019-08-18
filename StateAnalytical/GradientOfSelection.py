import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pickle

class EGTModel:
    def __init__(self, fit_diff_filename, pop_size, group_size, beta, nb_states):
        self.Z = pop_size
        self.N = group_size
        self.beta = beta
        self.nb_states = nb_states
        self.strategies = self.reducedStrategies(self.createStrategies())
        self.fit_diff = self.loadFitDiff(fit_diff_filename)

        self.gradient_matrix = self.gradientMatrix()

    def getNbStates(self):
        return self.nb_states

    def getStrategies(self):
        return self.strategies

    def getBeta(self):
        return self.beta

    def getPopSize(self):
        return self.Z

    def getGradientMatrix(self):
        return self.gradient_matrix

    def loadFitDiff(self, filename):
        f = open(filename, "rb")
        self.fit_diff_matrix = np.asarray(pickle.load(f,encoding='latin1'))
        f.close()

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
            return 1. / (1. + np.exp(-self.getBeta() * (fit_diff)))
        else:
            return 1. / (1. + np.exp(self.getBeta() * (fit_diff)))


    def probIncDec(self, n_A, fit_diff):
        """
        :param n_A: number of invaders
        :param inv_strat: invaders' strategy
        :param res_strat: residents' strategy
        :return: probability to change the number of k invaders (by +- one at each time step)
        """
        tmp = ((self.Z - n_A) / Z) * (n_A / Z)
        inc = np.clip(tmp * self.fermiDistrib(fit_diff, True), 0., 1.)
        dec = np.clip(tmp * self.fermiDistrib(fit_diff, False), 0., 1.)
        return [inc, dec]



    def gradientOfSelection(self, k, fit_diff):
        inc, dec = self.probIncDec(k, fit_diff)
        return inc - dec


    def gradientArray(self, res_index, inv_index):
        """
        :param res_index: resident index
        :param inv_index: invader index
        :return: fixation probability of the invader in a population of residents
        """

        fit_diff_array = self.fit_diff_matrix[res_index, inv_index]
        fit_diff_solo = self.fit_diff_matrix[inv_index, inv_index,0]
        gradient_array = np.zeros(Z+1)
        for i in range(0, self.Z+1):
            if i != Z:
                gradient_array[i] = self.gradientOfSelection(i, fit_diff_array[i])
            else:
                gradient_array[i] = self.gradientOfSelection(i, fit_diff_solo)
        return gradient_array

    def gradientMatrix(self):
        """
        Compute the fixation probability for each pair invader-resident of strategies and build the fixation probabilities
        matrix and the transition matrix
        :return: transition matrix and fixation probabilities matrix
        """
        strats = self.getStrategies()
        n = len(strats)
        gradient_mat = np.zeros((n,n, self.Z+1))
        for i in range(n):
            for j in range(n):
                if i != j:
                    gradient_mat[i,j] = self.gradientArray(i,j)

        return gradient_mat



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


def plotGradientSelection2states(game, N, egt_states):
    """
    Plot the gradient of selection
    :param game: evolutionary game being played
    :param N: group size
    :param egt_states: system
    """
    gradients_mat = egt_states.getGradientMatrix()
    strats = egt_states.getStrategies()
    n = len(strats)
    x = [i for i in range(len(gradients_mat[0,0]))]
    x = [x[i]/len(x) for i in range (len(x))]
    print(len(x))
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(3,6))
    fig.suptitle(game + " - group size : " + str(N))
    ax1.set_ylabel("gradient of selection (G)")
    ax1.set_xlabel("fraction of invaders (k/Z)")

    ax2.set_ylabel("gradient of selection (G)")
    ax2.set_xlabel("fraction of invaders (k/Z)")

    strat_1 = [1,0,1,0] #LRCD
    index_1 = strats.index(strat_1)
    strat_2 = [0,1,1,0] #RLCD
    index_2 = strats.index(strat_2)
    lstyle1 = ["o", "-", "+"]
    k1=0
    lstyle2 = ["o", "-", "+"]
    k2 = 0
    color = ["blue", "green", "red"]


    for i in range (n):
        res_strat = strats[i]
        if res_strat != strat_1:
            cur_label = "invader " + "".join(displayableStrat(strat_1)) + " - " + "resident " + "".join(
                displayableStrat(strats[i]))
            ax1.plot(x, gradients_mat[index_1, i], lstyle1[k1], label=cur_label, color=color[k1])
            #ax1.plot(x, gradients_mat[index_1, i], label=cur_label, color=color[k1])

            k1 += 1
        if res_strat != strat_2:
            cur_label = "invader " + "".join(displayableStrat(strat_2)) + " - " + "resident " + "".join(
                displayableStrat(strats[i]))
            ax2.plot(x, gradients_mat[index_2, i], lstyle2[k2], label=cur_label, color=color[k2])
            #ax2.plot(x, gradients_mat[index_2, i], label=cur_label, color=color[k2])
            k2 += 1
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    legend1 = ax1.legend(loc='lower left')
    legend2 = ax2.legend(loc='lower left')#,bbox_to_anchor=(1.15,1))

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





if __name__ == '__main__':

    game = "SH"
    Z = 150
    N = 150

    beta = 0.05
    nb_states = 2

    #fit_diff_filename = "FitDiff/" + game + "/analytical_groups/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"
    fit_diff_filename = "FitDiff/" + game + "/Reduced/fit_diff_" + str(nb_states) + "st_groupsize" + str(N) + ".pickle"
    #fit_diff_filename = "FitDiff/" + game + "/analytical_groups/fit_diff_LRCD_RLCD_groupsize" + str(N) + ".pickle"

    egt_states = EGTModel(fit_diff_filename, Z, N, beta, nb_states)
    print(egt_states.getStrategies())

    plotGradientSelection2states(game, N, egt_states)
    #plotGradientSelectionCoopDefect(game, N, egt_states)