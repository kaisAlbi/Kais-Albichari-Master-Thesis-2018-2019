import numpy as np
from itertools import product
import json
import time


class EGTModel:
    def __init__(self, R, S, T, P, pop_size, beta, nb_signals):
        self.Z = pop_size
        self.beta = beta
        self.nb_signals = nb_signals
        self.R = R
        self.S = S
        self.T = T
        self.P = P

        self.strategies = self.createStrategies()
        self.transition_proba, self.fix_probs, self.stationary = self.stationaryDistrib()

    def getNbSignals(self):
        return self.nb_signals

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

    def createStrategies(self):
        """
        Generate all possible combination of strategies, depending on the number of signals
        :return: list of all strategies
        """
        s = list(list(item) for item in product("CD", repeat=self.getNbSignals()))
        strats = []
        for i in range(self.getNbSignals()):
            for j in range(len(s)):
                ss = list(s[j])
                ss.insert(0, i)
                strats.append(ss)
        return strats

    def getPayoff(self, first, second):
        """
        :param first: first agent
        :param second: second agent
        :return: payoff value an agent obtains after an interaction with another agent
        """
        if first == "C":
            if second == "C":
                return self.R
            else:
                return self.S
        else:
            if second == "C":
                return self.T
            else:
                return self.P

    def fitness(self, k, inv_strat, res_strat):
        """
        This method determines the average payoff of k invaders and Z-k residents in
        the population of Z agents
        :param k: number of invaders
        :param inv_strat: invaders' strategy
        :param res_strat: residents' strategy
        :return: fitness for invaders and for residents
        """
        inv_signal = inv_strat[0]
        res_signal = res_strat[0]
        p_inv_inv = self.getPayoff(inv_strat[inv_signal + 1], inv_strat[inv_signal + 1])
        p_inv_res = self.getPayoff(inv_strat[res_signal + 1], res_strat[inv_signal + 1])
        p_res_res = self.getPayoff(res_strat[res_signal + 1], res_strat[res_signal + 1])
        p_res_inv = self.getPayoff(res_strat[inv_signal + 1], inv_strat[res_signal + 1])

        result_inv = (k * p_inv_inv + (self.Z - k) * p_inv_res) / float(self.Z)
        result_res = (k * p_res_inv + (self.Z - k) * p_res_res) / float(self.Z)
        return [result_inv, result_res]

    def fermiDistrib(self, first_fitness, second_fitness, positive):
        """
        :param first_payoff: payoff obtained by the first agent after the interaction
        :param second_payoff: payoff obtained by the second agent after the interaction
        :param positive: boolean value used to calculate probability increase and decrease (T +-)
        :return: probability that the first agent imitates the second ond
        """
        if positive:
            return 1. / (1. + np.exp(-self.getBeta() * (first_fitness - second_fitness)))
            #return 1./(1. + np.exp(self.getBeta()*(second_fitness-first_fitness)))     #Francisco's fermi distribution
        else:
            return 1. / (1. + np.exp(self.getBeta() * (first_fitness - second_fitness)))
            #return 1./(1. + np.exp(-self.getBeta()*(second_fitness-first_fitness)))

    def probIncDec(self, k, inv_strat, res_strat):
        """
        :param k: number of invaders
        :param inv_strat: invaders' strategy
        :param res_strat: residents' strategy
        :return: probability to change the number of k invaders (by +- one at each time step)
        """
        fitness_val = self.fitness(k, inv_strat, res_strat)
        inc = ((self.Z - k) * k * self.fermiDistrib(fitness_val[1], fitness_val[0], False)) / float(self.Z)
        dec = ((self.Z - k) * k * self.fermiDistrib(fitness_val[1], fitness_val[0], True)) / float(self.Z)
        return [inc, dec]

    def fixationProba(self, inv_strat, res_strat):
        """
        :param inv_strat: invaders' strategy
        :param res_strat: residents' strategy
        :return: fixation probability of the invader in a population of residents
        """
        result = 0.
        for i in range(0, self.Z):
            mul = 1.
            for j in range(1, i + 1):
                inc, dec = self.probIncDec(j, inv_strat, res_strat)
                lambda_j = dec / float(inc)
                mul *= lambda_j
            result += mul
        return np.clip(1. / result, 0., 1.)

    def transitionMatrix(self):
        """
        :return: transition matrix of all possible transitions between strategies
        """
        strats = self.getStrategies()
        n = len(strats)
        norm_fact = 1 / float((n - 1))
        drift = 1.0 / self.Z  # Random drift between strategies
        fix_probs = np.zeros((n, n))
        transitions = np.zeros((n, n))
        for i in range(n):
            res_strat = strats[i]
            transitions[i, i] = 1
            for j in range(n):
                inv_strat = strats[j]
                if i != j:
                    f_proba = self.fixationProba(inv_strat, res_strat)
                    fix_probs[i, j] = f_proba
                    trans_value = f_proba * norm_fact
                    transitions[i, j] = trans_value
                    transitions[i, i] -= trans_value
        return [transitions, fix_probs]

    def stationaryDistrib(self):
        """
        Calculate the transition matrix, and based on that matrix, the stationary distribution of each strategy
        :return: transition matrix, stationary distribution
        """
        start_time = time.time()
        t, f = self.transitionMatrix()
        print("The transition and fixation proba matrices calculation took --- %s seconds ---" % (time.time() - start_time))
        n = len(self.getStrategies())
        norm_fact = 1 / float(n-1)
        #new_t = np.zeros((n, n))
        #for i in range (len(f)):
        #    new_t[i, i] = 1
        #    for j in range (len(f[0])):
        #        if i != j:
        #            trans = f[i,j]/ norm_fact
        #            new_t[i,j] = trans
        #            new_t[i,i] -= trans

        start_time = time.time()
        val, vect = np.linalg.eig(t.transpose())
        j_stationary = np.argmin(abs(val - 1.0))  # look for the element closest to 1 in the list of eigenvalues
        p_stationary = abs(vect[:, j_stationary].real)  # the, is essential to access the matrix by column
        p_stationary /= p_stationary.sum()  # normalize
        print("The stationary distribution calculation took --- %s seconds ---" % (
                    time.time() - start_time))
        return t, f, p_stationary


def displayStrats(strats):
    """
    Displays the  strategies
    """
    for i in range(len(strats)):
        if i == 0:
            print("1st ", end=" ")
        elif i == 1:
            print("2nd ", end=" ")
        elif i == 2:
            print("3rd ", end=" ")
        else:
            print(i + 1, "th ", end=" ")
        print(strats[i], end=" ")
        if i != len(strats) - 1:
            print(", ", end=" ")
        else:
            print()


def displayTransMat(transition_matrix):
    for i in range(len(transition_matrix)):
        print("[", end="")
        for j in range(len(transition_matrix[i])):
            if j != len(transition_matrix[i]) - 1:
                print(round(transition_matrix[i][j], 7), end=", ")
            else:
                print(round(transition_matrix[i][j], 7), end="")
        print("]")


def displayTransitionAndStationaryDistrib(egt_model):
    """
    Displays the obtained strategies with the corresponding transition probabilities matrix
    and the strationary distribution
    """
    t, f, p_stationary = egt_model.stationaryDistrib()
    strats = egt_model.getStrategies()
    print("fixation probabilities")
    displayStrats(strats)
    print("Read as row invading column")
    displayTransMat(f)
    print("transition matrix")
    displayStrats(strats)
    print("Read as row invading column")
    displayTransMat(t)

    print("Stationary distribution")
    strats = egt_model.getStrategies()
    for i in range(len(strats)):
        print(strats[i], " : ", round(p_stationary[i], 3))

def save(filename, egt_model):
    f = open(filename, "w")
    all = [egt_model.getTransitionMatrix().tolist(), egt_model.getFixProbs().tolist(), egt_model.getStrategies(), egt_model.getStationaryDistrib().tolist()]
    json.dump(all, f)
    f.close()

def load(filename):
    f = open(filename, "r")
    transition_matrix, fix_probs, strats, stationary_dist = json.load(f)
    return transition_matrix, fix_probs, strats, stationary_dist


if __name__ == '__main__':
    R, S, T, P = 1, -0.5, 0.5, 0       #SH first config
    Z = 150 #Population size
    beta = 0.05

    start_time = time.time()
    #egt_1sig = EGTModel(R,S,T,P,Z,beta,1)
    egt_2sig = EGTModel(R,S,T,P,Z,beta,2)



    #save("test_egt_1sig.txt", egt_1sig)
    #save("test_egt_2sig.txt", egt_2sig)

    #print("1 signal")
    #displayTransitionAndStationaryDistrib(egt_1sig)
    print("computation took --- %s seconds---" % (time.time() - start_time))
    print("\n2 signals")

    displayTransitionAndStationaryDistrib(egt_2sig)