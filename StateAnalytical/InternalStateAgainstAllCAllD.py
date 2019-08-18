import numpy as np
import time
import matplotlib.pyplot as plt
from TreeStrategy import Tree

strategies = ["AllC", "AllD", "LRCD"]
nb_strats = len(strategies)
game = "PD"
R, P = 1, 0
T = 1.5
S = -0.5

beta = 0.05
rounds = 10
Z = 150
nb_states = 2
trans_states_proba = [0.8, 0.8]


def modifyStratFormat(strat_1):
    modified_strat = []
    for i, item in enumerate(strat_1):
        if i < 2:
            if item == "L":
                modified_strat.append(1)
            else:
                modified_strat.append(0)
        else:
            if item == "C":
                modified_strat.append(1)
            else:
                modified_strat.append(0)
    return modified_strat


def computeProbaCPairsForState(strategies, init_state, nb_states, rounds, trans_states_proba):
    n = len(strategies)

    proba_c_pairs = np.zeros((n,n))
    for i in range (n):
        strat_1 = strategies[i]
        if strat_1 == "AllC":
            for j in range (n):
                proba_c_pairs[i, j] = 1
        elif strat_1 == "AllD":
            for j in range(n):
                proba_c_pairs[i, j] = 0
        else:
            strat_1 = modifyStratFormat(strat_1)
            for j in range(n):
                if i == j:
                    proba_c_pairs[i, i] = computeSingleProbaForState(strat_1, init_state, nb_states, rounds, trans_states_proba)
                else:
                    proba_j = 1
                    if strategies[j] == "AllD":
                        proba_j = 0
                    root_1 = Tree(init_state, strat_1, 1, nb_states - 1, trans_states_proba)
                    cur_prob_c1 = buildTreeOptimised(proba_j, root_1, rounds)
                    proba_i = cur_prob_c1

                    proba_c_pairs[i, j] = proba_i
    return proba_c_pairs



def computeProbaCPairs(strategies, nb_states, rounds, trans_states_proba):
    n = len(strategies)

    proba_c_pairs = np.zeros((n,n))
    for i in range (n):
        strat_1 = strategies[i]
        if strat_1 == "AllC":
            for j in range (n):
                proba_c_pairs[i, j] = 1
        elif strat_1 == "AllD":
            for j in range(n):
                proba_c_pairs[i, j] = 0
        else:
            strat_1 = modifyStratFormat(strat_1)
            for j in range(n):
                if i == j:
                    proba_c_pairs[i, i] = computeSingleProba(strat_1, nb_states, rounds, trans_states_proba)
                else:
                    proba_i = 0
                    proba_j = 1
                    if strategies[j] == "AllD":
                        proba_j = 0
                    for state in range(nb_states):
                        root_1 = Tree(state, strat_1, 1, nb_states - 1, trans_states_proba)
                        cur_prob_c1 = buildTreeOptimised(proba_j, root_1, rounds)
                        proba_i += cur_prob_c1

                        proba_i /= nb_states
                    proba_c_pairs[i, j] = proba_i
    return proba_c_pairs


def computeSingleProbaForState(strat, init_state,  nb_states, rounds, trans_states_proba):
    init_weight = 1
    proba_c = 0
    root = Tree(init_state, strat, init_weight, nb_states - 1, trans_states_proba)
    proba_c_current = root.getProbaC()
    root.addChildren(proba_c_current)
    current_level = root.getChildren()
    for i in range(rounds - 1):
        proba_c_current = sum([node.getProbaC() for node in current_level])

        next_level = []
        for child in current_level:
            child.addChildren(proba_c_current)
            for new_child in child.getChildren():
                next_level.append(new_child)
        current_level = list(next_level)
    proba_c += np.clip(sum([node.getProbaC() for node in next_level]), 0., 1.)
    return proba_c


def computeSingleProba(strat, nb_states, rounds, trans_states_proba):
    """
    Build a decision tree for a strategy with itself
    :param strat: current strategy
    :param nb_states: number of available internal states values
    :param rounds: depth of trees
    :param trans_states_proba: array of probabilities of transitions between states
    :return: probability of a strategy cooperating with itself
    """
    init_weight = 1
    proba_c = 0
    for state in range(nb_states):
        # Computes the probability of a strategy cooperating with itself, for the same initial state value
        root = Tree(state, strat, init_weight, nb_states - 1, trans_states_proba)
        proba_c_current = root.getProbaC()
        root.addChildren(proba_c_current)
        current_level = root.getChildren()
        for i in range(rounds - 1):
            proba_c_current = sum([node.getProbaC() for node in current_level])

            next_level = []
            for child in current_level:
                child.addChildren(proba_c_current)
                for new_child in child.getChildren():
                    next_level.append(new_child)
            current_level = list(next_level)
        proba_c += np.clip(sum([node.getProbaC() for node in next_level]), 0., 1.)
        del root

        for state_2 in range (state+1, nb_states):
            # Computes the probability of a strategy cooperating with itself, for different initial state value
            root_1 = Tree(state, strat, init_weight, nb_states - 1, trans_states_proba)
            root_2 = Tree(state_2, strat, init_weight, nb_states - 1, trans_states_proba)
            proba_c_current = root_1.getProbaC()
            proba_c2 = root_2.getProbaC()
            root_1.addChildren(proba_c2)
            root_2.addChildren(proba_c_current)
            current_level_1 = root_1.getChildren()
            current_level_2 = root_2.getChildren()
            del root_2
            del root_1
            for i in range(rounds - 1):
                proba_c1 = sum([node.getProbaC() for node in current_level_1])
                proba_c2 = sum([node.getProbaC() for node in current_level_2])

                next_level_1 = []
                for child in current_level_1:
                    child.addChildren(proba_c2)
                    for new_child in child.getChildren():
                        next_level_1.append(new_child)

                next_level_2 = []
                for child in current_level_2:
                    child.addChildren(proba_c1)
                    for new_child in child.getChildren():
                        next_level_2.append(new_child)

                for elem in current_level_1:
                    del elem
                for elem in current_level_2:
                    del elem
                current_level_1 = list(next_level_1)
                current_level_2 = list(next_level_2)
            proba_c += sum([node.getProbaC() for node in next_level_1])

            proba_c += sum([node.getProbaC() for node in next_level_2])

    return proba_c / np.power(nb_states, 2)


def buildTreeOptimised(proba_c_other, root, rounds):
    """
    Build one decision tree. Case where one of the two strategies has a fixed probability of cooperating
    :param proba_c_other: fixed probability of the other strategy
    :param root: root of the decision tree we want to build
    :param rounds: depth of the tree
    :return: probability of cooperation of strategy represented by 'root' when playing against some other strategy with a fixed probability of cooperation
    """
    root.addChildren(proba_c_other)
    current_level = root.getChildren()
    del root
    for i in range(rounds - 1):

        next_level = []
        for child in current_level:
            child.addChildren(proba_c_other)
            for new_child in child.getChildren():
                next_level.append(new_child)

        for elem in current_level:
            del elem
        current_level = list(next_level)
    proba_c = sum([node.getProbaC() for node in next_level])
    for elem in next_level:
        del elem

    return np.clip(proba_c, 0., 1.)


def getPayoff(R, S, T, P, proba_c1, proba_c2):
    """
    :param proba_c1: probability of first agent cooperating
    :param proba_c2: probability of second agent cooperating
    :return: payoff value an agent obtains after an interaction with another agent
    """
    payoff = 0
    payoff += np.float64(R * proba_c1 * proba_c2)
    payoff += np.float64(S * proba_c1 * (1 - proba_c2))
    payoff += np.float64(T * (1 - proba_c1) * proba_c2)
    payoff += np.float64(P * (1 - proba_c1) * (1 - proba_c2))
    return payoff

def computePayoffsPairs(R, S, T, P, n, proba_couples):
    """
    Computes a matrix of analytical payoffs for all pairs of strategies
    :param R: mutual cooperation payoff
    :param S:  payoff when the considered strategy cooperated while the other defected
    :param T: temptation to defect - payoff when the considered strategy defected while the other cooperated
    :param P: mutual defection payoff
    :param n: number of strategies
    :param proba_couples: matrix of probability of cooperation of strategies against other strategies and themselves
    :return: matrix of analytical payoffs
    """
    payoff_pairs = np.zeros((n,n))
    for i in range (n):
        proba_i_i = proba_couples[i,i]
        payoff_pairs[i,i] = getPayoff(R, S, T, P, proba_i_i, proba_i_i)

        for j in range (i+1, n):
            proba_i_j = proba_couples[i, j]
            proba_j_i = proba_couples[j, i]
            payoff_pairs[i, j] = getPayoff(R, S, T, P, proba_i_j, proba_j_i)
            payoff_pairs[j, i] = getPayoff(R, S, T, P, proba_j_i, proba_i_j)
    return payoff_pairs


def fermiDistrib(first_fitness, second_fitness, positive):
    """
    :param first_payoff: payoff obtained by the first agent after the interaction
    :param second_payoff: payoff obtained by the second agent after the interaction
    :param positive: boolean value used to calculate probability increase and decrease (T +-)
    :return: probability that the first agent imitates the second ond
    """
    if positive:
        return 1. / (1. + np.exp(-beta * (first_fitness - second_fitness)))
    else:
        return 1. / (1. + np.exp(beta * (first_fitness - second_fitness)))


def probIncDec(n_A, inv_strat, res_strat, payoffs):
    """
    :param n_A: number of invaders
    :param inv_strat: invaders' strategy
    :param res_strat: residents' strategy
    :return: probability to change the number of k invaders (by +- one at each time step)
    """

    p_inv_inv = payoffs[inv_strat, inv_strat]
    p_inv_res = payoffs[inv_strat, res_strat]
    p_res_inv = payoffs[res_strat, inv_strat]
    p_res_res = payoffs[res_strat, res_strat]
    result_inv = (((n_A) * p_inv_inv) + ((Z - n_A) * p_inv_res)) / float(Z)
    result_res = ((n_A * p_res_inv) + ((Z - n_A) * p_res_res)) / float(Z)

    inc = np.clip(((Z - n_A) * (n_A / Z) * fermiDistrib(result_res, result_inv, False)) / Z, 0., 1.)
    dec = np.clip(((Z - n_A) * (n_A / Z) * fermiDistrib(result_res, result_inv, True)) / Z, 0., 1.)
    return [inc, dec]


def fixationProba(res_strat, inv_strat, payoffs):
    """
    :param res_index: resident index
    :param inv_index: invader index
    :return: fixation probability of the invader in a population of residents
    """

    result = 0.
    for i in range(0, Z):
        mul = 1.
        for j in range(1, i + 1):
            inc, dec = probIncDec(j, inv_strat, res_strat, payoffs)
            lambda_j = np.float(dec / float(inc))
            mul *= lambda_j
        result += mul

    return np.clip(1. / result, 0., 1.)


def transitionMatrix(strats, payoffs):
    """
    Compute the fixation probability for each pair invader-resident of strategies and build the fixation probabilities
    matrix and the transition matrix
    :return: transition matrix and fixation probabilities matrix
    """
    n = len(strats)
    norm_fact = 1 / float((n - 1))
    fix_probs = np.zeros((n, n))
    transitions = np.zeros((n, n))
    for i in range(n):
        start_time = time.time()
        transitions[i, i] = 1
        for j in range(n):
            if i != j:
                f_proba = fixationProba(i, j, payoffs)
                fix_probs[i, j] = f_proba
                trans_value = f_proba * norm_fact
                transitions[i, j] = trans_value
                transitions[i, i] -= trans_value
        print("transitions values calculations for resident strat ", strats[i],
              " took --- %s seconds---" % (time.time() - start_time))
    return [transitions, fix_probs]


def stationaryDistrib(strats, payoffs):
    """
    Calculate the transition matrix, and based on that matrix, the stationary distribution of each strategy
    :return: transition matrix, fixation probabilities matrix, stationary distribution
    """
    t, f = transitionMatrix(strats, payoffs)
    val, vect = np.linalg.eig(t.transpose())
    j_stationary = np.argmin(abs(val - 1.0))  # look for the element closest to 1 in the list of eigenvalues
    p_stationary = abs(vect[:, j_stationary].real)  # the, is essential to access the matrix by column
    p_stationary /= p_stationary.sum()  # normalize
    return t, f, p_stationary


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

    plt.xticks(x, strats)
    for i in range (n):
        plt.bar(x[i], stationary[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

#proba_c_pairs = computeProbaCPairs(strategies, nb_states, rounds, trans_states_proba)
proba_c_pairs = computeProbaCPairsForState(strategies, 0, nb_states, rounds, trans_states_proba)

payoffs = computePayoffsPairs(R, S, T, P, nb_strats, proba_c_pairs)
t, f, stationary = stationaryDistrib(strategies, payoffs)

showStationaryDistrib(game, strategies, stationary)